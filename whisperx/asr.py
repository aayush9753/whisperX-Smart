"""
Automatic Speech Recognition (ASR) components for WhisperX.

This module provides the core ASR functionality for WhisperX, including:
- The WhisperModel class that extends faster-whisper for batched inference
- A Hugging Face Pipeline wrapper for easier integration
- Functions for loading and configuring the ASR models
"""

import os
from typing import List, Optional, Union
from dataclasses import replace

import ctranslate2
import faster_whisper
import numpy as np
import torch
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions, get_ctranslate2_storage
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from whisperx.types import SingleSegment, TranscriptionResult
from whisperx.vads import Vad, Silero, Pyannote, SileroCustom
from tqdm import tqdm

# List of Indic languages supported by the model
INDIC_LANGUAGES = [
    # ISO 639-1 codes
    "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "sd", "si", "ta", "te", "ur",
    # English names Whisper may return
    "assamese", "bengali", "gujarati", "hindi", "kannada", "malayalam", "marathi", "nepali",
    "odia", "oriya", "punjabi", "panjabi", "sanskrit", "sindhi", "sinhala", "tamil", "telugu", "urdu"
]


def find_numeral_symbol_tokens(tokenizer):
    """
    Find tokens in the tokenizer vocabulary that contain numerals or symbols.
    
    Used for optionally suppressing these tokens during transcription.
    
    Parameters
    ----------
    tokenizer : Tokenizer
        The Whisper tokenizer
        
    Returns
    -------
    List[int]
        List of token IDs that contain numerals or symbols
    """
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens


class WhisperModel(faster_whisper.WhisperModel):
    """
    Extended WhisperModel that provides batched inference for faster-whisper.
    
    This class extends the faster_whisper.WhisperModel to add support for
    batched inference, which can significantly speed up processing of multiple
    audio segments.
    
    Currently only works in non-timestamp mode and with fixed prompts for all
    samples in a batch.
    """

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        encoder_output=None,
    ):
        """
        Generate transcriptions for a batch of audio features.
        
        Parameters
        ----------
        features : np.ndarray
            Batch of log mel spectrograms, shape [batch_size, n_mels, n_frames]
        tokenizer : Tokenizer
            Whisper tokenizer for the current language and task
        options : TranscriptionOptions
            Transcription options including beam size, patience, etc.
        encoder_output : Optional
            Pre-computed encoder output if available
            
        Returns
        -------
        List[str]
            List of transcribed text for each item in the batch
        """
        batch_size = features.shape[0]
        
        # Prepare prompts
        all_tokens = []
        prompt_reset_since = 0
        
        # Handle initial prompt if provided
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
            
        previous_tokens = all_tokens[prompt_reset_since:]
        
        # Get the prompt tokens
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
            hotwords=options.hotwords
        )

        # Encode the audio features if not already done
        if encoder_output is None:
            encoder_output = self.encode(features)

        # Calculate max initial timestamp
        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        # Run the model inference
        result = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            beam_size=options.beam_size,
            patience=options.patience,
            length_penalty=options.length_penalty,
            max_length=self.max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
        )

        # Extract token IDs from the result
        tokens_batch = [x.sequences_ids[0] for x in result]

        # Decode tokens to text
        def decode_batch(tokens: List[List[int]]) -> List[str]:
            """Decode a batch of tokens to text, removing end-of-text tokens."""
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)
        return text

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        """
        Encode audio features using the Whisper encoder.
        
        Parameters
        ----------
        features : np.ndarray
            Log mel spectrogram features, can be a batch [batch_size, n_mels, n_frames]
            or a single sample [n_mels, n_frames]
            
        Returns
        -------
        ctranslate2.StorageView
            Encoded features that can be used for generation
        """
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        
        # Add batch dimension if needed
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
            
        features = get_ctranslate2_storage(features)
        return self.model.encode(features, to_cpu=to_cpu)

class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
        self,
        model: WhisperModel,
        vad,
        vad_params: dict,
        options: TranscriptionOptions,
        tokenizer: Optional[Tokenizer] = None,
        device: Union[int, str, "torch.device"] = -1,
        framework="pt",
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}

    def _forward(self, model_inputs):
        outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        return {'text': outputs}

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params: dict,
        forward_params: dict,
        postprocess_params: dict,
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        verbose: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe audio using the model.
        
        Args:
            audio: Path to audio file or audio array
            batch_size: Batch size for processing
            num_workers: Number of workers for data loading
            language: Language code (auto-detected if None)
            task: Task to perform (transcribe or translate)
            chunk_size: Size of audio chunks in seconds
            print_progress: Whether to print progress
            combined_progress: Whether to show combined progress
            verbose: Whether to print detailed information
            
        Returns:
            List of transcribed segments with timing information
        """
        # Load audio if path is provided
        if isinstance(audio, str):
            audio = load_audio(audio)

        # Pre-process audio based on VAD model type
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks
            
        # Get voice segments from VAD model
        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        
        # If language not specified, detect language for each segment
        if language is None and self.preset_language is None:
            # First merge segments with small gaps for better language detection
            merged_segments = []
            current_segment = None
            merge_threshold = 2  # seconds
            
            for seg in vad_segments:
                if current_segment is None:
                    current_segment = {
                        'start': seg['start'],
                        'end': seg['end'],
                        'original_segments': [seg]
                    }
                else:
                    # If gap is less than threshold seconds
                    if seg['start'] - current_segment['end'] <= merge_threshold:
                        # Merge by extending end time
                        current_segment['end'] = seg['end']
                        current_segment['original_segments'].append(seg)
                    else:
                        # Add current segment to merged list and start a new one
                        merged_segments.append(current_segment)
                        current_segment = {
                            'start': seg['start'],
                            'end': seg['end'],
                            'original_segments': [seg]
                        }

            # Add the last segment if it exists
            if current_segment is not None:
                merged_segments.append(current_segment)
            
            # Detect language for each merged segment
            language_segments = {}
            language_probabilities = {}
            for merged_seg in tqdm(merged_segments, desc="Detecting language for segments"):
                # Extract audio for merged segment
                f1 = int(merged_seg['start'] * SAMPLE_RATE)
                f2 = int(merged_seg['end'] * SAMPLE_RATE)
                segment_audio = audio[f1:f2]
                
                # Detect language for this merged segment
                segment_language, segment_language_probability = self.detect_language(segment_audio)
                if segment_language in ["en", "hi"]:
                    pass
                elif segment_language in INDIC_LANGUAGES:
                    segment_language = "hi"
                    segment_language_probability = 0
                else:
                    segment_language = "en"
                    segment_language_probability = 0
                
                # Assign language to all original segments in this merged segment
                for original_seg in merged_seg['original_segments']:
                    if segment_language not in language_segments:
                        language_segments[segment_language] = []
                        language_probabilities[segment_language] = []
                    
                    language_segments[segment_language].append(original_seg)
                    language_probabilities[segment_language].append(segment_language_probability)
        else:
            # Use specified language for all segments
            language_segments = {language: vad_segments}
            language_probabilities = {language: [1.0] * len(vad_segments)}
        audio_language = max(language_segments.keys(), key=lambda k: len(language_segments[k]))
        audio_language_probability = sum(language_probabilities[audio_language]) / len(language_probabilities[audio_language])
        
        # Set default task if not specified
        task = task or "transcribe"
        
        # Process each language group separately
        all_segments = []
        for lang, segments in language_segments.items():
            # Set up tokenizer for this language
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=lang,
            )
            
            # Handle numeral suppression if enabled
            previous_suppress_tokens = None
            if self.suppress_numerals:
                previous_suppress_tokens = self.options.suppress_tokens
                numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
                print(f"Suppressing numeral and symbol tokens")
                new_suppressed_tokens = list(set(numeral_symbol_tokens + self.options.suppress_tokens))
                self.options = replace(self.options, suppress_tokens=new_suppressed_tokens)

            # Helper function to yield audio segments
            def data(segments_to_process):
                for seg in segments_to_process:
                    f1 = int(seg['start'] * SAMPLE_RATE)
                    f2 = int(seg['end'] * SAMPLE_RATE)
                    yield {'inputs': audio[f1:f2]}

            # Process segments
            lang_segments = []
            batch_size = batch_size or self._batch_size
            total_segments = len(vad_segments)
            
            for idx, out in enumerate(tqdm(self.__call__(
                data(segments), 
                batch_size=batch_size, 
                num_workers=num_workers
            ), total=len(segments), desc="Processing segments of language: " + lang)):
                # Show progress if requested
                if print_progress:
                    base_progress = ((idx + 1) / total_segments) * 100
                    percent_complete = base_progress / 2 if combined_progress else base_progress
                    print(f"Progress: {percent_complete:.2f}%...")
                
                # Extract text from output
                text = out['text']
                if batch_size in [0, 1, None]:
                    text = text[0]
                    
                # Show verbose output if requested
                if verbose:
                    print(f"Transcript: [{round(segments[idx]['start'], 3)} --> {round(segments[idx]['end'], 3)}] {text}")
                
                lang_segments.append(
                    {
                        "text": text,
                        "start": round(segments[idx]['start'], 3),
                        "end": round(segments[idx]['end'], 3),
                        "language": lang
                    }
                )

            # Clean up
            self.tokenizer = None
            
            # Restore original suppress tokens if needed
            if self.suppress_numerals and previous_suppress_tokens is not None:
                self.options = replace(self.options, suppress_tokens=previous_suppress_tokens)

            for lang_seg, lang_prob in zip(lang_segments, language_probabilities[lang]):
                lang_seg['probability'] = lang_prob
            
            all_segments.extend(lang_segments)
        
        # Sort all segments by start time
        all_segments.sort(key=lambda x: x['start'])
        
        # Create the final result dictionary
        result = {
            "segments": all_segments,
            "language": audio_language,
            "language_probability": audio_language_probability
        }
        
        return result
        
        
    def detect_language(self, audio: np.ndarray) -> str:
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                      n_mels=model_n_mels if model_n_mels is not None else 80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        return language, round(language_probability, 2)


def load_model(
    whisper_arch: str,
    device: str,
    device_index=0,
    compute_type="float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad]= None,
    vad_method: Optional[str] = "silero_custom",
    vad_options: Optional[dict] = None,
    model: Optional[WhisperModel] = None,
    task="transcribe",
    download_root: Optional[str] = None,
    local_files_only=False,
    threads=4,
    vad_onnx=True,
    silero_merge_cutoff=0.1
) -> FasterWhisperPipeline:
    """Load a Whisper model for inference.
    Args:
        whisper_arch - The name of the Whisper model to load.
        device - The device to load the model on.
        compute_type - The compute type to use for the model.
        vad_method - The vad method to use. vad_model has higher priority if is not None.
        options - A dictionary of options to use for the model.
        language - The language of the model. (use English for now)
        model - The WhisperModel instance to use.
        download_root - The root directory to download the model to.
        local_files_only - If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        threads - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
        vad_onnx - If `True`, use the ONNX version of the Silero VAD model.
        silero_merge_cutoff - The merge cutoff for the Silero VAD model.
    Returns:
        A Whisper pipeline.
    """

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         local_files_only=local_files_only,
                         cpu_threads=threads)
    if language is not None:
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        print("No language specified, language will be first be detected for each audio file (increases inference time).")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'""¿([{-",
        "append_punctuations": "\"'.。,，!！?？:："")]}、",
        "multilingual": model.model.is_multilingual,
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "chunk_size": 30, # needed by silero since binarization happens before merge_chunks
        "vad_onset": 0.500,
        "vad_offset": 0.363,
        "vad_onnx": vad_onnx,
        "silero_merge_cutoff": silero_merge_cutoff
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    # Note: manually assigned vad_model has higher priority than vad_method!
    if vad_model is not None:
        print("Use manually assigned vad_model. vad_method is ignored.")
        vad_model = vad_model
    else:
        if vad_method == "silero":
            vad_model = Silero(**default_vad_options)
        elif vad_method == "silero_custom":
            vad_model = SileroCustom(**default_vad_options)
        elif vad_method == "pyannote":
            vad_model = Pyannote(torch.device(device), use_auth_token=None, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )
"""
Type definitions for WhisperX.

This module contains TypedDict classes that define the data structures used
throughout WhisperX for representing transcription and alignment results.
"""

from typing import TypedDict, Optional, List, Tuple
from dataclasses import dataclass

class SingleWordSegment(TypedDict):
    """
    A single word of a speech with timing information.
    
    Attributes
    ----------
    word : str
        The word text
    start : float
        Start time of the word in seconds
    end : float
        End time of the word in seconds
    score : float
        Confidence score for the word alignment
    """
    word: str
    start: float
    end: float
    score: float


class SingleCharSegment(TypedDict):
    """
    A single character of a speech with timing information.
    
    Attributes
    ----------
    char : str
        The character
    start : float
        Start time of the character in seconds
    end : float
        End time of the character in seconds
    score : float
        Confidence score for the character alignment
    """
    char: str
    start: float
    end: float
    score: float


@dataclass
class SingleSegment:
    """A single segment of transcribed audio."""
    text: str
    start: float
    end: float
    language: str
    language_probability: float


class SingleAlignedSegment(TypedDict):
    """
    A segment of speech with word and optionally character-level alignment.
    
    This extends SingleSegment with word and character timing information.
    
    Attributes
    ----------
    start : float
        Start time of the segment in seconds
    end : float
        End time of the segment in seconds
    text : str
        Transcribed text for the segment
    words : List[SingleWordSegment]
        List of words with timing information
    chars : Optional[List[SingleCharSegment]]
        List of characters with timing information (if character alignment was requested)
    """
    start: float
    end: float
    text: str
    words: List[SingleWordSegment]
    chars: Optional[List[SingleCharSegment]]


class SegmentData(TypedDict):
    """
    Temporary processing data used during alignment.
    
    This contains cleaned and preprocessed data for each segment to facilitate
    alignment with phoneme models.
    
    Attributes
    ----------
    clean_char : List[str]
        Cleaned characters that exist in the model dictionary
    clean_cdx : List[int]
        Original indices of cleaned characters in the input text
    clean_wdx : List[int]
        Indices of words containing valid characters
    sentence_spans : List[Tuple[int, int]]
        Start and end indices of sentences in the segment
    """
    clean_char: List[str]  # Cleaned characters that exist in model dictionary
    clean_cdx: List[int]   # Original indices of cleaned characters
    clean_wdx: List[int]   # Indices of words containing valid characters
    sentence_spans: List[Tuple[int, int]]  # Start and end indices of sentences


@dataclass
class TranscriptionResult:
    """The result of a transcription."""
    segments: List[SingleSegment]
    language: str
    language_probability: float


class AlignedTranscriptionResult(TypedDict):
    """
    Result of the full transcription and alignment process.
    
    Contains segments with word-level (and optionally character-level) timing information.
    
    Attributes
    ----------
    segments : List[SingleAlignedSegment]
        List of segments with word and character timing information
    word_segments : List[SingleWordSegment]
        Flattened list of all words across all segments
    language : str
        Detected or specified language code
    language_probability : float
        Probability score of the detected language
    """
    segments: List[SingleAlignedSegment]
    word_segments: List[SingleWordSegment]
    language: str
    language_probability: float

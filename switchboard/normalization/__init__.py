"""Transcript normalization with cross-script spelling equivalents and language-tag alignment preservation."""

from switchboard.normalization.normalizer import (
    DEFAULT_EQUIVALENTS_PATH,
    NormalizedHypothesis,
    NormalizedReference,
    NormalizedTokenMapping,
    TranscriptNormalizer,
)

__all__ = [
    "DEFAULT_EQUIVALENTS_PATH",
    "NormalizedHypothesis",
    "NormalizedReference",
    "NormalizedTokenMapping",
    "TranscriptNormalizer",
]

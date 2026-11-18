"""Transcript normalizer for Hindi–English code-switched speech recognition."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DEFAULT_EQUIVALENTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "normalization_equivalents.yaml"

# Characters to strip as punctuation (ASCII punctuation, Devanagari danda/double-danda, quotes, dashes)
_PUNCT_REGEX = re.compile(
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।\u0964\u0965\u0970“”‘’—–…«»]+""",
    re.UNICODE,
)
_ZERO_WIDTH_REGEX = re.compile(r"[\u200b\u200c\u200d\ufeff]+")


@dataclass
class NormalizedTokenMapping:
    """Tracks the link between an original reference token, its normalized form, and its language tag."""

    orig_index: int
    raw_word: str
    normalized_word: str
    lang_tag: str


@dataclass
class NormalizedReference:
    """Holds both raw and normalized reference representations alongside aligned language tags."""

    raw_transcript: str
    raw_words: List[str]
    raw_lang_tags: List[str]
    normalized_transcript: str
    normalized_words: List[str]
    normalized_lang_tags: List[str]
    token_mappings: List[NormalizedTokenMapping] = field(default_factory=list)


@dataclass
class NormalizedHypothesis:
    """Holds both raw and normalized hypothesis representations."""

    raw_transcript: str
    raw_words: List[str]
    normalized_transcript: str
    normalized_words: List[str]


class TranscriptNormalizer:
    """Applies identical Unicode, punctuation, case, whitespace, and reviewed cross-script spelling rules
    to both reference and hypothesis transcripts while preserving reference word-to-language-tag mappings.
    """

    def __init__(
        self,
        equivalents_path: Optional[str | Path] = None,
        extra_equivalents: Optional[Dict[str, str]] = None,
        apply_equivalents: bool = True,
    ) -> None:
        self.apply_equivalents = apply_equivalents
        self.equivalents: Dict[str, str] = {}

        path_to_load = Path(equivalents_path) if equivalents_path is not None else DEFAULT_EQUIVALENTS_PATH
        if path_to_load and path_to_load.exists():
            with path_to_load.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                raw_map = loaded.get("canonical_equivalents", loaded)
                if isinstance(raw_map, dict):
                    for k, v in raw_map.items():
                        norm_k = self._clean_surface(str(k))
                        norm_v = self._clean_surface(str(v))
                        if norm_k and norm_v:
                            self.equivalents[norm_k] = norm_v

        if extra_equivalents:
            for k, v in extra_equivalents.items():
                norm_k = self._clean_surface(str(k))
                norm_v = self._clean_surface(str(v))
                if norm_k and norm_v:
                    self.equivalents[norm_k] = norm_v

    @staticmethod
    def _clean_surface(token: str) -> str:
        """Apply Unicode NFC normalization, zero-width removal, lowercase folding, and punctuation stripping."""
        s = unicodedata.normalize("NFC", token)
        s = _ZERO_WIDTH_REGEX.sub("", s)
        s = s.lower()
        s = _PUNCT_REGEX.sub("", s)
        return s.strip()

    def normalize_word(self, word: str) -> str:
        """Normalize a single token and apply the reviewed cross-script/spelling equivalent lookup."""
        cleaned = self._clean_surface(word)
        if not cleaned:
            return ""
        if self.apply_equivalents and cleaned in self.equivalents:
            return self.equivalents[cleaned]
        return cleaned

    def normalize_text(self, text: str) -> str:
        """Normalize an arbitrary string using the exact reference/hypothesis pipeline."""
        raw_tokens = text.strip().split()
        norm_tokens = [self.normalize_word(t) for t in raw_tokens]
        norm_tokens = [t for t in norm_tokens if t]
        return " ".join(norm_tokens)

    def normalize_reference(self, transcript: str, lang_tags: List[str]) -> NormalizedReference:
        """Normalize a reference transcript while preserving the exact mapping to word-level language tags."""
        raw_words = transcript.strip().split()
        if len(raw_words) != len(lang_tags):
            raise ValueError(
                f"Reference word count ({len(raw_words)}) does not match lang_tags count ({len(lang_tags)})."
            )

        token_mappings: List[NormalizedTokenMapping] = []
        norm_words: List[str] = []
        norm_tags: List[str] = []

        for idx, (raw_w, tag) in enumerate(zip(raw_words, lang_tags)):
            # If punctuation inside a word splits it or leaves a single token, handle cleanly
            norm_w = self.normalize_word(raw_w)
            if not norm_w:
                # Token was pure punctuation (e.g., '।'); omit from normalized word list
                continue
            sub_parts = norm_w.split()
            for sp in sub_parts:
                norm_words.append(sp)
                norm_tags.append(tag)
                token_mappings.append(
                    NormalizedTokenMapping(
                        orig_index=idx,
                        raw_word=raw_w,
                        normalized_word=sp,
                        lang_tag=tag,
                    )
                )

        return NormalizedReference(
            raw_transcript=transcript,
            raw_words=raw_words,
            raw_lang_tags=list(lang_tags),
            normalized_transcript=" ".join(norm_words),
            normalized_words=norm_words,
            normalized_lang_tags=norm_tags,
            token_mappings=token_mappings,
        )

    def normalize_hypothesis(self, hypothesis: str) -> NormalizedHypothesis:
        """Normalize an ASR hypothesis using the exact same rules applied to references."""
        raw_words = hypothesis.strip().split()
        norm_words: List[str] = []
        for raw_w in raw_words:
            norm_w = self.normalize_word(raw_w)
            if not norm_w:
                continue
            norm_words.extend(norm_w.split())

        return NormalizedHypothesis(
            raw_transcript=hypothesis,
            raw_words=raw_words,
            normalized_transcript=" ".join(norm_words),
            normalized_words=norm_words,
        )

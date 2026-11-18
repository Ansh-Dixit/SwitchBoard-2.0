"""Unified dataset manifest schema, 16 kHz mono audio resampler, and split integrity validators."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

VALID_LANG_TAGS: Set[str] = {"HI", "EN", "OTHER"}
VALID_SPLITS: Set[str] = {"train", "val", "test"}
TARGET_SAMPLE_RATE: int = 16000


@dataclass
class ManifestRecord:
    """Single utterance record in the unified SwitchBoard manifest format."""

    utterance_id: str
    audio_path: str
    transcript: str
    lang_tags: List[str]
    speaker_id: str
    accent: Optional[str]
    source: str
    split: str
    duration_sec: Optional[float] = None
    is_synthetic: bool = False
    audio_sha256: Optional[str] = None
    review_status: str = "reviewed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def words(self) -> List[str]:
        return self.transcript.strip().split()

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestRecord":
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        if extra and "metadata" not in filtered:
            filtered["metadata"] = extra
        return cls(**filtered)


def validate_record(record: ManifestRecord, check_audio_exists: bool = False, base_dir: Optional[Path] = None) -> None:
    """Validate a single ManifestRecord for schema, word/tag counts, and synthetic split rules."""
    if not record.utterance_id or not str(record.utterance_id).strip():
        raise ValueError("utterance_id must be a non-empty string.")
    if not record.audio_path or not str(record.audio_path).strip():
        raise ValueError(f"[{record.utterance_id}] audio_path must be non-empty.")
    if not record.speaker_id or not str(record.speaker_id).strip():
        raise ValueError(f"[{record.utterance_id}] speaker_id must be non-empty.")
    if not record.source or not str(record.source).strip():
        raise ValueError(f"[{record.utterance_id}] source must be non-empty.")
    if record.split not in VALID_SPLITS:
        raise ValueError(
            f"[{record.utterance_id}] Invalid split '{record.split}'. Expected one of {sorted(VALID_SPLITS)}."
        )

    words = record.words
    if len(words) == 0:
        raise ValueError(f"[{record.utterance_id}] transcript must contain at least one word.")
    if len(words) != len(record.lang_tags):
        raise ValueError(
            f"[{record.utterance_id}] Word count ({len(words)}) does not match lang_tags count ({len(record.lang_tags)}). "
            f"Words={words}, Tags={record.lang_tags}"
        )

    invalid_tags = [t for t in record.lang_tags if t not in VALID_LANG_TAGS]
    if invalid_tags:
        raise ValueError(
            f"[{record.utterance_id}] Invalid language tags {invalid_tags}. Allowed tags: {sorted(VALID_LANG_TAGS)}."
        )

    is_syn = record.is_synthetic or ("synthetic" in record.source.lower())
    if is_syn and record.split != "train":
        raise ValueError(
            f"[{record.utterance_id}] Synthetic mixed speech (source='{record.source}', is_synthetic={record.is_synthetic}) "
            f"is only permitted in the 'train' split, but found in '{record.split}'."
        )

    if check_audio_exists:
        audio_p = Path(record.audio_path)
        if not audio_p.is_absolute() and base_dir is not None:
            audio_p = base_dir / audio_p
        if not audio_p.exists():
            raise FileNotFoundError(f"[{record.utterance_id}] Audio file not found: {audio_p}")


def validate_manifest_splits(
    records: Iterable[ManifestRecord],
    check_audio_exists: bool = False,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate an entire manifest collection for speaker separation, duplicate separation, and synthetic constraints.

    Raises ValueError if:
    - Any record fails `validate_record`
    - Any utterance_id is duplicated
    - Any speaker_id appears in more than one split (train vs val vs test)
    - Any duplicate audio recording (by audio_sha256 or normalized audio_path) appears across multiple splits
    - Any synthetic utterance appears outside 'train'
    """
    record_list = list(records)
    if not record_list:
        raise ValueError("Manifest contains zero records.")

    seen_utt_ids: Set[str] = set()
    speaker_to_splits: Dict[str, Set[str]] = {}
    audio_key_to_splits: Dict[str, Set[str]] = {}
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    synthetic_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for rec in record_list:
        validate_record(rec, check_audio_exists=check_audio_exists, base_dir=base_dir)
        if rec.utterance_id in seen_utt_ids:
            raise ValueError(f"Duplicate utterance_id detected: '{rec.utterance_id}'")
        seen_utt_ids.add(rec.utterance_id)

        split_counts[rec.split] = split_counts.get(rec.split, 0) + 1
        if rec.is_synthetic or "synthetic" in rec.source.lower():
            synthetic_counts[rec.split] = synthetic_counts.get(rec.split, 0) + 1

        speaker_to_splits.setdefault(rec.speaker_id, set()).add(rec.split)

        # Track duplicate recordings by audio_sha256 (if computed) and canonical audio_path
        norm_path = str(Path(rec.audio_path)).replace("\\", "/").lower()
        audio_key_to_splits.setdefault(f"path:{norm_path}", set()).add(rec.split)
        if rec.audio_sha256:
            audio_key_to_splits.setdefault(f"sha256:{rec.audio_sha256}", set()).add(rec.split)

    overlapping_speakers = {spk: splits for spk, splits in speaker_to_splits.items() if len(splits) > 1}
    if overlapping_speakers:
        raise ValueError(
            f"Speaker leakage across splits detected! Speakers present in multiple splits: {overlapping_speakers}"
        )

    overlapping_recordings = {k: splits for k, splits in audio_key_to_splits.items() if len(splits) > 1}
    if overlapping_recordings:
        raise ValueError(
            f"Duplicate recording leakage across splits detected! Recordings in multiple splits: {overlapping_recordings}"
        )

    return {
        "total_records": len(record_list),
        "split_counts": split_counts,
        "synthetic_counts": synthetic_counts,
        "unique_speakers": len(speaker_to_splits),
        "status": "valid",
    }


def compute_audio_sha256(waveform: np.ndarray) -> str:
    """Compute a deterministic SHA-256 fingerprint of a 16-bit PCM quantized waveform."""
    pcm16 = np.clip(np.round(waveform * 32767.0), -32768, 32767).astype(np.int16)
    return hashlib.sha256(pcm16.tobytes()).hexdigest()


def resample_to_16k_mono(
    input_path: str | Path,
    output_path: str | Path,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[float, str]:
    """Load an audio file, convert to mono, resample to 16 kHz, save as 16-bit PCM WAV, and return (duration_sec, sha256)."""
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        data, sr = sf.read(str(in_p), dtype="float32", always_2d=True)
        # Average across channels to produce mono
        mono = np.mean(data, axis=1)
    except Exception:
        # Fallback to librosa for MP3/M4A formats if soundfile backend lacks codec
        import librosa

        mono, sr = librosa.load(str(in_p), sr=None, mono=True)
        mono = mono.astype(np.float32)

    if sr != target_sr:
        gcd = math.gcd(int(sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(sr) // gcd
        mono = resample_poly(mono, up, down).astype(np.float32)

    # Peak normalize gently if clipping
    peak = float(np.max(np.abs(mono))) if mono.size > 0 else 0.0
    if peak > 0.99:
        mono = mono / peak * 0.98

    sf.write(str(out_p), mono, target_sr, subtype="PCM_16")
    duration_sec = round(float(len(mono)) / float(target_sr), 4)
    sha256 = compute_audio_sha256(mono)
    return duration_sec, sha256


def save_manifest(records: Iterable[ManifestRecord], manifest_path: str | Path, validate: bool = True) -> Path:
    """Save a sequence of ManifestRecords to a JSONL file."""
    record_list = list(records)
    if validate:
        validate_manifest_splits(record_list)
    out_p = Path(manifest_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for rec in record_list:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return out_p


def load_manifest(manifest_path: str | Path, validate: bool = True) -> List[ManifestRecord]:
    """Load ManifestRecords from a JSONL file."""
    in_p = Path(manifest_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Manifest file not found: {in_p}")
    records: List[ManifestRecord] = []
    with in_p.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            records.append(ManifestRecord.from_dict(data))
    if validate and records:
        validate_manifest_splits(records)
    return records

"""Registry of verified ASR checkpoints, language coverage, access requirements, licenses, and CTC head dependencies."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckpointMetadata:
    """Verified metadata for a baseline or fine-tuned ASR model."""

    key: str
    checkpoint_id: str
    architecture: str
    role: str
    languages: List[str]
    script_output: str
    license: str
    access: str
    requires_ctc_head_training: bool
    ctc_dependency_note: str
    recommended_decoding: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


MODEL_REGISTRY: Dict[str, CheckpointMetadata] = {
    "whisper_small_baseline": CheckpointMetadata(
        key="whisper_small_baseline",
        checkpoint_id="openai/whisper-small",
        architecture="WhisperForConditionalGeneration (244M params, 12 encoder / 12 decoder layers)",
        role="Primary Baseline & Fine-Tuning Base Checkpoint",
        languages=["hi", "en", "multilingual (99 languages)"],
        script_output="Mixed Latin / Devanagari depending on acoustic and prompt context",
        license="Apache-2.0",
        access="Public, non-gated on Hugging Face Hub",
        requires_ctc_head_training=False,
        ctc_dependency_note="Seq2Seq encoder-decoder with pretrained token head; ready for transcription mode.",
        recommended_decoding={"task": "transcribe", "num_beams": 1, "temperature": 0.0},
    ),
    "indicwhisper_hi": CheckpointMetadata(
        key="indicwhisper_hi",
        checkpoint_id="vasista22/whisper-hindi-small",
        architecture="WhisperForConditionalGeneration (AI4Bharat Vistaar / Indic fine-tuned Whisper)",
        role="Comparison Baseline 1 (AI4Bharat IndicWhisper family)",
        languages=["hi", "en-loanwords-in-devanagari"],
        script_output="Devanagari (transliterates embedded English words such as 'meeting' -> 'मीटिंग')",
        license="MIT / CC-BY-4.0 (Vistaar / IndicSUPERB)",
        access="Public on Hugging Face Hub (`vasista22/whisper-hindi-small`); official AI4Bharat Vistaar archives also loadable from local directory",
        requires_ctc_head_training=False,
        ctc_dependency_note="Seq2Seq encoder-decoder; ready for transcription mode (`task='transcribe'`, `language='hi'`).",
        recommended_decoding={"task": "transcribe", "language": "hi", "num_beams": 1, "temperature": 0.0},
    ),
    "wav2vec2_xlsr_300m_base": CheckpointMetadata(
        key="wav2vec2_xlsr_300m_base",
        checkpoint_id="facebook/wav2vec2-xls-r-300m",
        architecture="Wav2Vec2Model / XLS-R Self-Supervised Speech Encoder (300M params)",
        role="Comparison Baseline 2a (Pretrained SSL XLS-R Base without CTC Head)",
        languages=["hi", "en", "multilingual (128 languages SSL pretraining)"],
        script_output="N/A until CTC linear head and character vocabulary are trained",
        license="Apache-2.0",
        access="Public, non-gated on Hugging Face Hub",
        requires_ctc_head_training=True,
        ctc_dependency_note=(
            "DEPENDENCY BLOCK: `facebook/wav2vec2-xls-r-300m` is a self-supervised pretrained encoder WITHOUT a trained "
            "`lm_head` CTC projection layer or character tokenizer (`vocab.json`). Using it directly for ASR requires "
            "fine-tuning a linear CTC head on labeled Hindi-English speech, or using a fine-tuned CTC checkpoint such as "
            "`Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`."
        ),
        recommended_decoding={"decoding": "ctc_greedy"},
    ),
    "wav2vec2_xlsr_ctc_hi": CheckpointMetadata(
        key="wav2vec2_xlsr_ctc_hi",
        checkpoint_id="Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
        architecture="Wav2Vec2ForCTC (Fine-tuned Hindi CTC Head)",
        role="Comparison Baseline 2b (Fine-Tuned wav2vec 2.0 CTC Checkpoint)",
        languages=["hi"],
        script_output="Devanagari character CTC vocabulary",
        license="MIT",
        access="Public, non-gated on Hugging Face Hub",
        requires_ctc_head_training=False,
        ctc_dependency_note="Includes a pretrained Devanagari CTC linear head (`lm_head`) and `vocab.json`.",
        recommended_decoding={"decoding": "ctc_greedy"},
    ),
}


def get_checkpoint_metadata(key_or_id: str) -> CheckpointMetadata:
    """Look up checkpoint metadata by registry key or Hugging Face checkpoint ID."""
    if key_or_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[key_or_id]
    for meta in MODEL_REGISTRY.values():
        if meta.checkpoint_id.lower() == key_or_id.lower():
            return meta
    # Default fallback for custom local or HF checkpoints
    is_base_xlsr = "xls-r-300m" in key_or_id.lower() or key_or_id.lower().endswith("wav2vec2-base")
    return CheckpointMetadata(
        key=key_or_id,
        checkpoint_id=key_or_id,
        architecture="Custom / Hugging Face ASR Checkpoint",
        role="Custom Comparison Model",
        languages=["hi", "en"],
        script_output="Depends on tokenizer",
        license="See checkpoint model card",
        access="Local or Hugging Face Hub",
        requires_ctc_head_training=is_base_xlsr,
        ctc_dependency_note=(
            "Requires CTC head training before transcription."
            if is_base_xlsr
            else "Assumed ready for transcription."
        ),
        recommended_decoding={"task": "transcribe", "num_beams": 1},
    )


def list_registered_models() -> List[Dict[str, Any]]:
    """Return all registered baseline and comparison models as dictionaries."""
    return [m.to_dict() for m in MODEL_REGISTRY.values()]


def verify_checkpoint_status(checkpoint_id: str) -> Dict[str, Any]:
    """Verify a checkpoint's local cache presence, CTC head status, and metadata."""
    meta = get_checkpoint_metadata(checkpoint_id)
    local_path = Path(checkpoint_id)
    is_local_dir = local_path.exists()

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_folder_name = "models--" + checkpoint_id.replace("/", "--")
    is_cached = (cache_root / cache_folder_name).exists()

    status = "ready"
    if meta.requires_ctc_head_training:
        status = "requires_ctc_head_training"

    return {
        "checkpoint_id": checkpoint_id,
        "role": meta.role,
        "architecture": meta.architecture,
        "languages": meta.languages,
        "license": meta.license,
        "access": meta.access,
        "requires_ctc_head_training": meta.requires_ctc_head_training,
        "ctc_dependency_note": meta.ctc_dependency_note,
        "is_local_dir": is_local_dir,
        "is_cached_locally": is_cached,
        "status": status,
    }

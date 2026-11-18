"""Whisper Small PEFT LoRA fine-tuning, auxiliary decoder Language Identification head, and curriculum learning."""

from switchboard.training.whisper_lora_lid import (
    ID_TO_LID_TAG,
    IGNORE_INDEX,
    LID_TAG_TO_ID,
    AuxiliaryLIDHead,
    CurriculumTrainingRunner,
    align_word_tags_to_subwords,
    build_whisper_lora_lid_model,
    compute_joint_asr_lid_loss,
    filter_records_for_curriculum_stage,
    find_decoder_attention_target_modules,
    load_finetuned_whisper_lora_lid,
    verify_checkpoint_reload,
)

__all__ = [
    "AuxiliaryLIDHead",
    "CurriculumTrainingRunner",
    "ID_TO_LID_TAG",
    "IGNORE_INDEX",
    "LID_TAG_TO_ID",
    "align_word_tags_to_subwords",
    "build_whisper_lora_lid_model",
    "compute_joint_asr_lid_loss",
    "filter_records_for_curriculum_stage",
    "find_decoder_attention_target_modules",
    "load_finetuned_whisper_lora_lid",
    "verify_checkpoint_reload",
]

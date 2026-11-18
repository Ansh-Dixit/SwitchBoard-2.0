"""Comprehensive unit and integration test suite for SwitchBoard."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from switchboard.data.importers import (
    build_demo_fixture,
    import_common_voice,
    import_kathbath_hi,
    import_local_team_recordings,
    import_mucs2021_hi_en,
)
from switchboard.data.manifest import (
    ManifestRecord,
    resample_to_16k_mono,
    validate_manifest_splits,
    validate_record,
)
from switchboard.evaluation.boundary import (
    align_and_attribute_edits,
    compute_cmi,
    detect_switch_boundaries,
    evaluate_corpus,
    evaluate_utterance,
    export_evaluation_results,
)
from switchboard.models.registry import get_checkpoint_metadata, verify_checkpoint_status
from switchboard.models.runners import Wav2Vec2CTCRunner
from switchboard.normalization.normalizer import TranscriptNormalizer
from switchboard.training.whisper_lora_lid import (
    IGNORE_INDEX,
    LID_TAG_TO_ID,
    SimpleWordSubwordTokenizer,
    align_word_tags_to_subwords,
    build_whisper_lora_lid_model,
    collate_whisper_lid_batch,
    compute_joint_asr_lid_loss,
    filter_records_for_curriculum_stage,
    save_finetuned_whisper_lora_lid,
    verify_checkpoint_reload,
)


# =============================================================================
# 1. TRANSCRIPT NORMALIZATION TESTS
# =============================================================================


def test_normalization_equivalents_and_tag_preservation() -> None:
    """Verify cross-script equivalents ('मीटिंग' <-> 'meeting'), Unicode/case/punctuation rules,
    reference language-tag alignment preservation, and that true lexical errors still count.
    """
    normalizer = TranscriptNormalizer()

    # Cross-script Devanagari -> canonical English/Romanized equivalent
    assert normalizer.normalize_word("मीटिंग") == "meeting"
    assert normalizer.normalize_word("MEETING!") == "meeting"
    assert normalizer.normalize_word("रिपोर्ट।") == "report"
    assert normalizer.normalize_word("कल") == "kal"
    assert normalizer.normalize_word("दूँगा") == "dunga"
    assert normalizer.normalize_word("doonga") == "dunga"

    # Reference normalization preserves 1-to-1 language tags
    ref_text = "कल, मीटिंग के बाद! मैं रिपोर्ट सबमिट कर दूँगा।"
    ref_tags = ["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"]
    norm_ref = normalizer.normalize_reference(ref_text, ref_tags)

    assert norm_ref.raw_transcript == ref_text
    assert norm_ref.normalized_words == [
        "kal", "meeting", "ke", "baad", "main", "report", "submit", "kar", "dunga"
    ]
    assert norm_ref.normalized_lang_tags == ref_tags
    assert len(norm_ref.token_mappings) == 9
    assert norm_ref.token_mappings[1].raw_word == "मीटिंग"
    assert norm_ref.token_mappings[1].normalized_word == "meeting"
    assert norm_ref.token_mappings[1].lang_tag == "EN"


def test_true_word_errors_still_count_after_normalization() -> None:
    """Verify that normalization eliminates script/spelling noise ('मीटिंग' vs 'meeting')
    while true lexical errors ('eating' vs 'meeting', 'deport' vs 'report') still count as errors.
    """
    normalizer = TranscriptNormalizer()
    rec = ManifestRecord(
        utterance_id="utt_norm_test",
        audio_path="dummy.wav",
        transcript="कल मीटिंग के बाद मैं रिपोर्ट सबमिट कर दूंगा",
        lang_tags=["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"],
        speaker_id="spk1",
        accent="Delhi",
        source="test",
        split="test",
        duration_sec=3.0,
    )

    # Hypothesis 1: Romanized equivalent of exact words -> 0 normalized errors, but raw WER is 9/9 (100%)
    hyp_script_variant = "kal meeting ke baad main report submit kar dunga"
    res_variant = evaluate_utterance(rec, hyp_script_variant, normalizer=normalizer)
    assert res_variant.raw_overall_wer["errors"] == 9
    assert res_variant.overall_wer["errors"] == 0
    assert res_variant.overall_wer["rate"] == 0.0

    # Hypothesis 2: True lexical errors ('eating' instead of 'meeting', 'deport' instead of 'report')
    hyp_true_errors = "kal eating ke baad main deport submit kar dunga"
    res_true_err = evaluate_utterance(rec, hyp_true_errors, normalizer=normalizer)
    assert res_true_err.overall_wer["errors"] == 2
    assert res_true_err.overall_wer["substitutions"] == 2
    assert res_true_err.overall_wer["words"] == 9
    assert math.isclose(res_true_err.overall_wer["rate"], 2.0 / 9.0, rel_tol=1e-5)


# =============================================================================
# 2. SWITCH DETECTION, OVERLAPPING BOUNDARIES & OTHER TOKENS
# =============================================================================


def test_switch_detection_and_overlapping_boundaries() -> None:
    """Test prompt example: 'kal meeting ke baad main report submit kar dunga'.
    - 'kal' (0, HI) -> 'meeting' (1, EN) is a switch boundary (HI->EN)
    - 'meeting' (1, EN) -> 'ke' (2, HI) is a switch boundary (EN->HI)
    - 'meeting' (index 1) is touched by TWO switches and must be counted ONLY ONCE in pooled boundary score.
    - 'report' (5, EN) -> 'submit' (6, EN) stays within English (not a switch between 5 and 6).
    - 'main' (4, HI) -> 'report' (5, EN) is HI->EN, and 'submit' (6, EN) -> 'kar' (7, HI) is EN->HI.
    """
    words = "kal meeting ke baad main report submit kar dunga".split()
    tags = ["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"]

    res = detect_switch_boundaries(words, tags, bridge_other=False)

    assert len(res.transitions) == 4
    directions = [(t.pre_index, t.post_index, t.direction) for t in res.transitions]
    assert directions == [
        (0, 1, "HI->EN"),  # kal -> meeting
        (1, 2, "EN->HI"),  # meeting -> ke
        (4, 5, "HI->EN"),  # main -> report
        (6, 7, "EN->HI"),  # submit -> kar
    ]

    # Index 1 ('meeting') is touched by two switches (0->1 and 1->2)
    assert res.overlapping_boundary_indices == {1}
    # Pooled boundary indices count index 1 only once: {0, 1, 2, 4, 5, 6, 7} -> 7 words
    assert res.boundary_indices == {0, 1, 2, 4, 5, 6, 7}
    assert len(res.boundary_indices) == 7
    # Non-boundary indices are index 3 ('baad') and index 8 ('dunga') -> 2 words
    assert res.non_boundary_indices == {3, 8}


def test_other_tokens_boundary_policy() -> None:
    """Verify how OTHER tokens (names, fillers) affect switch boundary detection."""
    words = ["Rahul", "umm", "kal", "meeting", "हम्म", "है"]
    tags = ["OTHER", "OTHER", "HI", "EN", "OTHER", "HI"]

    # Default strict policy (bridge_other=False): OTHER does not trigger switches
    res_strict = detect_switch_boundaries(words, tags, bridge_other=False)
    assert len(res_strict.transitions) == 1
    assert res_strict.transitions[0].pre_index == 2  # 'kal'
    assert res_strict.transitions[0].post_index == 3  # 'meeting'
    assert res_strict.boundary_indices == {2, 3}
    assert 0 not in res_strict.boundary_indices
    assert 1 not in res_strict.boundary_indices
    assert 4 not in res_strict.boundary_indices

    # Optional bridged mode (bridge_other=True): bridges across 'हम्म' (index 4) between 'meeting' (3, EN) and 'है' (5, HI)
    res_bridged = detect_switch_boundaries(words, tags, bridge_other=True)
    assert len(res_bridged.transitions) == 2
    assert res_bridged.boundary_indices == {2, 3, 5}
    assert 4 not in res_bridged.boundary_indices  # OTHER token itself remains non-boundary


# =============================================================================
# 3. DETERMINISTIC INSERTION ATTRIBUTION TESTS
# =============================================================================


def test_deterministic_insertion_attribution_beginning_interior_and_end() -> None:
    """Test deterministic insertion assignment at utterance beginning (i=0 -> ref 0),
    interior (between i-1 and i -> ref i-1), and end (i=N -> ref N-1).
    """
    ref_words = ["kal", "meeting", "hai"]
    ref_tags = ["HI", "EN", "HI"]
    # Insert 'extra_start' before 'kal', 'extra_mid' after 'meeting', and 'extra_end' after 'hai'
    hyp_words = ["extra_start", "kal", "meeting", "extra_mid", "hai", "extra_end"]

    ops, attrs = align_and_attribute_edits(ref_words, hyp_words, ref_tags)
    ins_ops = [op for op in ops if op.op == "INS"]
    assert len(ins_ops) == 3

    # Leading insertion ('extra_start') -> attributed to reference position 0 ('kal')
    assert ins_ops[0].hyp_word == "extra_start"
    assert ins_ops[0].attributed_ref_index == 0
    # Interior insertion ('extra_mid' after 'meeting') -> attributed to reference position 1 ('meeting')
    assert ins_ops[1].hyp_word == "extra_mid"
    assert ins_ops[1].attributed_ref_index == 1
    # Trailing insertion ('extra_end' after 'hai') -> attributed to reference position 2 ('hai')
    assert ins_ops[2].hyp_word == "extra_end"
    assert ins_ops[2].attributed_ref_index == 2

    assert attrs[0].insertions == 1
    assert attrs[1].insertions == 1
    assert attrs[2].insertions == 1
    assert sum(a.total_edits for a in attrs) == 3


# =============================================================================
# 4. HAND-CALCULATED BOUNDARY WER & CMI VERIFICATION
# =============================================================================


def test_hand_calculated_boundary_wer_and_cmi() -> None:
    """Verify exact hand-calculated Overall WER, Boundary WER, Non-Boundary WER, directional WERs, and CMI.

    Hand Calculation Setup:
    - Reference (9 words):
        0: 'kal'     (HI) -> Boundary (before HI->EN switch 0->1)
        1: 'meeting' (EN) -> Boundary (after 0->1 HI->EN, and before 1->2 EN->HI; OVERLAPPING, counted ONCE in pooled boundary)
        2: 'ke'      (HI) -> Boundary (after 1->2 EN->HI)
        3: 'baad'    (HI) -> Non-Boundary
        4: 'main'    (HI) -> Boundary (before 4->5 HI->EN)
        5: 'report'  (EN) -> Boundary (after 4->5 HI->EN)
        6: 'submit'  (EN) -> Boundary (before 6->7 EN->HI)
        7: 'kar'     (HI) -> Boundary (after 6->7 EN->HI)
        8: 'dunga'   (HI) -> Non-Boundary
      Word counts:
        - Total reference words N = 9
        - Boundary words = {0, 1, 2, 4, 5, 6, 7} -> 7 words
        - Non-boundary words = {3, 8} -> 2 words
        - HI count = 6, EN count = 3, OTHER count = 0
        - Hand-calculated CMI = 100 * (1 - max(6, 3) / 9) = 100 * (1 - 6/9) = 33.3333

    - Hypothesis:
        'kal eating ke baad main report submit dunga extra'
      Edits vs Reference:
        1. Index 1 ('meeting' -> 'eating'): 1 Substitution at ref index 1 (Boundary word!)
        2. Index 7 ('kar' deleted): 1 Deletion at ref index 7 (Boundary word!)
        3. Trailing insertion 'extra' after index 8 ('dunga'): 1 Insertion attributed to ref index 8 (Non-Boundary word!)

    - Hand-Calculated Expected Scores:
        - Overall WER: 3 errors (1 SUB + 1 DEL + 1 INS) / 9 words = 3/9 = 0.333333
        - Boundary WER: 2 errors (SUB at 1 + DEL at 7) / 7 boundary words = 2/7 = 0.285714
        - Non-Boundary WER: 1 error (INS at 8) / 2 non-boundary words = 1/2 = 0.500000
        - Before-Switch indices: {0, 1, 4, 6} (4 words) -> 1 error (at index 1) -> 1/4 = 0.250000
        - After-Switch indices: {1, 2, 5, 7} (4 words) -> 2 errors (at indices 1 and 7) -> 2/4 = 0.500000
        - HI->EN indices: {0, 1, 4, 5} (4 words) -> 1 error (at index 1) -> 1/4 = 0.250000
        - EN->HI indices: {1, 2, 6, 7} (4 words) -> 2 errors (at indices 1 and 7) -> 2/4 = 0.500000
    """
    rec = ManifestRecord(
        utterance_id="utt_hand_calc",
        audio_path="dummy.wav",
        transcript="kal meeting ke baad main report submit kar dunga",
        lang_tags=["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"],
        speaker_id="spk_hand",
        accent="North Indian Hinglish",
        source="hand_calc",
        split="test",
        duration_sec=4.0,
    )
    hyp = "kal eating ke baad main report submit dunga extra"

    res = evaluate_utterance(rec, hyp, transcription_time_sec=0.8)

    # CMI check: 100 * (1 - 6/9) = 33.3333
    assert math.isclose(res.cmi, 33.3333, abs_tol=1e-3)
    assert res.cmi_bucket == ">30 (High Mixing)"
    assert math.isclose(res.rtf, 0.8 / 4.0, abs_tol=1e-6)

    # Overall WER: 3 / 9
    assert res.overall_wer["errors"] == 3
    assert res.overall_wer["words"] == 9
    assert res.overall_wer["substitutions"] == 1
    assert res.overall_wer["deletions"] == 1
    assert res.overall_wer["insertions"] == 1
    assert math.isclose(res.overall_wer["rate"], 3.0 / 9.0, abs_tol=1e-6)

    # Boundary WER: 2 / 7
    assert res.boundary_wer["errors"] == 2
    assert res.boundary_wer["words"] == 7
    assert math.isclose(res.boundary_wer["rate"], 2.0 / 7.0, abs_tol=1e-6)

    # Non-Boundary WER: 1 / 2
    assert res.non_boundary_wer["errors"] == 1
    assert res.non_boundary_wer["words"] == 2
    assert math.isclose(res.non_boundary_wer["rate"], 1.0 / 2.0, abs_tol=1e-6)

    # Directional & positional breakdowns
    assert res.before_switch_wer["errors"] == 1 and res.before_switch_wer["words"] == 4
    assert math.isclose(res.before_switch_wer["rate"], 0.25, abs_tol=1e-6)

    assert res.after_switch_wer["errors"] == 2 and res.after_switch_wer["words"] == 4
    assert math.isclose(res.after_switch_wer["rate"], 0.50, abs_tol=1e-6)

    assert res.hi_to_en_wer["errors"] == 1 and res.hi_to_en_wer["words"] == 4
    assert math.isclose(res.hi_to_en_wer["rate"], 0.25, abs_tol=1e-6)

    assert res.en_to_hi_wer["errors"] == 2 and res.en_to_hi_wer["words"] == 4
    assert math.isclose(res.en_to_hi_wer["rate"], 0.50, abs_tol=1e-6)


def test_empty_boundary_group_marked_unavailable() -> None:
    """When an utterance is purely monolingual (0 switches), boundary_wer must be marked 'unavailable'."""
    rec = ManifestRecord(
        utterance_id="utt_mono",
        audio_path="dummy.wav",
        transcript="कल हम सब घर जाएंगे",
        lang_tags=["HI", "HI", "HI", "HI", "HI"],
        speaker_id="spk_mono",
        accent="Hindi",
        source="test",
        split="test",
    )
    res = evaluate_utterance(rec, "कल हम सब घर जाएंगे")
    assert res.cmi == 0.0
    assert res.boundary_wer["words"] == 0
    assert res.boundary_wer["rate"] is None
    assert res.boundary_wer["status"] == "unavailable"
    assert res.non_boundary_wer["words"] == 5
    assert res.non_boundary_wer["status"] == "available"


# =============================================================================
# 5. DATASET MANIFEST VALIDATION, AUDIO RESAMPLING & IMPORTERS
# =============================================================================


def test_manifest_split_constraints_and_synthetic_rule() -> None:
    """Verify speaker separation, duplicate recording separation, and synthetic-in-train-only rule."""
    r_train = ManifestRecord(
        utterance_id="u1",
        audio_path="a1.wav",
        transcript="kal meeting hai",
        lang_tags=["HI", "EN", "HI"],
        speaker_id="spk_shared",
        accent="North",
        source="mucs",
        split="train",
    )
    r_test_same_spk = ManifestRecord(
        utterance_id="u2",
        audio_path="a2.wav",
        transcript="aaj office chalo",
        lang_tags=["HI", "EN", "HI"],
        speaker_id="spk_shared",
        accent="North",
        source="mucs",
        split="test",
    )
    with pytest.raises(ValueError, match="Speaker leakage"):
        validate_manifest_splits([r_train, r_test_same_spk])

    # Synthetic in test split must raise ValueError
    r_synth_bad = ManifestRecord(
        utterance_id="u_syn",
        audio_path="a_syn.wav",
        transcript="kal report bhejo",
        lang_tags=["HI", "EN", "HI"],
        speaker_id="spk_syn",
        accent="Synth",
        source="synthetic_tts",
        split="test",
        is_synthetic=True,
    )
    with pytest.raises(ValueError, match="Synthetic mixed speech"):
        validate_record(r_synth_bad)


def test_audio_resampling_to_16k_mono_and_importers(tmp_path: Path) -> None:
    """Test 44.1 kHz stereo -> 16 kHz mono resampling and all four dataset importers."""
    # Create a 44.1 kHz stereo WAV
    sr_orig = 44100
    t = np.linspace(0, 0.5, int(sr_orig * 0.5), endpoint=False, dtype=np.float32)
    stereo = np.stack([0.5 * np.sin(2 * np.pi * 220 * t), 0.5 * np.cos(2 * np.pi * 440 * t)], axis=1)
    raw_wav = tmp_path / "stereo_44k.wav"
    sf.write(str(raw_wav), stereo, sr_orig)

    out_16k = tmp_path / "mono_16k.wav"
    dur, sha = resample_to_16k_mono(raw_wav, out_16k)
    info = sf.info(str(out_16k))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert math.isclose(dur, 0.5, abs_tol=0.01)
    assert len(sha) == 64

    # 1. Test MUCS 2021 Kaldi importer
    mucs_dir = tmp_path / "mucs_raw" / "train"
    mucs_dir.mkdir(parents=True)
    (mucs_dir / "text").write_text("utt_m1 कल मीटिंग है\n", encoding="utf-8")
    (mucs_dir / "wav.scp").write_text(f"utt_m1 {raw_wav.as_posix()}\n", encoding="utf-8")
    (mucs_dir / "utt2spk").write_text("utt_m1 spk_m1\n", encoding="utf-8")
    mucs_recs = import_mucs2021_hi_en(tmp_path / "mucs_raw", tmp_path / "out_audio")
    assert len(mucs_recs) == 1
    assert mucs_recs[0].lang_tags == ["HI", "EN", "HI"]

    # 2. Test IndicSUPERB Kathbath importer
    kb_dir = tmp_path / "kb_raw"
    kb_dir.mkdir(parents=True)
    (kb_dir / "train_manifest.json").write_text(
        json.dumps({"audio_filepath": raw_wav.as_posix(), "text": "आज हम घर जाएंगे", "speaker_id": "kb1"}) + "\n",
        encoding="utf-8",
    )
    kb_recs = import_kathbath_hi(kb_dir, tmp_path / "out_audio")
    assert len(kb_recs) == 1
    assert kb_recs[0].source == "kathbath_hi"

    # 3. Test Common Voice importer
    cv_dir = tmp_path / "cv_raw"
    (cv_dir / "clips").mkdir(parents=True)
    sf.write(str(cv_dir / "clips" / "clip01.wav"), stereo, sr_orig)
    (cv_dir / "train.tsv").write_text(
        "client_id\tpath\tsentence\taccents\nclient_abc\tclip01.wav\tplease submit the report\tIndia and South Asia\n",
        encoding="utf-8",
    )
    cv_recs = import_common_voice(cv_dir, tmp_path / "out_audio", locale="en")
    assert len(cv_recs) == 1
    assert cv_recs[0].accent == "India and South Asia"
    assert cv_recs[0].lang_tags == ["EN", "EN", "EN", "EN"]

    # 4. Test Local Team Recordings importer (including consent check)
    team_dir = tmp_path / "team_raw"
    team_dir.mkdir(parents=True)
    with (team_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "audio_file", "transcript", "lang_tags", "speaker_id", "accent", "consent_verified", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "utterance_id": "team_01",
                "audio_file": raw_wav.as_posix(),
                "transcript": "Rahul kal office aayega",
                "lang_tags": "OTHER HI EN HI",
                "speaker_id": "team_spk_1",
                "accent": "Delhi",
                "consent_verified": "true",
                "split": "train",
            }
        )
    team_recs = import_local_team_recordings(team_dir, tmp_path / "out_audio")
    assert len(team_recs) == 1
    assert team_recs[0].lang_tags == ["OTHER", "HI", "EN", "HI"]


# =============================================================================
# 6. WHISPER LORA + AUXILIARY LID HEAD + CHECKPOINT RELOAD VERIFICATION
# =============================================================================


def test_whisper_lora_decoder_attention_and_lid_head_and_reload(tmp_path: Path) -> None:
    """Verify frozen base Whisper weights, decoder-attention-only LoRA targets, subword LID alignment,
    joint ASR+LID loss backprop, and checkpoint save/reload equivalence.
    """
    tokenizer = SimpleWordSubwordTokenizer()
    words = ["kal", "meeting", "Rahul"]
    tags = ["HI", "EN", "OTHER"]
    prefix = [50258, 50259, 50359, 50363]
    eos = 50257

    t_ids, lid_ids = align_word_tags_to_subwords(words, tags, tokenizer, prefix_token_ids=prefix, eos_token_id=eos)
    assert len(t_ids) == len(lid_ids)
    # Prefix special tokens and EOS token must be masked with IGNORE_INDEX (-100)
    assert lid_ids[:4] == [IGNORE_INDEX] * 4
    assert lid_ids[-1] == IGNORE_INDEX
    # Content subword tokens must carry HI (0), EN (1), OTHER (2)
    content_lid = [x for x in lid_ids if x != IGNORE_INDEX]
    assert LID_TAG_TO_ID["HI"] in content_lid
    assert LID_TAG_TO_ID["EN"] in content_lid
    assert LID_TAG_TO_ID["OTHER"] in content_lid

    # Build compact Whisper + PEFT LoRA + LID Head
    bundle = build_whisper_lora_lid_model(
        base_model_name_or_path="nonexistent/offline-unit-test-whisper",
        lora_r=4,
        lora_alpha=8,
        decoder_attention_only=True,
        use_lid_head=True,
        local_tiny_fallback_if_offline=True,
        device="cpu",
    )
    peft_model = bundle["peft_model"]
    lid_head = bundle["lid_head"]
    assert lid_head is not None

    # Verify every trainable parameter in peft_model belongs exclusively to decoder attention LoRA layers
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name
            assert "decoder.layers." in name
            assert (".self_attn." in name) or (".encoder_attn." in name)
            assert "encoder.layers." not in name

    # Verify joint ASR + LID loss computation and backward pass
    mel = torch.randn(80, 3000)
    batch = collate_whisper_lid_batch([{"input_features": mel, "token_ids": t_ids, "lid_labels": lid_ids}])
    loss_dict = compute_joint_asr_lid_loss(
        peft_model=peft_model,
        lid_head=lid_head,
        input_features=batch["input_features"],
        labels=batch["labels"],
        lid_labels=batch["lid_labels"],
        lid_loss_weight=0.3,
    )
    assert loss_dict["total_loss"].requires_grad
    assert loss_dict["lid_loss"].item() > 0.0
    loss_dict["total_loss"].backward()

    # Save and verify checkpoint reload equivalence
    ckpt_dir = tmp_path / "saved_lora_lid"
    save_finetuned_whisper_lora_lid(
        output_dir=ckpt_dir,
        peft_model=peft_model,
        lid_head=lid_head,
        processor=None,
        training_config={"experiment_name": "unit_test_lora_lid", "base_model_name_or_path": "tiny_fallback"},
    )
    reload_report = verify_checkpoint_reload(
        checkpoint_dir=ckpt_dir,
        original_peft_model=peft_model,
        original_lid_head=lid_head,
    )
    assert reload_report["reload_verified"] is True
    assert reload_report["asr_logits_match"] is True
    assert reload_report["lid_logits_match"] is True


def test_curriculum_stage_filtering_and_end_to_end_pipeline(tmp_path: Path) -> None:
    """Test curriculum stage filtering (Monolingual -> Real Mixed -> Synthetic Supplement)
    and end-to-end evaluation + export pipeline.
    """
    _, records = build_demo_fixture(tmp_path / "demo_fix")
    stage1 = filter_records_for_curriculum_stage(records, "monolingual")
    stage2 = filter_records_for_curriculum_stage(records, "real_mixed")
    stage3 = filter_records_for_curriculum_stage(records, "all_with_synthetic")

    assert all(compute_cmi(r.lang_tags) == 0.0 and not r.is_synthetic for r in stage1)
    assert all(compute_cmi(r.lang_tags) > 0.0 and not r.is_synthetic for r in stage2)
    assert any(r.is_synthetic for r in stage3)

    # Verify CTC dependency block on un-finetuned facebook/wav2vec2-xls-r-300m
    xlsr_status = verify_checkpoint_status("facebook/wav2vec2-xls-r-300m")
    assert xlsr_status["requires_ctc_head_training"] is True
    assert xlsr_status["status"] == "requires_ctc_head_training"
    with pytest.raises(RuntimeError, match="CTC Head Dependency Error"):
        Wav2Vec2CTCRunner("facebook/wav2vec2-xls-r-300m")

    # End-to-end corpus evaluation and export
    test_recs = [r for r in records if r.split == "test"]
    base_hyps = {r.utterance_id: "kal eating ke baad main report submit dunga" for r in test_recs}
    ft_hyps = {r.utterance_id: r.transcript for r in test_recs}

    s_base = evaluate_corpus(test_recs, base_hyps, model_name="baseline_whisper")
    s_ft = evaluate_corpus(test_recs, ft_hyps, model_name="finetuned_whisper_lora_lid")

    assert s_ft.boundary_wer["rate"] < s_base.boundary_wer["rate"]
    exported = export_evaluation_results([s_base, s_ft], tmp_path / "eval_out", baseline_model_name="baseline_whisper")
    assert exported["comparison_md"].exists()
    md_text = exported["comparison_md"].read_text(encoding="utf-8")
    assert "YES (" in md_text

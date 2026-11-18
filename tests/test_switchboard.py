"""Comprehensive 23-test suite for SwitchBoard covering normalization, switch detection, insertion attribution,
hand-calculated Boundary/Non-Boundary WER, CMI, split leakage, real Whisper loading, PEFT reload, and research guardrails.
"""

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
from switchboard.models.registry import verify_checkpoint_status
from switchboard.models.runners import Wav2Vec2CTCRunner, WhisperBaselineRunner, WhisperPeftRunner
from switchboard.normalization.normalizer import TranscriptNormalizer
from switchboard.training.whisper_lora_lid import (
    IGNORE_INDEX,
    LID_TAG_TO_ID,
    LazyWhisperLIDDataset,
    SimpleWordSubwordTokenizer,
    align_word_tags_to_subwords,
    build_whisper_lora_lid_model,
    collate_whisper_lid_batch,
    compute_joint_asr_lid_loss,
    detect_cpu_hardware_info,
    filter_records_for_curriculum_stage,
    save_finetuned_whisper_lora_lid,
    snapshot_parameter_fingerprints,
    verify_checkpoint_reload,
    verify_parameter_updates,
)


# 1. Unicode normalization
def test_01_unicode_normalization() -> None:
    normalizer = TranscriptNormalizer()
    # Composed vs decomposed NFC + zero-width joiner removal + punctuation/danda stripping
    decomposed = "मीटिंग\u200b।"
    assert normalizer.normalize_word(decomposed) == "meeting"
    assert normalizer.normalize_text("  कल\u200d   मीटिंग॥  ") == "kal meeting"


# 2. Romanized Hindi
def test_02_romanized_hindi_normalization() -> None:
    normalizer = TranscriptNormalizer()
    assert normalizer.normalize_word("doonga") == "dunga"
    assert normalizer.normalize_word("Dunga!") == "dunga"
    assert normalizer.normalize_word("achha") == "accha"
    assert normalizer.normalize_word("bohot") == "bahut"
    assert normalizer.normalize_word("pahle") == "pehle"


# 3. English in Devanagari
def test_03_english_in_devanagari_normalization() -> None:
    normalizer = TranscriptNormalizer()
    assert normalizer.normalize_word("मीटिंग") == "meeting"
    assert normalizer.normalize_word("रिपोर्ट") == "report"
    assert normalizer.normalize_word("सबमिट") == "submit"
    assert normalizer.normalize_word("क्लिक") == "click"
    assert normalizer.normalize_word("ट्यूटोरियल") == "tutorial"
    assert normalizer.normalize_word("फाइल") == "file"


# 4. Reviewed spelling equivalences
def test_04_reviewed_spelling_equivalences() -> None:
    normalizer = TranscriptNormalizer()
    dev_sentence = "कल मीटिंग के बाद मैं रिपोर्ट सबमिट कर दूँगा।"
    rom_sentence = "Kal meeting ke baad main report submit kar doonga!"
    assert normalizer.normalize_text(dev_sentence) == normalizer.normalize_text(rom_sentence)
    assert normalizer.normalize_text(dev_sentence) == "kal meeting ke baad main report submit kar dunga"


# 5. Language-tag preservation & true word errors still counting
def test_05_language_tag_preservation() -> None:
    normalizer = TranscriptNormalizer()
    ref_text = "कल, मीटिंग के बाद! मैं रिपोर्ट सबमिट कर दूँगा।"
    ref_tags = ["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"]
    norm_ref = normalizer.normalize_reference(ref_text, ref_tags)

    assert norm_ref.raw_transcript == ref_text
    assert norm_ref.normalized_lang_tags == ref_tags
    assert len(norm_ref.normalized_words) == len(norm_ref.normalized_lang_tags) == 9
    assert norm_ref.token_mappings[1].raw_word == "मीटिंग"
    assert norm_ref.token_mappings[1].normalized_word == "meeting"
    assert norm_ref.token_mappings[1].lang_tag == "EN"

    # True lexical errors ('eating' vs 'meeting', 'deport' vs 'report') must still count after normalization
    rec = ManifestRecord(
        utterance_id="utt_tag_pres",
        audio_path="dummy.wav",
        transcript=ref_text,
        lang_tags=ref_tags,
        speaker_id="spk1",
        accent="Delhi",
        source="unit_test",
        split="test",
    )
    res = evaluate_utterance(rec, "kal eating ke baad main deport submit kar dunga", normalizer=normalizer)
    assert res.overall_wer["errors"] == 2
    assert res.overall_wer["substitutions"] == 2


# 6. HI -> EN switch detection
def test_06_hi_to_en_switch_detection() -> None:
    words = ["kal", "meeting", "schedule", "confirmed"]
    tags = ["HI", "EN", "EN", "EN"]
    res = detect_switch_boundaries(words, tags)
    assert len(res.transitions) == 1
    assert res.transitions[0].direction == "HI->EN"
    assert res.transitions[0].pre_index == 0
    assert res.transitions[0].post_index == 1
    assert res.hi_to_en_indices == {0, 1}
    assert res.en_to_hi_indices == set()


# 7. EN -> HI switch detection
def test_07_en_to_hi_switch_detection() -> None:
    words = ["report", "submit", "kar", "dunga"]
    tags = ["EN", "EN", "HI", "HI"]
    res = detect_switch_boundaries(words, tags)
    assert len(res.transitions) == 1
    assert res.transitions[0].direction == "EN->HI"
    assert res.transitions[0].pre_index == 1
    assert res.transitions[0].post_index == 2
    assert res.en_to_hi_indices == {1, 2}
    assert res.hi_to_en_indices == set()


# 8. Consecutive switches
def test_08_consecutive_switches() -> None:
    words = ["kal", "meeting", "ke", "time", "par"]
    tags = ["HI", "EN", "HI", "EN", "HI"]
    res = detect_switch_boundaries(words, tags)
    assert len(res.transitions) == 4
    assert [t.direction for t in res.transitions] == ["HI->EN", "EN->HI", "HI->EN", "EN->HI"]
    assert res.boundary_indices == {0, 1, 2, 3, 4}
    assert res.overlapping_boundary_indices == {1, 2, 3}


# 9. Overlapping boundary words counted once
def test_09_overlapping_boundary_words_counted_once() -> None:
    words = "kal meeting ke baad main report submit kar dunga".split()
    tags = ["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"]
    res = detect_switch_boundaries(words, tags)
    assert len(res.transitions) == 4
    # 'meeting' (index 1) is touched by kal->meeting and meeting->ke, counted once in pooled boundary_indices
    assert res.overlapping_boundary_indices == {1}
    assert res.boundary_indices == {0, 1, 2, 4, 5, 6, 7}
    assert len(res.boundary_indices) == 7
    assert res.non_boundary_indices == {3, 8}


# 10. OTHER token behavior
def test_10_other_token_behavior() -> None:
    words = ["Rahul", "umm", "kal", "meeting", "हम्म", "है"]
    tags = ["OTHER", "OTHER", "HI", "EN", "OTHER", "HI"]
    res_strict = detect_switch_boundaries(words, tags, bridge_other=False)
    assert len(res_strict.transitions) == 1
    assert res_strict.boundary_indices == {2, 3}
    assert 0 not in res_strict.boundary_indices
    assert 1 not in res_strict.boundary_indices
    assert 4 not in res_strict.boundary_indices


# 11. Beginning insertions
def test_11_beginning_insertions() -> None:
    ref_words = ["kal", "meeting", "hai"]
    hyp_words = ["hello", "kal", "meeting", "hai"]
    ops, attrs = align_and_attribute_edits(ref_words, hyp_words, ["HI", "EN", "HI"])
    ins_ops = [op for op in ops if op.op == "INS"]
    assert len(ins_ops) == 1
    assert ins_ops[0].hyp_word == "hello"
    assert ins_ops[0].attributed_ref_index == 0
    assert attrs[0].insertions == 1


# 12. Internal insertions
def test_12_internal_insertions() -> None:
    ref_words = ["kal", "meeting", "hai"]
    hyp_words = ["kal", "meeting", "abhi", "hai"]
    ops, attrs = align_and_attribute_edits(ref_words, hyp_words, ["HI", "EN", "HI"])
    ins_ops = [op for op in ops if op.op == "INS"]
    assert len(ins_ops) == 1
    assert ins_ops[0].hyp_word == "abhi"
    assert ins_ops[0].attributed_ref_index == 1  # preceding reference position
    assert attrs[1].insertions == 1


# 13. Ending insertions
def test_13_ending_insertions() -> None:
    ref_words = ["kal", "meeting", "hai"]
    hyp_words = ["kal", "meeting", "hai", "thanks"]
    ops, attrs = align_and_attribute_edits(ref_words, hyp_words, ["HI", "EN", "HI"])
    ins_ops = [op for op in ops if op.op == "INS"]
    assert len(ins_ops) == 1
    assert ins_ops[0].hyp_word == "thanks"
    assert ins_ops[0].attributed_ref_index == 2  # last reference position N-1
    assert attrs[2].insertions == 1


# 14. Empty references
def test_14_empty_reference_utterance_level_insertion() -> None:
    ops, attrs = align_and_attribute_edits([], ["spurious", "words"], [])
    assert len(attrs) == 0
    assert len(ops) == 2
    assert all(op.op == "UTT_INS" and op.attributed_ref_index == -1 for op in ops)


# 15. CMI calculation
def test_15_cmi_calculation() -> None:
    # 6 HI, 3 EN -> 100 * (1 - 6/9) = 33.3333
    assert math.isclose(compute_cmi(["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"]), 33.3333, abs_tol=1e-3)
    # Monolingual + OTHER -> 0.0
    assert compute_cmi(["OTHER", "HI", "HI", "HI"]) == 0.0
    # 50/50 balanced + OTHER -> 50.0
    assert math.isclose(compute_cmi(["OTHER", "HI", "EN", "HI", "EN"]), 50.0, abs_tol=1e-4)
    # All OTHER -> 0.0
    assert compute_cmi(["OTHER", "OTHER"]) == 0.0


# 16. Hand-calculated Boundary WER
def test_16_hand_calculated_boundary_wer() -> None:
    rec = ManifestRecord(
        utterance_id="utt_hand_b",
        audio_path="dummy.wav",
        transcript="kal meeting ke baad main report submit kar dunga",
        lang_tags=["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"],
        speaker_id="spk_hand",
        accent="North Indian Hinglish",
        source="hand_calc",
        split="test",
        duration_sec=4.0,
    )
    # SUB at index 1 ('meeting'->'eating', boundary), DEL at index 7 ('kar', boundary), INS after index 8 ('extra', non-boundary)
    hyp = "kal eating ke baad main report submit dunga extra"
    res = evaluate_utterance(rec, hyp, transcription_time_sec=0.8)
    assert res.boundary_wer["errors"] == 2
    assert res.boundary_wer["words"] == 7
    assert math.isclose(res.boundary_wer["rate"], 2.0 / 7.0, abs_tol=1e-6)


# 17. Hand-calculated Non-Boundary WER
def test_17_hand_calculated_non_boundary_wer() -> None:
    rec = ManifestRecord(
        utterance_id="utt_hand_nb",
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
    assert res.non_boundary_wer["errors"] == 1
    assert res.non_boundary_wer["words"] == 2
    assert math.isclose(res.non_boundary_wer["rate"], 1.0 / 2.0, abs_tol=1e-6)
    assert res.overall_wer["errors"] == 3 and res.overall_wer["words"] == 9


# 18. Empty boundary groups marked unavailable (never 0%)
def test_18_empty_boundary_groups_marked_unavailable() -> None:
    rec = ManifestRecord(
        utterance_id="utt_mono",
        audio_path="dummy.wav",
        transcript="कल हम सब घर जाएंगे",
        lang_tags=["HI", "HI", "HI", "HI", "HI"],
        speaker_id="spk_mono",
        accent="Hindi",
        source="unit_test",
        split="test",
    )
    res = evaluate_utterance(rec, "कल हम सब घर जाएंगे")
    assert res.boundary_wer["words"] == 0
    assert res.boundary_wer["rate"] is None
    assert res.boundary_wer["status"] == "unavailable"


# 19. Speaker-disjoint splits
def test_19_speaker_disjoint_splits() -> None:
    r1 = ManifestRecord("u1", "a1.wav", "kal meeting hai", ["HI", "EN", "HI"], "spk_dup", "North", "mucs", "train")
    r2 = ManifestRecord("u2", "a2.wav", "aaj office chalo", ["HI", "EN", "HI"], "spk_dup", "North", "mucs", "test")
    with pytest.raises(ValueError, match="Speaker leakage"):
        validate_manifest_splits([r1, r2])


# 20. Duplicate detection & synthetic isolation
def test_20_duplicate_recording_detection_and_synthetic_isolation() -> None:
    r1 = ManifestRecord("u1", "same.wav", "kal meeting hai", ["HI", "EN", "HI"], "spk1", "North", "mucs", "train", audio_sha256="abc123")
    r2 = ManifestRecord("u2", "other.wav", "kal meeting hai", ["HI", "EN", "HI"], "spk2", "North", "mucs", "val", audio_sha256="abc123")
    with pytest.raises(ValueError, match="Duplicate recording leakage"):
        validate_manifest_splits([r1, r2])

    r_syn_bad = ManifestRecord("u_syn", "syn.wav", "kal report bhejo", ["HI", "EN", "HI"], "spk_s", "Synth", "synthetic_tts", "test", is_synthetic=True)
    with pytest.raises(ValueError, match="Synthetic mixed speech"):
        validate_record(r_syn_bad)


# 21. Model checkpoint loading & CTC dependency verification
def test_21_real_whisper_small_checkpoint_loading_and_ctc_dependency() -> None:
    ws_status = verify_checkpoint_status("openai/whisper-small")
    assert ws_status["is_cached_locally"] is True
    assert ws_status["requires_ctc_head_training"] is False

    xlsr_status = verify_checkpoint_status("facebook/wav2vec2-xls-r-300m")
    assert xlsr_status["requires_ctc_head_training"] is True
    with pytest.raises(RuntimeError, match="CTC Head Dependency Error"):
        Wav2Vec2CTCRunner("facebook/wav2vec2-xls-r-300m")


# 22. Adapter & Auxiliary LID head reload verification
def test_22_peft_lora_and_lid_head_reload_verification(tmp_path: Path) -> None:
    tokenizer = SimpleWordSubwordTokenizer()
    t_ids, lid_ids = align_word_tags_to_subwords(
        ["kal", "meeting", "Rahul"],
        ["HI", "EN", "OTHER"],
        tokenizer,
        prefix_token_ids=[50258, 50259, 50359, 50363],
        eos_token_id=50257,
    )
    assert lid_ids[:4] == [IGNORE_INDEX] * 4
    assert lid_ids[-1] == IGNORE_INDEX

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

    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name and "decoder.layers." in name and "encoder.layers." not in name

    mel = torch.randn(80, 3000)
    batch = collate_whisper_lid_batch([{"input_features": mel, "token_ids": t_ids, "lid_labels": lid_ids}])
    loss_dict = compute_joint_asr_lid_loss(peft_model, lid_head, batch["input_features"], batch["labels"], batch["lid_labels"], 0.3)
    loss_dict["total_loss"].backward()

    ckpt_dir = tmp_path / "saved_lora_lid"
    save_finetuned_whisper_lora_lid(
        ckpt_dir,
        peft_model,
        lid_head,
        None,
        {"experiment_name": "unit_test_lora_lid", "base_model_name_or_path": "tiny_fallback"},
        {"used_tiny_fallback": True},
    )
    reload_report = verify_checkpoint_reload(ckpt_dir, peft_model, lid_head)
    assert reload_report["reload_verified"] is True

    # Verify WhisperPeftRunner refuses to use a randomly initialized tiny fallback checkpoint for speech recognition
    with pytest.raises(RuntimeError, match="randomly initialized tiny fallback"):
        WhisperPeftRunner(ckpt_dir)


# 23. End-to-end real inference, importers, and research export guardrails
def test_23_end_to_end_real_inference_and_research_export_guardrails(tmp_path: Path) -> None:
    # Test audio resampling & importers
    sr_orig = 44100
    t = np.linspace(0, 0.5, int(sr_orig * 0.5), endpoint=False, dtype=np.float32)
    stereo = np.stack([0.4 * np.sin(2 * np.pi * 220 * t), 0.4 * np.cos(2 * np.pi * 440 * t)], axis=1)
    raw_wav = tmp_path / "stereo.wav"
    sf.write(str(raw_wav), stereo, sr_orig)
    out_16k = tmp_path / "mono_16k.wav"
    dur, _ = resample_to_16k_mono(raw_wav, out_16k)
    assert sf.info(str(out_16k)).samplerate == 16000 and sf.info(str(out_16k)).channels == 1

    # Verify research integrity guardrail: simulated or demo records cannot enter full_research evaluation
    _, demo_recs = build_demo_fixture(tmp_path / "demo_fix")
    demo_test = [r for r in demo_recs if r.split == "test"]
    hyps = {r.utterance_id: r.transcript for r in demo_test}
    with pytest.raises(ValueError, match="Demo or synthetic record"):
        evaluate_corpus(demo_test, hyps, run_tier="full_research", is_simulated=False)

    real_rec = ManifestRecord(
        utterance_id="mucs_real_01",
        audio_path=str(out_16k),
        transcript="kal meeting ke baad report submit kar dunga",
        lang_tags=["HI", "EN", "HI", "HI", "EN", "EN", "HI", "HI"],
        speaker_id="mucs_spk_01",
        accent="Indian Hindi-English",
        source="mucs2021_hi_en",
        split="test",
        duration_sec=dur,
    )
    with pytest.raises(ValueError, match="Research Integrity Violation"):
        evaluate_corpus([real_rec], {"mucs_real_01": "kal meeting"}, run_tier="full_research", is_simulated=True)

    real_summary = evaluate_corpus([real_rec], {"mucs_real_01": "kal meeting ke baad report submit kar dunga"}, run_tier="full_research", is_simulated=False)
    exported = export_evaluation_results([real_summary], tmp_path / "full_benchmark_real", forbid_simulated=True)
    assert exported["per_utterance_csv"].exists()
    assert exported["per_word_alignment_csv"].exists()
    assert exported["error_analysis_csv"].exists()
    assert exported["evaluation_config_json"].exists()


# 24. CPU hardware detection, lazy dataset loader, and parameter delta verification
def test_24_cpu_hardware_detection_lazy_loader_and_parameter_delta_verification(tmp_path: Path) -> None:
    hw = detect_cpu_hardware_info()
    assert hw["physical_cores"] >= 1
    assert hw["logical_threads"] >= hw["physical_cores"]
    assert hw["total_ram_gb"] > 0.0

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

    snap_before = snapshot_parameter_fingerprints(peft_model, lid_head)
    params = [p for p in peft_model.parameters() if p.requires_grad] + [p for p in lid_head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)

    mel = torch.randn(80, 3000)
    t_ids = [50258, 50276, 50359, 50363, 200, 300, 50257]
    lid_ids = [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 0, 1, IGNORE_INDEX]
    batch = collate_whisper_lid_batch([{"input_features": mel, "token_ids": t_ids, "lid_labels": lid_ids}])

    # Execute 2 steps so both lora_B (step 1) and lora_A (step 2) update while frozen base stays untouched
    for _ in range(2):
        opt.zero_grad()
        out = compute_joint_asr_lid_loss(peft_model, lid_head, batch["input_features"], batch["labels"], batch["lid_labels"], 0.3)
        out["total_loss"].backward()
        opt.step()

    delta_report = verify_parameter_updates(peft_model, lid_head, snap_before)
    assert delta_report["frozen_base_params_unchanged"] is True
    assert delta_report["frozen_base_max_norm_diff"] == 0.0
    assert delta_report["trainable_lora_params_updated"] is True
    assert delta_report["trainable_lora_max_abs_diff"] > 0.0
    assert delta_report["trainable_lid_head_updated"] is True
    assert delta_report["trainable_lid_head_max_abs_diff"] > 0.0

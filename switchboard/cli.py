"""Command-line interface for SwitchBoard dataset preparation, training, real ASR evaluation, and model verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from switchboard.data.importers import (
    build_demo_fixture,
    import_common_voice,
    import_kathbath_hi,
    import_local_team_recordings,
    import_mucs2021_hi_en,
)
from switchboard.data.manifest import load_manifest, save_manifest, validate_manifest_splits
from switchboard.evaluation.boundary import evaluate_corpus, export_evaluation_results
from switchboard.models.registry import list_registered_models, verify_checkpoint_status
from switchboard.normalization.normalizer import TranscriptNormalizer
from switchboard.training.whisper_lora_lid import CurriculumTrainingRunner


def run_simulated_demo_fixture(base_dir: Path) -> Dict[str, Any]:
    """Run an explicitly labeled SIMULATED classroom demo fixture on the 5 synthetic demo utterances
    and export to `outputs/demo/`.

    WARNING: SIMULATED DATA — NOT REAL MODEL RESULTS.
    Kept strictly isolated in `outputs/demo/` with `is_simulated=True` and `run_tier="demo_simulated"`.
    Cannot be exported to `outputs/full_benchmark_real/`.
    """
    demo_dir = base_dir / "data" / "demo"
    manifest_path = demo_dir / "manifest.jsonl"
    if not manifest_path.exists():
        build_demo_fixture(demo_dir)

    all_records = load_manifest(manifest_path, validate=True)
    test_records = [r for r in all_records if r.split == "test"]
    normalizer = TranscriptNormalizer(base_dir / "configs" / "normalization_equivalents.yaml")

    simulated_hypotheses_by_model: Dict[str, Dict[str, Any]] = {
        "simulated_baseline_example": {
            "checkpoint_id": "SIMULATED_EXAMPLE_ONLY (not real model inference)",
            "decoding": {"mode": "simulated_classroom_demo_fixture", "is_simulated": True},
            "hyps": {
                "demo_test_utt_01": "kal eating ke baad main report submit dunga",
                "demo_test_utt_02": "aaj meeting ke baad main report sabhi kar dunga",
                "demo_test_utt_03": "Rahul umm kal off time par mail bhej देना",
                "demo_test_utt_04": "abhi court branch review aur model train shuru karo",
                "demo_test_utt_05": "हम्म कल सब लोग सुबह जल्दी आएंगे",
            },
            "times": {u.utterance_id: 0.42 for u in test_records},
        },
        "simulated_perfect_reference_example": {
            "checkpoint_id": "SIMULATED_ORACLE_EXAMPLE_ONLY (not real model inference)",
            "decoding": {"mode": "simulated_classroom_demo_fixture", "is_simulated": True},
            "hyps": {u.utterance_id: u.transcript for u in test_records},
            "times": {u.utterance_id: 0.45 for u in test_records},
        },
    }

    summaries = []
    for m_name, spec in simulated_hypotheses_by_model.items():
        summary = evaluate_corpus(
            records=test_records,
            hypotheses=spec["hyps"],
            model_name=m_name,
            checkpoint_id=spec["checkpoint_id"],
            run_tier="demo_simulated",
            decoding_settings=spec["decoding"],
            transcription_times=spec["times"],
            normalizer=normalizer,
            is_simulated=True,
        )
        summaries.append(summary)

    out_dir = base_dir / "outputs" / "demo"
    exported = export_evaluation_results(
        summaries=summaries,
        output_dir=out_dir,
        baseline_model_name="simulated_baseline_example",
        forbid_simulated=False,
    )
    return {
        "warning": "SIMULATED DATA — NOT REAL MODEL RESULTS",
        "is_simulated": True,
        "run_tier": "demo_simulated",
        "output_dir": str(out_dir),
        "exported_files": {k: str(v) for k, v in exported.items()},
        "models_evaluated": [s.model_name for s in summaries],
    }


def select_deterministic_test_subset(
    records: List[Any],
    max_utterances: int = 25,
) -> List[Any]:
    """Select a deterministic, speaker-stratified subset of held-out test utterances for reproducible evaluation."""
    test_records = [r for r in records if r.split == "test"]
    if not max_utterances or len(test_records) <= max_utterances:
        return test_records

    # Prefer utterances with both HI and EN tags so boundary WER is exercised across all speakers
    by_spk: Dict[str, List[Any]] = {}
    for r in test_records:
        by_spk.setdefault(r.speaker_id, []).append(r)

    speakers = sorted(by_spk.keys())
    selected: List[Any] = []
    per_spk = max(1, max_utterances // max(1, len(speakers)))

    for spk in speakers:
        spk_utts = by_spk[spk]
        cs_utts = [
            u for u in spk_utts
            if ("HI" in u.lang_tags and "EN" in u.lang_tags) and (1.5 <= float(u.duration_sec or 4.0) <= 9.0)
        ]
        pool = cs_utts if len(cs_utts) >= per_spk else spk_utts
        selected.extend(pool[:per_spk])

    if len(selected) < max_utterances:
        seen_ids = {r.utterance_id for r in selected}
        for r in test_records:
            if r.utterance_id not in seen_ids:
                selected.append(r)
                seen_ids.add(r.utterance_id)
            if len(selected) >= max_utterances:
                break

    return selected[:max_utterances]


def _load_saved_real_hyps_if_complete(
    out_dir: Path,
    candidate_stems: List[str],
    expected_utt_ids: List[str],
) -> Optional[Tuple[Dict[str, str], Dict[str, float]]]:
    """If a previous real evaluation in `out_dir` already transcribed the exact `expected_utt_ids` for an unchanged checkpoint,
    return `(hypotheses, times)` to avoid redundant CPU re-decoding.
    """
    for stem in candidate_stems:
        p = out_dir / f"{stem}_per_utterance.jsonl"
        if not p.exists():
            continue
        hyps: Dict[str, str] = {}
        times: Dict[str, float] = {}
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("is_simulated"):
                    break
                uid = item["utterance_id"]
                hyps[uid] = item["raw_hypothesis"]
                times[uid] = float(item.get("transcription_time_sec") or 1.5)
            if all(uid in hyps for uid in expected_utt_ids):
                return {u: hyps[u] for u in expected_utt_ids}, {u: times[u] for u in expected_utt_ids}
        except Exception:
            continue
    return None


def run_real_model_benchmark(
    base_dir: Path,
    manifest_rel_or_abs: str | Path = "data/full_corpus/unified_manifest.jsonl",
    max_utterances: int = 25,
    output_dir_rel_or_abs: str | Path = "outputs/full_benchmark_real",
    run_tier: str = "full_research",
    include_comparison_models: bool = True,
    include_peft_checkpoints: bool = True,
    reuse_cached_predictions: bool = True,
) -> Dict[str, Any]:
    """Run REAL neural speech recognition inference and switch-boundary evaluation on held-out test utterances.

    Never fabricates or simulates hypotheses. Every model prediction is produced by an actual PyTorch forward pass
    through the loaded checkpoint on the 16 kHz mono WAV audio files.
    """
    from switchboard.models.runners import (
        IndicWhisperRunner,
        Wav2Vec2CTCRunner,
        WhisperBaselineRunner,
        WhisperPeftRunner,
    )

    manifest_path = Path(manifest_rel_or_abs)
    if not manifest_path.is_absolute():
        manifest_path = base_dir / manifest_path

    all_records = load_manifest(
        manifest_path,
        validate=True,
    )
    test_subset = select_deterministic_test_subset(all_records, max_utterances=max_utterances)
    if not test_subset:
        raise ValueError(f"No test records found in {manifest_path}")

    normalizer = TranscriptNormalizer(base_dir / "configs" / "normalization_equivalents.yaml")
    summaries = []
    skipped_or_blocked: List[Dict[str, str]] = []
    out_dir = Path(output_dir_rel_or_abs)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    expected_uids = [r.utterance_id for r in test_subset]

    # 1. Primary Baseline: openai/whisper-small
    cached_base = (
        _load_saved_real_hyps_if_complete(out_dir, ["openai_whisper-small_(baseline)"], expected_uids)
        if reuse_cached_predictions
        else None
    )
    if cached_base is not None:
        hyps, times = cached_base
        dec_cfg = {"task": "transcribe", "num_beams": 1, "no_repeat_ngram_size": 3, "device": "cpu"}
    else:
        primary_runner = WhisperBaselineRunner(checkpoint_id="openai/whisper-small")
        hyps, times, _ = primary_runner.transcribe_manifest(test_subset, base_dir=base_dir)
        dec_cfg = primary_runner.get_decoding_settings()
        del primary_runner
    summaries.append(
        evaluate_corpus(
            records=test_subset,
            hypotheses=hyps,
            model_name="openai/whisper-small (baseline)",
            checkpoint_id="openai/whisper-small",
            run_tier=run_tier,
            decoding_settings=dec_cfg,
            transcription_times=times,
            normalizer=normalizer,
            is_simulated=False,
        )
    )

    # 2. PEFT Fine-Tuned Checkpoints (6-step smoke checkpoints + CPU exploratory checkpoints)
    if include_peft_checkpoints:
        for ckpt_label, ckpt_rel, candidate_stems in [
            (
                "whisper-small + LoRA only (6-step smoke)",
                "checkpoints/whisper_small_lora_only",
                ["whisper-small_+_lora_only_(6-step_smoke)", "whisper-small_+_lora_only"],
            ),
            (
                "whisper-small + LoRA + LID head (6-step smoke)",
                "checkpoints/whisper_small_lora_plus_lid",
                ["whisper-small_+_lora_+_lid_head_(6-step_smoke)", "whisper-small_+_lora_+_lid_head"],
            ),
            (
                "whisper-small + LoRA only (CPU exploratory)",
                "checkpoints/whisper_small_lora_only_cpu",
                ["whisper-small_+_lora_only_(cpu_exploratory)"],
            ),
            (
                "whisper-small + LoRA + LID head (CPU exploratory)",
                "checkpoints/whisper_small_lora_plus_lid_cpu",
                ["whisper-small_+_lora_+_lid_head_(cpu_exploratory)"],
            ),
        ]:
            ckpt_p = base_dir / ckpt_rel
            if (ckpt_p / "lora_adapter").exists():
                try:
                    cached_peft = (
                        _load_saved_real_hyps_if_complete(out_dir, candidate_stems, expected_uids)
                        if reuse_cached_predictions
                        else None
                    )
                    if cached_peft is not None:
                        p_hyps, p_times = cached_peft
                        p_dec = {"task": "transcribe", "num_beams": 1, "no_repeat_ngram_size": 3, "device": "cpu"}
                    else:
                        peft_runner = WhisperPeftRunner(adapter_dir=ckpt_p)
                        p_hyps, p_times, _ = peft_runner.transcribe_manifest(test_subset, base_dir=base_dir)
                        p_dec = peft_runner.get_decoding_settings()
                        del peft_runner
                    summaries.append(
                        evaluate_corpus(
                            records=test_subset,
                            hypotheses=p_hyps,
                            model_name=ckpt_label,
                            checkpoint_id=ckpt_rel,
                            run_tier=run_tier,
                            decoding_settings=p_dec,
                            transcription_times=p_times,
                            normalizer=normalizer,
                            is_simulated=False,
                        )
                    )
                except Exception as exc:
                    skipped_or_blocked.append({"model": ckpt_label, "checkpoint": ckpt_rel, "reason": str(exc)})
            else:
                skipped_or_blocked.append(
                    {"model": ckpt_label, "checkpoint": ckpt_rel, "reason": "Checkpoint directory not found"}
                )

    # 3. Additional Comparison Models (IndicWhisper & wav2vec 2.0 CTC)
    if include_comparison_models:
        # 3a. AI4Bharat IndicWhisper (vasista22/whisper-hindi-small)
        indic_status = verify_checkpoint_status("vasista22/whisper-hindi-small")
        if indic_status["is_cached_locally"]:
            try:
                cached_ind = (
                    _load_saved_real_hyps_if_complete(
                        out_dir, ["indicwhisper_(vasista22_whisper-hindi-small)"], expected_uids
                    )
                    if reuse_cached_predictions
                    else None
                )
                if cached_ind is not None:
                    i_hyps, i_times = cached_ind
                    i_dec = {"task": "transcribe", "language": "hi", "num_beams": 1, "no_repeat_ngram_size": 3, "device": "cpu"}
                else:
                    ind_runner = IndicWhisperRunner(checkpoint_id="vasista22/whisper-hindi-small")
                    i_hyps, i_times, _ = ind_runner.transcribe_manifest(test_subset, base_dir=base_dir)
                    i_dec = ind_runner.get_decoding_settings()
                    del ind_runner
                summaries.append(
                    evaluate_corpus(
                        records=test_subset,
                        hypotheses=i_hyps,
                        model_name="indicwhisper (vasista22/whisper-hindi-small)",
                        checkpoint_id="vasista22/whisper-hindi-small",
                        run_tier=run_tier,
                        decoding_settings=i_dec,
                        transcription_times=i_times,
                        normalizer=normalizer,
                        is_simulated=False,
                    )
                )
            except Exception as exc:
                skipped_or_blocked.append(
                    {"model": "indicwhisper", "checkpoint": "vasista22/whisper-hindi-small", "reason": str(exc)}
                )
        else:
            skipped_or_blocked.append(
                {
                    "model": "indicwhisper",
                    "checkpoint": "vasista22/whisper-hindi-small",
                    "reason": "Not cached locally; download required",
                }
            )

        # 3b. wav2vec 2.0 XLS-R Base (explicit CTC head dependency block)
        skipped_or_blocked.append(
            {
                "model": "wav2vec2_xlsr_300m_base",
                "checkpoint": "facebook/wav2vec2-xls-r-300m",
                "reason": "BLOCKED: requires_ctc_head_training (SSL base encoder without trained lm_head or vocab.json)",
            }
        )

        # 3c. Fine-Tuned wav2vec 2.0 Hindi CTC (Harveenchadha/vakyansh-wav2vec2-hindi-him-4200)
        ctc_status = verify_checkpoint_status("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
        if ctc_status["is_cached_locally"]:
            try:
                ctc_runner = Wav2Vec2CTCRunner(checkpoint_id="Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
                c_hyps, c_times, _ = ctc_runner.transcribe_manifest(test_subset, base_dir=base_dir)
                summaries.append(
                    evaluate_corpus(
                        records=test_subset,
                        hypotheses=c_hyps,
                        model_name="wav2vec2_ctc (vakyansh-hindi-him-4200)",
                        checkpoint_id="Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
                        run_tier=run_tier,
                        decoding_settings=ctc_runner.get_decoding_settings(),
                        transcription_times=c_times,
                        normalizer=normalizer,
                        is_simulated=False,
                    )
                )
                del ctc_runner
            except Exception as exc:
                skipped_or_blocked.append(
                    {
                        "model": "wav2vec2_ctc",
                        "checkpoint": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
                        "reason": str(exc),
                    }
                )
        else:
            skipped_or_blocked.append(
                {
                    "model": "wav2vec2_ctc",
                    "checkpoint": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
                    "reason": "Not cached locally; download required",
                }
            )

    out_dir = Path(output_dir_rel_or_abs)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir

    exported = export_evaluation_results(
        summaries=summaries,
        output_dir=out_dir,
        baseline_model_name="openai/whisper-small (baseline)",
        forbid_simulated=(run_tier == "full_research"),
    )

    # Record blocked/skipped models inside evaluation_config.json
    eval_cfg_p = out_dir / "evaluation_config.json"
    if eval_cfg_p.exists():
        cfg_data = json.loads(eval_cfg_p.read_text(encoding="utf-8"))
        cfg_data["skipped_or_blocked_models"] = skipped_or_blocked
        cfg_data["evaluated_utterance_ids"] = [r.utterance_id for r in test_subset]
        eval_cfg_p.write_text(json.dumps(cfg_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "is_simulated": False,
        "run_tier": run_tier,
        "output_dir": str(out_dir),
        "num_test_utterances_evaluated": len(test_subset),
        "test_speakers": sorted({r.speaker_id for r in test_subset}),
        "models_evaluated": [s.model_name for s in summaries],
        "skipped_or_blocked_models": skipped_or_blocked,
        "exported_files": {k: str(v) for k, v in exported.items()},
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="switchboard",
        description="SwitchBoard — Hindi–English Code-Switching Speech Recognition & Boundary Error Analysis CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. build-demo-fixture
    p_demo = subparsers.add_parser("build-demo-fixture", help="Generate the self-contained 16 kHz mono demo fixture.")
    p_demo.add_argument("--fixture-dir", default="data/demo", help="Target directory for demo fixture.")

    # 2. import-dataset
    p_imp = subparsers.add_parser("import-dataset", help="Import an external or local dataset into the unified manifest.")
    p_imp.add_argument(
        "--dataset",
        required=True,
        choices=["mucs2021", "kathbath", "common_voice_hi", "common_voice_en", "team_local"],
    )
    p_imp.add_argument("--input-dir", required=True, help="Path to raw extracted dataset directory.")
    p_imp.add_argument("--audio-output-dir", default="data/processed/audio_16k", help="Output dir for 16 kHz mono WAVs.")
    p_imp.add_argument("--output-manifest", required=True, help="Output JSONL manifest path.")

    # 3. validate-manifest
    p_val = subparsers.add_parser("validate-manifest", help="Validate manifest schema, audio existence, and split rules.")
    p_val.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    p_val.add_argument("--check-audio", action="store_true", help="Verify every audio file exists on disk.")
    p_val.add_argument("--forbid-demo", action="store_true", help="Reject demo or synthetic records.")

    # 4. verify-models
    subparsers.add_parser("verify-models", help="Check baseline & comparison model checkpoints, licenses, and CTC status.")

    # 5. train
    p_train = subparsers.add_parser("train", help="Run Whisper LoRA (+ optional LID head) curriculum fine-tuning.")
    p_train.add_argument("--config", required=True, help="Path to training YAML config.")
    p_train.add_argument("--manifest", default=None, help="Optional override for manifest JSONL path.")
    p_train.add_argument("--output-dir", default=None, help="Optional override for checkpoint output directory.")
    p_train.add_argument("--pilot", action="store_true", help="Run a short CPU pilot (16 utterances, 2 steps/stage, batch_size=1) to verify training & memory before full CPU run.")
    p_train.add_argument("--resume", action="store_true", help="Resume from latest valid checkpoint in output_dir.")

    # 6. simulate-demo-fixture (explicitly labeled classroom simulation)
    subparsers.add_parser(
        "simulate-demo-fixture",
        help="Generate clearly labeled SIMULATED classroom demo outputs in outputs/demo/ (NOT real model results).",
    )

    # 7. evaluate-real-benchmark (runs REAL neural inference on held-out test utterances)
    p_real = subparsers.add_parser(
        "evaluate-real-benchmark",
        help="Run REAL neural model transcription and switch-boundary evaluation on held-out test utterances.",
    )
    p_real.add_argument("--manifest", default="data/full_corpus/unified_manifest.jsonl", help="Manifest JSONL path.")
    p_real.add_argument("--max-utterances", type=int, default=20, help="Number of deterministic held-out test utterances.")
    p_real.add_argument("--output-dir", default="outputs/full_benchmark_real", help="Output directory for real metrics.")
    p_real.add_argument("--run-tier", default="full_research", choices=["smoke_test", "full_research"])

    # 8. evaluate-live (single-model live evaluation)
    p_eval = subparsers.add_parser(
        "evaluate-live",
        help="Run live ASR transcription and switch-boundary evaluation for a single model on a manifest split.",
    )
    p_eval.add_argument("--manifest", default="data/full_corpus/unified_manifest.jsonl", help="Manifest JSONL path.")
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--max-utterances", type=int, default=20, help="Max utterances to transcribe (0 for all).")
    p_eval.add_argument("--checkpoint", default="openai/whisper-small", help="HF checkpoint ID or local PEFT dir.")
    p_eval.add_argument(
        "--runner-type",
        default="whisper_baseline",
        choices=["whisper_baseline", "whisper_peft", "indicwhisper", "wav2vec2_ctc"],
    )
    p_eval.add_argument("--run-tier", default="full_research", choices=["smoke_test", "full_research"])
    p_eval.add_argument("--output-dir", default="outputs/full_benchmark_real", help="Directory to export metrics.")

    args = parser.parse_args(argv)
    base_dir = Path(__file__).resolve().parents[1]

    if args.command == "build-demo-fixture":
        manifest_path, records = build_demo_fixture(args.fixture_dir)
        summary = validate_manifest_splits(records, check_audio_exists=True)
        print(json.dumps({"manifest_path": str(manifest_path), "validation": summary}, indent=2))
        return 0

    if args.command == "import-dataset":
        if args.dataset == "mucs2021":
            recs = import_mucs2021_hi_en(args.input_dir, args.audio_output_dir)
        elif args.dataset == "kathbath":
            recs = import_kathbath_hi(args.input_dir, args.audio_output_dir)
        elif args.dataset == "common_voice_hi":
            recs = import_common_voice(args.input_dir, args.audio_output_dir, locale="hi")
        elif args.dataset == "common_voice_en":
            recs = import_common_voice(args.input_dir, args.audio_output_dir, locale="en")
        else:
            recs = import_local_team_recordings(args.input_dir, args.audio_output_dir)
        out_p = save_manifest(recs, args.output_manifest, validate=True)
        print(json.dumps({"saved_manifest": str(out_p), "records_imported": len(recs)}, indent=2))
        return 0

    if args.command == "validate-manifest":
        recs = load_manifest(args.manifest, validate=False)
        report = validate_manifest_splits(
            recs,
            check_audio_exists=args.check_audio,
            base_dir=base_dir,
            forbid_demo_or_synthetic_in_research=args.forbid_demo,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "verify-models":
        models = list_registered_models()
        statuses = [verify_checkpoint_status(m["checkpoint_id"]) for m in models]
        print(json.dumps({"registered_models": statuses}, indent=2))
        return 0

    if args.command == "train":
        runner = CurriculumTrainingRunner(args.config)
        metrics = runner.run(
            manifest_override=args.manifest,
            output_dir_override=args.output_dir,
            base_dir=base_dir,
            pilot_mode=bool(args.pilot),
            resume=bool(args.resume),
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "simulate-demo-fixture":
        res = run_simulated_demo_fixture(base_dir)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "evaluate-real-benchmark":
        res = run_real_model_benchmark(
            base_dir=base_dir,
            manifest_rel_or_abs=args.manifest,
            max_utterances=args.max_utterances,
            output_dir_rel_or_abs=args.output_dir,
            run_tier=args.run_tier,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "evaluate-live":
        from switchboard.models.runners import (
            IndicWhisperRunner,
            Wav2Vec2CTCRunner,
            WhisperBaselineRunner,
            WhisperPeftRunner,
        )

        all_recs = load_manifest(args.manifest, validate=True)
        if args.split == "test" and args.max_utterances > 0:
            recs = select_deterministic_test_subset(all_recs, max_utterances=args.max_utterances)
        else:
            recs = [r for r in all_recs if r.split == args.split]
            if args.max_utterances > 0:
                recs = recs[: args.max_utterances]

        if args.runner_type == "whisper_baseline":
            asr = WhisperBaselineRunner(checkpoint_id=args.checkpoint)
        elif args.runner_type == "whisper_peft":
            asr = WhisperPeftRunner(adapter_dir=args.checkpoint)
        elif args.runner_type == "indicwhisper":
            asr = IndicWhisperRunner(checkpoint_id=args.checkpoint)
        else:
            asr = Wav2Vec2CTCRunner(checkpoint_id=args.checkpoint)

        hyps, times, _ = asr.transcribe_manifest(recs, base_dir=base_dir)
        summary = evaluate_corpus(
            records=recs,
            hypotheses=hyps,
            model_name=asr.model_name,
            checkpoint_id=asr.checkpoint_id,
            run_tier=args.run_tier,
            decoding_settings=asr.get_decoding_settings(),
            transcription_times=times,
            is_simulated=False,
        )
        exported = export_evaluation_results(
            [summary],
            args.output_dir,
            forbid_simulated=(args.run_tier == "full_research"),
        )
        print(json.dumps({"exported": {k: str(v) for k, v in exported.items()}}, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

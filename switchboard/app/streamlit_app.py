"""SwitchBoard Streamlit Application — Hindi–English Code-Switching ASR & Switch-Boundary Error Analysis."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from switchboard.cli import run_simulated_demo_fixture
from switchboard.data.importers import build_demo_fixture, infer_word_tag
from switchboard.data.manifest import ManifestRecord, load_manifest, resample_to_16k_mono
from switchboard.evaluation.boundary import (
    compute_cmi,
    detect_switch_boundaries,
    evaluate_utterance,
)
from switchboard.models.registry import list_registered_models, verify_checkpoint_status
from switchboard.normalization.normalizer import TranscriptNormalizer


st.set_page_config(
    page_title="SwitchBoard — Hindi–English Code-Switching ASR",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource
def get_normalizer() -> TranscriptNormalizer:
    return TranscriptNormalizer(PROJECT_ROOT / "configs" / "normalization_equivalents.yaml")


@st.cache_resource
def get_cached_whisper_baseline_runner(checkpoint_id: str = "openai/whisper-small") -> Any:
    from switchboard.models.runners import WhisperBaselineRunner

    return WhisperBaselineRunner(checkpoint_id=checkpoint_id)


@st.cache_resource
def get_cached_whisper_peft_runner(adapter_dir_str: str) -> Any:
    from switchboard.models.runners import WhisperPeftRunner

    return WhisperPeftRunner(adapter_dir=adapter_dir_str)


def ensure_app_assets() -> Tuple[List[ManifestRecord], List[ManifestRecord]]:
    demo_dir = PROJECT_ROOT / "data" / "demo"
    demo_manifest_p = demo_dir / "manifest.jsonl"
    if not demo_manifest_p.exists():
        build_demo_fixture(demo_dir)
    demo_eval_p = PROJECT_ROOT / "outputs" / "demo" / "all_models_summary.json"
    if not demo_eval_p.exists():
        run_simulated_demo_fixture(PROJECT_ROOT)
    demo_recs = load_manifest(demo_manifest_p, validate=True)

    real_manifest_p = PROJECT_ROOT / "data" / "full_corpus" / "unified_manifest.jsonl"
    real_recs: List[ManifestRecord] = []
    if real_manifest_p.exists():
        try:
            real_recs = load_manifest(real_manifest_p, validate=False)
        except Exception:
            real_recs = []
    return demo_recs, real_recs


def load_saved_real_predictions() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load actual model predictions saved in `outputs/full_benchmark_real/*_per_utterance.jsonl`."""
    real_dir = PROJECT_ROOT / "outputs" / "full_benchmark_real"
    preds_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not real_dir.exists():
        return preds_by_model

    for jsonl_p in real_dir.glob("*_per_utterance.jsonl"):
        model_key = jsonl_p.stem.replace("_per_utterance", "")
        preds_by_model[model_key] = {}
        for line in jsonl_p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            preds_by_model[model_key][item["utterance_id"]] = item
    return preds_by_model


def check_peft_checkpoint_valid(ckpt_dir: Path) -> Tuple[bool, str]:
    """Check whether a fine-tuned PEFT checkpoint exists and is built on a real pretrained Whisper base model."""
    if not (ckpt_dir / "lora_adapter").exists():
        return False, f"Checkpoint not found at `{ckpt_dir}`."
    metrics_p = ckpt_dir / "training_metrics.json"
    if metrics_p.exists():
        try:
            m = json.loads(metrics_p.read_text(encoding="utf-8"))
            if m.get("used_tiny_fallback"):
                return (
                    False,
                    f"Checkpoint at `{ckpt_dir}` used a randomly initialized tiny fallback architecture "
                    f"and is blocked from being used as a speech recognition model.",
                )
        except Exception:
            pass
    return True, f"Valid PEFT checkpoint ready at `{ckpt_dir}`."


def render_boundary_highlighted_html(words: List[str], lang_tags: List[str]) -> str:
    """Render HTML badges highlighting switch-adjacent boundary words and their HI/EN/OTHER tags."""
    b_res = detect_switch_boundaries(words, lang_tags)
    chips: List[str] = []
    for idx, (w, tag) in enumerate(zip(words, lang_tags)):
        is_b = idx in b_res.boundary_indices
        is_overlap = idx in b_res.overlapping_boundary_indices
        if tag == "HI":
            tag_bg = "#1e40af"
        elif tag == "EN":
            tag_bg = "#065f46"
        else:
            tag_bg = "#4b5563"

        if is_overlap:
            border = "2px solid #dc2626"
            bg = "#fef2f2"
            badge_note = "⚡⇄2x-Switch"
        elif is_b:
            border = "2px solid #d97706"
            bg = "#fffbeb"
            roles = []
            if idx in b_res.before_switch_indices:
                roles.append("Pre")
            if idx in b_res.after_switch_indices:
                roles.append("Post")
            badge_note = "⚡" + "/".join(roles)
        else:
            border = "1px solid #d1d5db"
            bg = "#f9fafb"
            badge_note = ""

        chip = (
            f'<span style="display:inline-block; margin:4px 5px; padding:6px 10px; '
            f'border-radius:8px; border:{border}; background:{bg}; color:#111827; font-size:0.95rem;">'
            f"<strong>{w}</strong> "
            f'<span style="background:{tag_bg}; color:white; padding:1px 6px; border-radius:4px; font-size:0.75rem;">{tag}</span>'
            f'{f" <span style=\'color:#b45309; font-size:0.75rem; font-weight:600;\'>{badge_note}</span>" if badge_note else ""}'
            f"</span>"
        )
        chips.append(chip)
    return "".join(chips)


def render_alignment_ops_html(alignment_ops: List[Dict[str, Any]], boundary_indices: set[int]) -> str:
    """Render word-by-word edit operations (MATCH, SUB, DEL, INS, UTT_INS) with boundary indicators."""
    chips: List[str] = []
    for step in alignment_ops:
        op = step["op"]
        attr_idx = step["attributed_ref_index"]
        is_b = attr_idx in boundary_indices
        b_icon = "⚡" if is_b else ""

        if op == "MATCH":
            style = "background:#ecfdf5; border:1px solid #10b981; color:#065f46;"
            label = f"✓ {step['ref_word']} {b_icon}"
        elif op == "SUB":
            style = "background:#fef2f2; border:2px solid #ef4444; color:#991b1b;"
            label = f"SUB [{step['ref_word']} → {step['hyp_word']}] (ref#{attr_idx}{b_icon})"
        elif op == "DEL":
            style = "background:#fff1f2; border:2px dashed #e11d48; color:#881337;"
            label = f"DEL [{step['ref_word']} → ∅] (ref#{attr_idx}{b_icon})"
        elif op == "UTT_INS":
            style = "background:#f5f3ff; border:2px solid #8b5cf6; color:#5b21b6;"
            label = f"UTT_INS [+ {step['hyp_word']}] (empty-ref)"
        else:  # INS
            style = "background:#f5f3ff; border:2px solid #8b5cf6; color:#5b21b6;"
            label = f"INS [+ {step['hyp_word']}] (attr→ref#{attr_idx}{b_icon})"

        chips.append(
            f'<span style="display:inline-block; margin:3px 4px; padding:5px 9px; border-radius:6px; '
            f'font-size:0.88rem; font-family:monospace; {style}">{label}</span>'
        )
    return "".join(chips)


def _fmt_metric_card(m: Dict[str, Any]) -> str:
    if m.get("status") != "available" or m.get("rate") is None:
        return "Unavailable (0/0)"
    return f"{float(m['rate']) * 100.0:.2f}% ({m['errors']}/{m['words']})"


def _fmt_row(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "WER": _fmt_metric_card(m),
        "Errors": m.get("errors", 0),
        "Words": m.get("words", 0),
        "S / D / I": f"{m.get('substitutions', 0)} / {m.get('deletions', 0)} / {m.get('insertions', 0)}",
    }


def main() -> None:
    st.title("🎙️ SwitchBoard — Hindi–English Code-Switching ASR & Boundary Error Analysis")
    st.caption(
        "Comparing Pretrained Baseline Whisper Small (`openai/whisper-small`) vs. PEFT LoRA + Auxiliary Language "
        "Identification (LID) Fine-Tuning at Hindi–English (`HI → EN` and `EN → HI`) Language Switch Boundaries"
    )

    demo_records, real_records = ensure_app_assets()
    demo_test_records = [r for r in demo_records if r.split == "test"]
    real_test_records = [r for r in real_records if r.split == "test"]
    saved_real_preds = load_saved_real_predictions()
    normalizer = get_normalizer()

    tab_live, tab_bench, tab_docs = st.tabs(
        [
            "🔍 1. Interactive Demo & Switch-Boundary Inspector",
            "📊 2. Saved Benchmark Results & Error Analysis",
            "📖 3. Dataset, Provenance & Annotation Guidelines",
        ]
    )

    # =========================================================================
    # TAB 1: INTERACTIVE DEMO & SWITCH-BOUNDARY INSPECTOR
    # =========================================================================
    with tab_live:
        col_src, col_cfg = st.columns([3, 2])
        with col_src:
            st.subheader("Step 1: Select Audio Source")
            source_options = [
                "Real MUCS 2021 Held-Out Test Utterances (Real Whisper Predictions)",
                "Bundled Hinglish Examples (Ready-to-Try Demo Fixture)",
                "Upload Audio File (.wav, .mp3, .flac)",
                "Record via Microphone",
            ] if real_test_records else [
                "Bundled Hinglish Examples (Ready-to-Try Demo Fixture)",
                "Upload Audio File (.wav, .mp3, .flac)",
                "Record via Microphone",
            ]
            input_mode = st.radio(
                "Choose how to provide Hindi–English speech:",
                options=source_options,
                horizontal=True,
            )

        with col_cfg:
            st.subheader("Checkpoint & Inference Settings")
            lora_lid_cpu_dir = PROJECT_ROOT / "checkpoints" / "whisper_small_lora_plus_lid_cpu"
            lora_only_cpu_dir = PROJECT_ROOT / "checkpoints" / "whisper_small_lora_only_cpu"
            lora_lid_dir = PROJECT_ROOT / "checkpoints" / "whisper_small_lora_plus_lid"
            lora_only_dir = PROJECT_ROOT / "checkpoints" / "whisper_small_lora_only"
            colab_full_dir = PROJECT_ROOT / "checkpoints" / "colab_full_whisper_small_lora_plus_lid"

            ckpt_choice = st.selectbox(
                "Fine-Tuned Whisper Checkpoint:",
                options=[
                    "Whisper Small + LoRA + LID Head (CPU Exploratory — checkpoints/whisper_small_lora_plus_lid_cpu)",
                    "Whisper Small + LoRA Only (CPU Exploratory — checkpoints/whisper_small_lora_only_cpu)",
                    "Whisper Small + LoRA + LID Head (6-Step Smoke — checkpoints/whisper_small_lora_plus_lid)",
                    "Whisper Small + LoRA Only (6-Step Smoke — checkpoints/whisper_small_lora_only)",
                    "Full Colab GPU Checkpoint (checkpoints/colab_full_whisper_small_lora_plus_lid)",
                    "Simulate Fine-Tuned Checkpoint Unavailable (Test Missing-Checkpoint Warning)",
                ],
            )
            run_live_inference_btn = st.button(
                "⚡ Run Live PyTorch Transcription on Active Audio",
                help="Executes a live forward pass through `openai/whisper-small` and the selected PEFT checkpoint on the active audio file.",
            )

        selected_audio_path: Optional[Path] = None
        default_ref_text = "kal meeting ke baad main report submit kar dunga"
        default_tags = ["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"]
        selected_utt_id = "demo_test_utt_01"
        active_duration_sec = 3.26
        using_simulated_example = False

        # Find saved real baseline & fine-tuned predictions if available
        base_saved_map = saved_real_preds.get("openai_whisper-small_(baseline)", {})
        ft_lid_saved_map = (
            saved_real_preds.get("whisper-small_+_lora_+_lid_head_(cpu_exploratory)")
            or saved_real_preds.get("whisper-small_+_lora_+_lid_head_(6-step_smoke)")
            or saved_real_preds.get("whisper-small_+_lora_+_lid_head", {})
        )
        ft_only_saved_map = (
            saved_real_preds.get("whisper-small_+_lora_only_(cpu_exploratory)")
            or saved_real_preds.get("whisper-small_+_lora_only_(6-step_smoke)")
            or saved_real_preds.get("whisper-small_+_lora_only", {})
        )

        if "Real MUCS 2021" in input_mode and real_test_records:
            # Put utterances that have saved real Whisper predictions at the top of the selector
            evaluated_ids = set(base_saved_map.keys())
            eval_pool = [r for r in real_test_records if r.utterance_id in evaluated_ids]
            other_cs_pool = [
                r for r in real_test_records
                if r.utterance_id not in evaluated_ids and ("HI" in r.lang_tags and "EN" in r.lang_tags)
            ][:40]
            display_pool = eval_pool + other_cs_pool if (eval_pool or other_cs_pool) else real_test_records[:50]

            utt_options = {
                f"{'✅ [EVALUATED] ' if r.utterance_id in evaluated_ids else ''}{r.utterance_id} [{r.speaker_id}]: \"{r.transcript[:75]}\" ({r.duration_sec}s)": r
                for r in display_pool
            }
            chosen_label = st.selectbox(
                f"Choose a real MUCS 2021 held-out test utterance ({len(eval_pool)} pre-evaluated with real Whisper Small):",
                list(utt_options.keys()),
                index=0,
            )
            chosen_rec = utt_options[chosen_label]
            selected_utt_id = chosen_rec.utterance_id
            selected_audio_path = PROJECT_ROOT / chosen_rec.audio_path
            default_ref_text = chosen_rec.transcript
            default_tags = list(chosen_rec.lang_tags)
            active_duration_sec = float(chosen_rec.duration_sec or 4.0)

        elif "Bundled Hinglish Examples" in input_mode:
            utt_options = {
                f"{r.utterance_id}: \"{r.transcript}\" ({r.accent})": r for r in demo_test_records
            }
            chosen_label = st.selectbox("Choose a bundled Hinglish demo utterance:", list(utt_options.keys()), index=0)
            chosen_rec = utt_options[chosen_label]
            selected_utt_id = chosen_rec.utterance_id
            selected_audio_path = PROJECT_ROOT / chosen_rec.audio_path
            default_ref_text = chosen_rec.transcript
            default_tags = list(chosen_rec.lang_tags)
            active_duration_sec = float(chosen_rec.duration_sec or 3.2)

        elif "Upload Audio File" in input_mode:
            uploaded = st.file_uploader(
                "Upload a Hindi–English audio file (auto-resampled to 16 kHz mono)",
                type=["wav", "mp3", "flac", "ogg"],
            )
            if uploaded is not None:
                tmp_dir = Path(tempfile.gettempdir()) / "switchboard_uploads"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                raw_tmp = tmp_dir / f"raw_{uploaded.name}"
                norm_tmp = tmp_dir / f"resampled_16k_{Path(uploaded.name).stem}.wav"
                raw_tmp.write_bytes(uploaded.getvalue())
                dur, _ = resample_to_16k_mono(raw_tmp, norm_tmp)
                selected_audio_path = norm_tmp
                selected_utt_id = f"upload_{Path(uploaded.name).stem}"
                active_duration_sec = dur
                st.success(f"Uploaded and resampled to 16 kHz mono WAV ({dur:.2f}s).")
            else:
                st.info("Upload an audio file above, or click 'Run Live PyTorch Transcription' once uploaded.")

        else:  # Record via Microphone
            mic_audio = st.audio_input("Click the microphone icon to record a Hindi–English sentence:")
            if mic_audio is not None:
                tmp_dir = Path(tempfile.gettempdir()) / "switchboard_mic"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                raw_mic = tmp_dir / "mic_raw.wav"
                norm_mic = tmp_dir / "mic_16k_mono.wav"
                raw_mic.write_bytes(mic_audio.getvalue())
                dur, _ = resample_to_16k_mono(raw_mic, norm_mic)
                selected_audio_path = norm_mic
                selected_utt_id = "mic_recording"
                active_duration_sec = dur
                st.success(f"Recorded and resampled to 16 kHz mono WAV ({dur:.2f}s).")
            else:
                st.info("Record through your browser microphone above (requires browser microphone permission).")

        if selected_audio_path is not None and selected_audio_path.exists():
            st.audio(str(selected_audio_path), format="audio/wav")

        # Validate selected Fine-Tuned checkpoint availability
        if "Simulate Fine-Tuned Checkpoint Unavailable" in ckpt_choice:
            active_ft_dir = None
            finetuned_available = False
            finetuned_unavail_reason = (
                "Simulated Missing Checkpoint: Fine-tuned checkpoint disabled by user selection. "
                "Baseline Whisper Small (`openai/whisper-small`) remains fully functional."
            )
        elif "whisper_small_lora_plus_lid_cpu" in ckpt_choice:
            active_ft_dir = lora_lid_cpu_dir
            finetuned_available, finetuned_unavail_reason = check_peft_checkpoint_valid(lora_lid_cpu_dir)
            if not finetuned_available and check_peft_checkpoint_valid(lora_lid_dir)[0]:
                active_ft_dir = lora_lid_dir
                finetuned_available, finetuned_unavail_reason = check_peft_checkpoint_valid(lora_lid_dir)
        elif "whisper_small_lora_only_cpu" in ckpt_choice:
            active_ft_dir = lora_only_cpu_dir
            finetuned_available, finetuned_unavail_reason = check_peft_checkpoint_valid(lora_only_cpu_dir)
            if not finetuned_available and check_peft_checkpoint_valid(lora_only_dir)[0]:
                active_ft_dir = lora_only_dir
                finetuned_available, finetuned_unavail_reason = check_peft_checkpoint_valid(lora_only_dir)
        elif "whisper_small_lora_only" in ckpt_choice:
            active_ft_dir = lora_only_dir
            finetuned_available, finetuned_unavail_reason = check_peft_checkpoint_valid(lora_only_dir)
        elif "Full Colab GPU" in ckpt_choice:
            active_ft_dir = colab_full_dir
            finetuned_available, finetuned_unavail_reason = check_peft_checkpoint_valid(colab_full_dir)
        else:
            active_ft_dir = lora_lid_dir
            finetuned_available, finetuned_unavail_reason = check_peft_checkpoint_valid(lora_lid_dir)

        # Resolve transcripts:
        # 1. If user clicked "Run Live PyTorch Transcription" (or uploaded/recorded audio), run live `openai/whisper-small`
        # 2. Else if selected_utt_id is in `saved_real_preds`, load the actual recorded Whisper Small & PEFT outputs
        # 3. Else if in Bundled Demo Fixture mode, show the classroom simulation example with an explicit SIMULATED warning banner
        baseline_hyp = ""
        baseline_time = 0.0
        baseline_source_badge = ""
        finetuned_hyp: Optional[str] = None
        finetuned_time: Optional[float] = None
        finetuned_source_badge = ""

        should_run_live = run_live_inference_btn or (
            selected_audio_path is not None
            and selected_audio_path.exists()
            and ("Upload Audio" in input_mode or "Record via Microphone" in input_mode)
        )

        if should_run_live and selected_audio_path is not None and selected_audio_path.exists():
            with st.spinner("Running real PyTorch transcription with `openai/whisper-small`..."):
                base_runner = get_cached_whisper_baseline_runner("openai/whisper-small")
                out_b = base_runner.transcribe_file(selected_audio_path, utterance_id=selected_utt_id)
                baseline_hyp = out_b.hypothesis
                baseline_time = out_b.transcription_time_sec
                baseline_source_badge = "✅ REAL LIVE PYTORCH INFERENCE (`openai/whisper-small`)"

                if finetuned_available and active_ft_dir is not None:
                    try:
                        ft_runner = get_cached_whisper_peft_runner(str(active_ft_dir))
                        out_f = ft_runner.transcribe_file(selected_audio_path, utterance_id=selected_utt_id)
                        finetuned_hyp = out_f.hypothesis
                        finetuned_time = out_f.transcription_time_sec
                        finetuned_source_badge = f"✅ REAL LIVE PYTORCH INFERENCE (`{active_ft_dir.name}`)"
                    except Exception as exc:
                        finetuned_available = False
                        finetuned_unavail_reason = f"Fine-tuned checkpoint error: {exc}"

        elif selected_utt_id in base_saved_map:
            b_entry = base_saved_map[selected_utt_id]
            baseline_hyp = b_entry["raw_hypothesis"]
            baseline_time = float(b_entry.get("transcription_time_sec") or 1.5)
            baseline_source_badge = "✅ REAL SAVED WHISPER SMALL PREDICTION (`outputs/full_benchmark_real`)"

            active_ft_map = ft_only_saved_map if ("LoRA Only" in ckpt_choice) else ft_lid_saved_map
            if finetuned_available and selected_utt_id in active_ft_map:
                f_entry = active_ft_map[selected_utt_id]
                finetuned_hyp = f_entry["raw_hypothesis"]
                finetuned_time = float(f_entry.get("transcription_time_sec") or 1.6)
                finetuned_source_badge = "✅ REAL SAVED PEFT CHECKPOINT PREDICTION (`outputs/full_benchmark_real`)"
            elif finetuned_available and active_ft_dir is not None and selected_audio_path is not None:
                # Utterance not in pre-saved map for this checkpoint; prompt user to click live inference
                finetuned_hyp = None
                finetuned_source_badge = "Click '⚡ Run Live PyTorch Transcription' above to transcribe this utterance with the selected checkpoint."

        elif "Bundled Hinglish Examples" in input_mode:
            using_simulated_example = True
            sim_base_map = {
                "demo_test_utt_01": ("kal eating ke baad main report submit dunga", 0.42),
                "demo_test_utt_02": ("aaj meeting ke baad main report sabhi kar dunga", 0.44),
                "demo_test_utt_03": ("Rahul umm kal off time par mail bhej देना", 0.41),
                "demo_test_utt_04": ("abhi court branch review aur model train shuru karo", 0.43),
                "demo_test_utt_05": ("हम्म कल सब लोग सुबह जल्दी आएंगे", 0.35),
            }
            sim_ft_map = {
                "demo_test_utt_01": ("kal meeting ke baad main report submit kar dunga", 0.46),
                "demo_test_utt_02": ("आज मीटिंग के बाद मैं रिपोर्ट सबमिट कर दूंगा", 0.47),
                "demo_test_utt_03": ("Rahul umm kal office time par email bhej देना", 0.45),
                "demo_test_utt_04": ("abhi code branch review aur model training shuru karo", 0.47),
                "demo_test_utt_05": ("हम्म कल सब लोग सुबह जल्दी आएंगे", 0.38),
            }
            baseline_hyp, baseline_time = sim_base_map.get(selected_utt_id, (default_ref_text, 0.42))
            baseline_source_badge = "⚠️ SIMULATED CLASSROOM DEMO HYPOTHESIS (Not real ASR output)"
            if finetuned_available:
                finetuned_hyp, finetuned_time = sim_ft_map.get(selected_utt_id, (default_ref_text, 0.45))
                finetuned_source_badge = "⚠️ SIMULATED CLASSROOM DEMO HYPOTHESIS (Not real ASR output)"

        if using_simulated_example:
            st.warning(
                "🚨 **SIMULATED DATA — NOT REAL MODEL RESULTS:** You are viewing the **Bundled Classroom Demo Fixture**, "
                "which uses synthetic harmonic audio and scripted example hypotheses to illustrate switch-boundary edit "
                "attribution (`kal meeting ke baad...`). For actual neural ASR predictions from `openai/whisper-small`, "
                "select **Real MUCS 2021 Held-Out Test Utterances** or click **⚡ Run Live PyTorch Transcription**."
            )

        st.divider()
        st.subheader("Step 2: Side-by-Side ASR Output Comparison")

        col_base, col_ft = st.columns(2)
        with col_base:
            st.markdown("#### 🟦 Primary Baseline: `openai/whisper-small`")
            if baseline_source_badge:
                st.caption(baseline_source_badge)
            if selected_audio_path is not None and (baseline_hyp or should_run_live):
                base_edited = st.text_area(
                    "Baseline Whisper Transcript (editable to inspect custom error scenarios):",
                    value=baseline_hyp,
                    height=85,
                    key=f"base_hyp_{selected_utt_id}",
                )
                norm_base = normalizer.normalize_text(base_edited)
                st.markdown(f"**Normalized Baseline:** `{norm_base}`")
                st.caption(
                    f"Checkpoint: `openai/whisper-small` | Mode: `task='transcribe'` | "
                    f"Time: `{baseline_time:.3f}s` | RTF: `{baseline_time / max(0.1, active_duration_sec):.3f}`"
                )
            else:
                base_edited = ""
                st.info("Click **⚡ Run Live PyTorch Transcription on Active Audio** above to transcribe this utterance with `openai/whisper-small`.")

        with col_ft:
            st.markdown("#### 🟩 Fine-Tuned: `Whisper Small + PEFT LoRA (+ Auxiliary LID Head)`")
            if not finetuned_available:
                st.warning(f"⚠️ **Fine-Tuned Checkpoint Unavailable:** {finetuned_unavail_reason}")
                ft_edited = None
            elif selected_audio_path is not None and finetuned_hyp is not None:
                if finetuned_source_badge:
                    st.caption(finetuned_source_badge)
                ft_edited = st.text_area(
                    "Fine-Tuned Whisper Transcript (editable to inspect custom error scenarios):",
                    value=finetuned_hyp,
                    height=85,
                    key=f"ft_hyp_{selected_utt_id}_{ckpt_choice[:18]}",
                )
                norm_ft = normalizer.normalize_text(ft_edited)
                st.markdown(f"**Normalized Fine-Tuned:** `{norm_ft}`")
                ft_t = finetuned_time or 0.45
                st.caption(
                    f"Checkpoint: `{active_ft_dir.name if active_ft_dir else 'PEFT'}` | "
                    f"Time: `{ft_t:.3f}s` | RTF: `{ft_t / max(0.1, active_duration_sec):.3f}`"
                )
            else:
                ft_edited = None
                st.info(finetuned_source_badge or "Click **⚡ Run Live PyTorch Transcription** above to run the fine-tuned checkpoint.")

        st.divider()
        st.subheader("Step 3: Reference Transcript & Editable Word-Level Language Tags (`HI`, `EN`, `OTHER`)")

        ref_text_input = st.text_input(
            "Reference Transcript (Roman or Devanagari script):",
            value=default_ref_text,
            key=f"ref_input_{selected_utt_id}",
        )
        ref_words = ref_text_input.strip().split()

        if ref_words:
            # Build initial tag list matching reference word count
            initial_tags: List[str] = []
            for idx, w in enumerate(ref_words):
                if ref_text_input.strip() == default_ref_text.strip() and idx < len(default_tags):
                    initial_tags.append(default_tags[idx])
                else:
                    initial_tags.append(infer_word_tag(w, default_corpus_lang="HI"))

            # Standard HTML DOM text input & per-word selectboxes (replaces canvas-only grid so tag editing always works!)
            tags_str_input = st.text_input(
                "Edit Space-Separated Word-Level Language Tags (`HI`, `EN`, `OTHER` — 1 tag per reference word):",
                value=" ".join(initial_tags),
                key=f"tags_str_{selected_utt_id}_{len(ref_words)}",
                help="Modify any tag directly here (e.g., change HI to EN or OTHER) or use the per-word dropdowns below.",
            )
            parsed_str_tags = [
                t.upper() if t.upper() in {"HI", "EN", "OTHER"} else "OTHER"
                for t in tags_str_input.strip().split()
            ]
            while len(parsed_str_tags) < len(ref_words):
                parsed_str_tags.append(initial_tags[len(parsed_str_tags)])
            parsed_str_tags = parsed_str_tags[: len(ref_words)]

            with st.expander("🔧 Click to Edit Individual Word Tags via Dropdown Selectors", expanded=True):
                cols_per_row = min(6, max(1, len(ref_words)))
                reviewed_tags: List[str] = []
                for row_start in range(0, len(ref_words), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for offset in range(cols_per_row):
                        w_idx = row_start + offset
                        if w_idx >= len(ref_words):
                            break
                        w_surface = ref_words[w_idx]
                        curr_t = parsed_str_tags[w_idx]
                        tag_options = ["HI", "EN", "OTHER"]
                        chosen_t = row_cols[offset].selectbox(
                            f"#{w_idx}: {w_surface}",
                            options=tag_options,
                            index=tag_options.index(curr_t) if curr_t in tag_options else 0,
                            key=f"word_tag_sel_{selected_utt_id}_{w_idx}_{curr_t}",
                        )
                        reviewed_tags.append(chosen_t)

            b_preview = detect_switch_boundaries(ref_words, reviewed_tags)
            tag_summary_rows = []
            for idx, (w, t) in enumerate(zip(ref_words, reviewed_tags)):
                if idx in b_preview.overlapping_boundary_indices:
                    role = "⚡⇄ Overlapping 2x-Switch Boundary (counted once)"
                elif idx in b_preview.before_switch_indices and idx in b_preview.after_switch_indices:
                    role = "⚡⇄ Overlapping 2x-Switch Boundary"
                elif idx in b_preview.before_switch_indices:
                    role = "⚡ Before Switch (Pre-Switch)"
                elif idx in b_preview.after_switch_indices:
                    role = "⚡ After Switch (Post-Switch)"
                else:
                    role = "Non-Boundary"
                tag_summary_rows.append(
                    {
                        "Index": idx,
                        "Raw Word": w,
                        "Normalized Word": normalizer.normalize_word(w),
                        "Reviewed Tag": t,
                        "Boundary Role": role,
                    }
                )
            st.dataframe(pd.DataFrame(tag_summary_rows), use_container_width=True, hide_index=True)

            st.markdown("##### ⚡ Switch-Boundary Word Highlighting (`HI → EN` and `EN → HI`)")
            st.markdown(
                render_boundary_highlighted_html(ref_words, reviewed_tags),
                unsafe_allow_html=True,
            )

            temp_record = ManifestRecord(
                utterance_id=selected_utt_id,
                audio_path=str(selected_audio_path or "demo.wav"),
                transcript=" ".join(ref_words),
                lang_tags=reviewed_tags,
                speaker_id="interactive_user",
                accent="Hinglish",
                source="interactive_demo",
                split="test",
                duration_sec=active_duration_sec,
            )

            if base_edited:
                base_eval = evaluate_utterance(
                    temp_record,
                    base_edited,
                    normalizer=normalizer,
                    transcription_time_sec=baseline_time,
                )
                norm_ref_obj = normalizer.normalize_reference(temp_record.transcript, reviewed_tags)
                norm_b_res = detect_switch_boundaries(
                    norm_ref_obj.normalized_words, norm_ref_obj.normalized_lang_tags
                )

                st.markdown("##### 🔬 Word-Level Alignment & Edit Attribution (`MATCH`, `SUB`, `DEL`, `INS`)")
                al_col1, al_col2 = st.columns(2)
                with al_col1:
                    st.markdown("**Baseline Alignment & Attributed Edits:**")
                    st.markdown(
                        render_alignment_ops_html(base_eval.alignment_ops, norm_b_res.boundary_indices),
                        unsafe_allow_html=True,
                    )
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Norm Overall WER", _fmt_metric_card(base_eval.overall_wer))
                    m2.metric("Raw Overall WER", _fmt_metric_card(base_eval.raw_overall_wer))
                    m3.metric("Boundary WER", _fmt_metric_card(base_eval.boundary_wer))
                    m4.metric("Non-Boundary WER", _fmt_metric_card(base_eval.non_boundary_wer))

                with al_col2:
                    if finetuned_available and ft_edited is not None:
                        ft_eval = evaluate_utterance(
                            temp_record,
                            ft_edited,
                            normalizer=normalizer,
                            transcription_time_sec=finetuned_time or 0.45,
                        )
                        st.markdown("**Fine-Tuned Alignment & Attributed Edits:**")
                        st.markdown(
                            render_alignment_ops_html(ft_eval.alignment_ops, norm_b_res.boundary_indices),
                            unsafe_allow_html=True,
                        )
                        f1, f2, f3, f4 = st.columns(4)
                        f1.metric("Norm Overall WER", _fmt_metric_card(ft_eval.overall_wer))
                        f2.metric("Raw Overall WER", _fmt_metric_card(ft_eval.raw_overall_wer))
                        f3.metric("Boundary WER", _fmt_metric_card(ft_eval.boundary_wer))
                        f4.metric("Non-Boundary WER", _fmt_metric_card(ft_eval.non_boundary_wer))
                    else:
                        st.info("Fine-tuned alignment unavailable while fine-tuned checkpoint is offline/disabled.")

                st.caption(
                    f"**Utterance CMI:** `{base_eval.cmi:.2f}` (`{base_eval.cmi_bucket}`) | "
                    f"**Switches Detected:** `{base_eval.num_switches}` ({', '.join(base_eval.switch_directions) or 'None'}) | "
                    f"**Pooled Boundary Words:** `{len(norm_b_res.boundary_indices)}` | "
                    f"**Overlapping Boundary Words (counted once):** `{len(norm_b_res.overlapping_boundary_indices)}`"
                )

    # =========================================================================
    # TAB 2: SAVED BENCHMARK RESULTS & ERROR ANALYSIS
    # =========================================================================
    with tab_bench:
        st.subheader("Saved Benchmark Evaluation Results")
        tier_choice = st.radio(
            "Select Result Tier (Real Research Measurements, Smoke-Test Runs, and Classroom Simulations are strictly isolated):",
            options=[
                "Real Model Benchmark Results (outputs/full_benchmark_real) — [REAL MODEL MEASUREMENTS]",
                "Smoke-Test Verification Results (outputs/smoke_test) — [SMOKE TEST ONLY]",
                "Classroom Demo Fixture (outputs/demo) — [SIMULATED DATA — NOT REAL MODEL RESULTS]",
            ],
            horizontal=False,
        )

        if "outputs/full_benchmark_real" in tier_choice:
            active_eval_dir = PROJECT_ROOT / "outputs" / "full_benchmark_real"
            if (active_eval_dir / "all_models_summary.json").exists():
                st.success(
                    "✅ **REAL MODEL BENCHMARK RESULTS (`outputs/full_benchmark_real/`)** — Every transcript and metric "
                    "below was produced by actual neural model execution on held-out MUCS 2021 Hindi–English test utterances. "
                    "Zero simulated or demo records are permitted in this tier."
                )
            else:
                st.warning(
                    "⏳ **Real Model Benchmark Currently Running or Not Yet Generated** — Run "
                    "`python -m switchboard.cli evaluate-real-benchmark --max-utterances 20` to populate `outputs/full_benchmark_real/`."
                )
        elif "outputs/smoke_test" in tier_choice:
            active_eval_dir = PROJECT_ROOT / "outputs" / "smoke_test"
            st.info(
                "🧪 **SMOKE TEST TIER (`outputs/smoke_test/`)** — Short pipeline verification outputs. "
                "Do not cite smoke-test runs as full research conclusions."
            )
        else:
            active_eval_dir = PROJECT_ROOT / "outputs" / "demo"
            st.error(
                "🚨 **SIMULATED DATA — NOT REAL MODEL RESULTS (`outputs/demo/`)** — This tier displays the 5-utterance "
                "synthetic classroom demo fixture used to verify switch-boundary edit attribution by hand. "
                "Never include these numbers in research tables."
            )

        summary_json_p = active_eval_dir / "all_models_summary.json"
        if summary_json_p.exists():
            payload = json.loads(summary_json_p.read_text(encoding="utf-8"))
            comp_rows = payload.get("comparison_rows", [])
            aggregates = payload.get("model_aggregates", [])

            st.markdown("#### 1. Model Comparison Table (Same Held-Out Test Split)")
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            eval_cfg_p = active_eval_dir / "evaluation_config.json"
            if eval_cfg_p.exists():
                eval_cfg = json.loads(eval_cfg_p.read_text(encoding="utf-8"))
                skipped = eval_cfg.get("skipped_or_blocked_models", [])
                if skipped:
                    with st.expander("ℹ️ View Skipped or Blocked Checkpoints & Dependencies", expanded=False):
                        st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)

            if aggregates:
                model_names = [a["model_name"] for a in aggregates]
                selected_m = st.selectbox("Inspect detailed breakdowns for model:", model_names, index=0)
                m_agg = next(a for a in aggregates if a["model_name"] == selected_m)

                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Normalized WER", _fmt_metric_card(m_agg["overall_wer"]))
                k2.metric("Raw WER (Pre-Norm)", _fmt_metric_card(m_agg["raw_overall_wer"]))
                k3.metric("Boundary WER", _fmt_metric_card(m_agg["boundary_wer"]))
                k4.metric("Non-Boundary WER", _fmt_metric_card(m_agg["non_boundary_wer"]))
                k5.metric(
                    "Mean CMI / RTF",
                    f"CMI {m_agg['mean_cmi']:.1f} | RTF {m_agg['overall_rtf'] or 'N/A'}",
                )

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    st.markdown("##### Switch Direction & Position Breakdown")
                    dir_pos_rows = [
                        {"Subgroup": "Switch Direction: HI → EN", **_fmt_row(m_agg["hi_to_en_wer"])},
                        {"Subgroup": "Switch Direction: EN → HI", **_fmt_row(m_agg["en_to_hi_wer"])},
                        {"Subgroup": "Position: Word Immediately Before Switch", **_fmt_row(m_agg["before_switch_wer"])},
                        {"Subgroup": "Position: Word Immediately After Switch", **_fmt_row(m_agg["after_switch_wer"])},
                    ]
                    st.dataframe(pd.DataFrame(dir_pos_rows), use_container_width=True, hide_index=True)

                with b_col2:
                    st.markdown("##### Error Rates Across Code-Mixing Index (CMI) Ranges")
                    cmi_rows = []
                    for rng_label, stats in m_agg.get("by_cmi_range", {}).items():
                        cmi_rows.append(
                            {
                                "CMI Range": rng_label,
                                "Utterances": stats["utterance_count"],
                                "Overall WER": _fmt_metric_card(stats["overall_wer"]),
                                "Boundary WER": _fmt_metric_card(stats["boundary_wer"]),
                                "Non-Boundary WER": _fmt_metric_card(stats["non_boundary_wer"]),
                            }
                        )
                    st.dataframe(pd.DataFrame(cmi_rows), use_container_width=True, hide_index=True)
                    corr = m_agg.get("cmi_wer_correlation", {})
                    st.caption(
                        f"CMI vs. Utterance WER Pearson Correlation: `r = {corr.get('r')}` "
                        f"(n = {corr.get('n')}, status = `{corr.get('status')}`)"
                    )

                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.markdown("##### Breakdown by Speaker ID")
                    spk_rows = [
                        {
                            "Speaker ID": spk,
                            "Accent": info.get("accent"),
                            "Utterances": info["utterance_count"],
                            "Overall WER": _fmt_metric_card(info["overall_wer"]),
                            "Boundary WER": _fmt_metric_card(info["boundary_wer"]),
                            "Non-Boundary WER": _fmt_metric_card(info["non_boundary_wer"]),
                        }
                        for spk, info in m_agg.get("by_speaker", {}).items()
                    ]
                    st.dataframe(pd.DataFrame(spk_rows), use_container_width=True, hide_index=True)

                with s_col2:
                    st.markdown("##### Breakdown by Accent")
                    acc_rows = [
                        {
                            "Accent": acc,
                            "Utterances": info["utterance_count"],
                            "Overall WER": _fmt_metric_card(info["overall_wer"]),
                            "Boundary WER": _fmt_metric_card(info["boundary_wer"]),
                            "Non-Boundary WER": _fmt_metric_card(info["non_boundary_wer"]),
                        }
                        for acc, info in m_agg.get("by_accent", {}).items()
                    ]
                    st.dataframe(pd.DataFrame(acc_rows), use_container_width=True, hide_index=True)

                safe_name = selected_m.replace("/", "_").replace(" ", "_").lower()
                utt_csv_p = active_eval_dir / f"{safe_name}_per_utterance.csv"
                if utt_csv_p.exists():
                    st.markdown("##### Per-Utterance Results Table")
                    st.dataframe(pd.read_csv(utt_csv_p), use_container_width=True, hide_index=True)
        else:
            st.warning(f"No saved benchmark file found at `{summary_json_p}`.")

    # =========================================================================
    # TAB 3: DATASET, PROVENANCE & ANNOTATION GUIDELINES
    # =========================================================================
    with tab_docs:
        st.subheader("1. Dataset Split Summary & Provenance (`data/full_corpus/unified_manifest.jsonl`)")
        if real_records:
            split_summary = []
            for sp in ["train", "val", "test"]:
                sub = [r for r in real_records if r.split == sp]
                spks = sorted({r.speaker_id for r in sub})
                dur_hrs = sum(float(r.duration_sec or 0.0) for r in sub) / 3600.0
                split_summary.append(
                    {
                        "Split": sp,
                        "Utterances": len(sub),
                        "Unique Speakers (Disjoint)": len(spks),
                        "Duration (Hours)": round(dur_hrs, 3),
                        "Sources": ", ".join(sorted({r.source for r in sub})),
                        "Synthetic Records": sum(1 for r in sub if r.is_synthetic),
                    }
                )
            st.dataframe(pd.DataFrame(split_summary), use_container_width=True, hide_index=True)
            with st.expander("Inspect Sample Unified Manifest Records (Canonical Fields)", expanded=False):
                st.json([r.to_dict() for r in real_records[:3]])

        st.subheader("2. Sample Reviewed Word-Level Annotation Sheet (`docs/sample_annotation_sheet.csv`)")
        sample_csv_p = PROJECT_ROOT / "docs" / "sample_annotation_sheet.csv"
        if sample_csv_p.exists():
            st.dataframe(pd.read_csv(sample_csv_p), use_container_width=True, hide_index=True)

        st.subheader("3. Verified Baseline & Comparison Checkpoints")
        reg_models = [verify_checkpoint_status(m["checkpoint_id"]) for m in list_registered_models()]
        st.dataframe(pd.DataFrame(reg_models), use_container_width=True, hide_index=True)

        doc_col1, doc_col2 = st.columns(2)
        with doc_col1:
            st.markdown("#### Annotation Guidelines (`HI`, `EN`, `OTHER`, Names, Fillers, Ambiguous Words)")
            guide_p = PROJECT_ROOT / "docs" / "ANNOTATION_GUIDELINES.md"
            if guide_p.exists():
                with st.expander("Read Full Annotation Guidelines", expanded=False):
                    st.markdown(guide_p.read_text(encoding="utf-8"))
        with doc_col2:
            st.markdown("#### Reviewed Cross-Script Spelling Equivalents")
            eq_p = PROJECT_ROOT / "configs" / "normalization_equivalents.yaml"
            if eq_p.exists():
                with st.expander("View `configs/normalization_equivalents.yaml`", expanded=False):
                    st.code(eq_p.read_text(encoding="utf-8"), language="yaml")


if __name__ == "__main__":
    main()

"""Unified transcription runners for Whisper Small baseline, PEFT LoRA+LID Whisper, IndicWhisper, and wav2vec 2.0 CTC."""

from __future__ import annotations

import dataclasses
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from switchboard.data.manifest import TARGET_SAMPLE_RATE, ManifestRecord
from switchboard.models.registry import get_checkpoint_metadata, verify_checkpoint_status


@dataclass
class TranscriptionOutput:
    """Single utterance inference result with timing, RTF, and optional token-level LID predictions."""

    utterance_id: str
    hypothesis: str
    audio_duration_sec: float
    transcription_time_sec: float
    rtf: float
    checkpoint_id: str
    decoding_settings: Dict[str, Any]
    predicted_subword_lid: Optional[List[Dict[str, Any]]] = None
    status: str = "ok"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def load_audio_16k(audio_path: str | Path) -> Tuple[np.ndarray, float]:
    """Load an audio file as a 16 kHz mono float32 numpy array and return (waveform, duration_sec)."""
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1)
    if sr != TARGET_SAMPLE_RATE:
        import math
        from scipy.signal import resample_poly

        gcd = math.gcd(int(sr), int(TARGET_SAMPLE_RATE))
        mono = resample_poly(mono, int(TARGET_SAMPLE_RATE) // gcd, int(sr) // gcd).astype(np.float32)
    dur = max(1e-4, float(len(mono)) / float(TARGET_SAMPLE_RATE))
    return mono, round(dur, 4)


class ASRRunner:
    """Base interface for all SwitchBoard speech recognition runners."""

    model_name: str = "base_runner"
    checkpoint_id: str = ""

    def get_decoding_settings(self) -> Dict[str, Any]:
        return {"task": "transcribe"}

    def transcribe_file(self, audio_path: str | Path, utterance_id: str = "utt") -> TranscriptionOutput:
        raise NotImplementedError

    def transcribe_manifest(
        self,
        records: List[ManifestRecord],
        base_dir: Optional[Path] = None,
    ) -> Tuple[Dict[str, str], Dict[str, float], List[TranscriptionOutput]]:
        """Run transcription on a list of held-out manifest records and return (hypotheses, times, outputs)."""
        hypotheses: Dict[str, str] = {}
        times: Dict[str, float] = {}
        outputs: List[TranscriptionOutput] = []

        for rec in records:
            audio_p = Path(rec.audio_path)
            if not audio_p.is_absolute() and base_dir is not None:
                audio_p = base_dir / audio_p
            out = self.transcribe_file(audio_p, utterance_id=rec.utterance_id)
            hypotheses[rec.utterance_id] = out.hypothesis
            times[rec.utterance_id] = out.transcription_time_sec
            outputs.append(out)

        return hypotheses, times, outputs


class WhisperBaselineRunner(ASRRunner):
    """Primary baseline runner for `openai/whisper-small` (or any pretrained Whisper checkpoint) in transcription mode."""

    def __init__(
        self,
        checkpoint_id: str = "openai/whisper-small",
        device: Optional[str] = None,
        language: Optional[str] = None,
        num_beams: int = 1,
        max_new_tokens: int = 128,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.model_name = checkpoint_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language  # None allows Whisper multilingual identification or specify 'hi'/'en'
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        self.processor = WhisperProcessor.from_pretrained(checkpoint_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(checkpoint_id)
        self.model_revision = getattr(self.model.config, "_commit_hash", None) or "local_or_default"
        self.model.to(self.device)
        self.model.eval()

    def get_decoding_settings(self) -> Dict[str, Any]:
        return {
            "task": "transcribe",
            "language": self.language or "auto_multilingual",
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "device": self.device,
            "model_revision": self.model_revision,
        }

    @torch.no_grad()
    def transcribe_file(self, audio_path: str | Path, utterance_id: str = "utt") -> TranscriptionOutput:
        waveform, dur = load_audio_16k(audio_path)
        t0 = time.perf_counter()

        inputs = self.processor(
            waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.device)

        gen_kwargs: Dict[str, Any] = {
            "task": "transcribe",
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "no_repeat_ngram_size": 3,
        }
        if "attention_mask" in inputs and inputs.attention_mask is not None:
            gen_kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
        if self.language:
            gen_kwargs["language"] = self.language

        try:
            generated_ids = self.model.generate(input_features, **gen_kwargs)
        except Exception:
            # Fallback for older/custom generation configs without forced_decoder_ids
            generated_ids = self.model.generate(
                input_features,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                no_repeat_ngram_size=3,
            )

        raw_decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = re.sub(r"<\|[^|>]+\|>", " ", raw_decoded).strip()
        text = " ".join(text.split())
        elapsed = max(1e-6, time.perf_counter() - t0)
        rtf = round(elapsed / dur, 6)

        return TranscriptionOutput(
            utterance_id=utterance_id,
            hypothesis=text,
            audio_duration_sec=dur,
            transcription_time_sec=round(elapsed, 6),
            rtf=rtf,
            checkpoint_id=self.checkpoint_id,
            decoding_settings=self.get_decoding_settings(),
        )


class IndicWhisperRunner(WhisperBaselineRunner):
    """Comparison runner for AI4Bharat IndicWhisper checkpoints (defaults to Hindi transcription mode)."""

    def __init__(
        self,
        checkpoint_id: str = "vasista22/whisper-hindi-small",
        device: Optional[str] = None,
        num_beams: int = 1,
        max_new_tokens: int = 128,
    ) -> None:
        super().__init__(
            checkpoint_id=checkpoint_id,
            device=device,
            language="hi",
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        self.model_name = f"indicwhisper ({checkpoint_id})"


class WhisperPeftRunner(ASRRunner):
    """Runner for fine-tuned Whisper Small with PEFT LoRA adapter and optional Auxiliary LID Head."""

    def __init__(
        self,
        adapter_dir: str | Path,
        base_checkpoint_id: Optional[str] = None,
        device: Optional[str] = None,
        language: Optional[str] = None,
        num_beams: int = 1,
        max_new_tokens: int = 128,
    ) -> None:
        from switchboard.training.whisper_lora_lid import ID_TO_LID_TAG, load_finetuned_whisper_lora_lid

        self.adapter_dir = Path(adapter_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self._id_to_tag = ID_TO_LID_TAG

        loaded = load_finetuned_whisper_lora_lid(
            checkpoint_dir=self.adapter_dir,
            base_model_override=base_checkpoint_id,
            device=self.device,
        )
        metrics_p = self.adapter_dir / "training_metrics.json"
        if metrics_p.exists():
            m_data = json.loads(metrics_p.read_text(encoding="utf-8"))
            if m_data.get("used_tiny_fallback"):
                raise RuntimeError(
                    f"Checkpoint at '{self.adapter_dir}' was created with a randomly initialized tiny fallback architecture "
                    f"(used_tiny_fallback=True) and is NOT a functional speech recognition model. "
                    f"Please train on real 'openai/whisper-small' with allow_tiny_architecture_for_cpu_smoke=false."
                )

        base_d_model = int(getattr(loaded["peft_model"].base_model.model.config, "d_model", 0))
        if base_d_model < 384:
            raise RuntimeError(
                f"Checkpoint at '{self.adapter_dir}' has d_model={base_d_model} (< 384) indicating a randomly initialized "
                f"unit-test fallback model rather than a pretrained Whisper checkpoint."
            )

        self.peft_model = loaded["peft_model"]
        self.lid_head = loaded["lid_head"]
        self.processor = loaded["processor"]
        self.training_config = loaded["training_config"]
        self.checkpoint_id = str(self.adapter_dir)
        exp_name = self.training_config.get("experiment_name", self.adapter_dir.name)
        self.model_name = f"whisper_peft ({exp_name})"

    def get_decoding_settings(self) -> Dict[str, Any]:
        return {
            "task": "transcribe",
            "language": self.language or "auto_multilingual",
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "device": self.device,
            "auxiliary_lid_enabled": self.lid_head is not None,
        }

    @torch.no_grad()
    def transcribe_file(self, audio_path: str | Path, utterance_id: str = "utt") -> TranscriptionOutput:
        waveform, dur = load_audio_16k(audio_path)
        t0 = time.perf_counter()

        inputs = self.processor(
            waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.device)

        gen_kwargs: Dict[str, Any] = {
            "task": "transcribe",
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "no_repeat_ngram_size": 3,
        }
        if "attention_mask" in inputs and inputs.attention_mask is not None:
            gen_kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
        if self.language:
            gen_kwargs["language"] = self.language

        try:
            generated_ids = self.peft_model.generate(input_features=input_features, **gen_kwargs)
        except Exception:
            generated_ids = self.peft_model.generate(
                input_features=input_features,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                no_repeat_ngram_size=3,
            )

        raw_decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = re.sub(r"<\|[^|>]+\|>", " ", raw_decoded).strip()
        text = " ".join(text.split())

        subword_lid_preds: Optional[List[Dict[str, Any]]] = None
        if self.lid_head is not None and generated_ids.shape[1] > 1:
            outputs = self.peft_model(
                input_features=input_features,
                decoder_input_ids=generated_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            last_dec_hidden = outputs.decoder_hidden_states[-1]
            lid_logits = self.lid_head(last_dec_hidden)
            pred_ids = torch.argmax(lid_logits, dim=-1)[0].cpu().tolist()
            tok_ids = generated_ids[0].cpu().tolist()
            special_ids = set(getattr(self.processor.tokenizer, "all_special_ids", []))
            subword_lid_preds = []
            for tid, lid_idx in zip(tok_ids, pred_ids):
                if tid in special_ids:
                    continue
                sub_str = self.processor.tokenizer.decode([tid])
                subword_lid_preds.append(
                    {"token_id": tid, "subword": sub_str, "predicted_tag": self._id_to_tag.get(lid_idx, "OTHER")}
                )

        elapsed = max(1e-6, time.perf_counter() - t0)
        rtf = round(elapsed / dur, 6)

        return TranscriptionOutput(
            utterance_id=utterance_id,
            hypothesis=text,
            audio_duration_sec=dur,
            transcription_time_sec=round(elapsed, 6),
            rtf=rtf,
            checkpoint_id=self.checkpoint_id,
            decoding_settings=self.get_decoding_settings(),
            predicted_subword_lid=subword_lid_preds,
        )


class Wav2Vec2CTCRunner(ASRRunner):
    """Comparison runner for wav2vec 2.0 / XLS-R CTC checkpoints with explicit CTC head verification."""

    def __init__(
        self,
        checkpoint_id: str = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.model_name = f"wav2vec2_ctc ({checkpoint_id})"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        meta = get_checkpoint_metadata(checkpoint_id)
        if meta.requires_ctc_head_training:
            raise RuntimeError(
                f"CTC Head Dependency Error for '{checkpoint_id}': {meta.ctc_dependency_note}"
            )

        config = AutoConfig.from_pretrained(checkpoint_id)
        archs = getattr(config, "architectures", None) or []
        if not any("ForCTC" in a for a in archs):
            raise RuntimeError(
                f"Checkpoint '{checkpoint_id}' (architectures={archs}) does not contain a trained CTC head "
                f"(Wav2Vec2ForCTC). Please fine-tune a CTC head first or provide a fine-tuned CTC checkpoint."
            )

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(checkpoint_id)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(checkpoint_id)
        self.model = AutoModelForCTC.from_pretrained(checkpoint_id)
        self.model_revision = getattr(self.model.config, "_commit_hash", None) or "local_or_default"
        self.model.to(self.device)
        self.model.eval()

    def get_decoding_settings(self) -> Dict[str, Any]:
        return {
            "task": "transcribe",
            "decoding": "ctc_greedy",
            "device": self.device,
            "model_revision": self.model_revision,
        }

    @torch.no_grad()
    def transcribe_file(self, audio_path: str | Path, utterance_id: str = "utt") -> TranscriptionOutput:
        waveform, dur = load_audio_16k(audio_path)
        t0 = time.perf_counter()

        inputs = self.processor(
            waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(pred_ids)[0].strip()

        elapsed = max(1e-6, time.perf_counter() - t0)
        rtf = round(elapsed / dur, 6)

        return TranscriptionOutput(
            utterance_id=utterance_id,
            hypothesis=text,
            audio_duration_sec=dur,
            transcription_time_sec=round(elapsed, 6),
            rtf=rtf,
            checkpoint_id=self.checkpoint_id,
            decoding_settings=self.get_decoding_settings(),
        )

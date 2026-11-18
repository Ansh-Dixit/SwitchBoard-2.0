"""CPU-optimized PEFT LoRA fine-tuning for Whisper Small decoder attention with an Auxiliary Language Identification (LID) head."""

from __future__ import annotations

import gc
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from switchboard.data.manifest import TARGET_SAMPLE_RATE, ManifestRecord, load_manifest
from switchboard.evaluation.boundary import compute_cmi, evaluate_corpus
from switchboard.models.runners import load_audio_16k
from switchboard.normalization.normalizer import TranscriptNormalizer

LID_TAG_TO_ID: Dict[str, int] = {"HI": 0, "EN": 1, "OTHER": 2}
ID_TO_LID_TAG: Dict[int, str] = {0: "HI", 1: "EN", 2: "OTHER"}
IGNORE_INDEX: int = -100


def detect_cpu_hardware_info() -> Dict[str, Any]:
    """Detect exact CPU model (including Windows registry ProcessorNameString), core/thread counts, and system RAM."""
    cpu_model = platform.processor() or "Unknown CPU"
    if platform.system().lower() == "windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if val and str(val).strip():
                cpu_model = str(val).strip()
        except Exception:
            pass

    phys_cores = psutil.cpu_count(logical=False) or 4
    log_threads = psutil.cpu_count(logical=True) or phys_cores
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    rss_mb = round(proc.memory_info().rss / (1024.0 * 1024.0), 2)

    return {
        "cpu_model": cpu_model,
        "physical_cores": int(phys_cores),
        "logical_threads": int(log_threads),
        "total_ram_gb": round(vm.total / (1024.0**3), 2),
        "available_ram_gb": round(vm.available / (1024.0**3), 2),
        "ram_percent_used": float(vm.percent),
        "process_rss_mb": rss_mb,
        "pytorch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "os_platform": platform.platform(),
    }


def get_process_rss_mb() -> float:
    """Return current process resident set size (RSS) in megabytes."""
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0), 2)


class AuxiliaryLIDHead(nn.Module):
    """Token-level Language Identification classifier attached to Whisper decoder hidden states."""

    def __init__(self, d_model: int, num_labels: int = 3, hidden_dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.hidden_dropout = hidden_dropout
        mid_dim = max(32, d_model // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(hidden_dropout),
            nn.Linear(d_model, mid_dim),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(mid_dim, num_labels),
        )

    def forward(self, decoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """Map decoder hidden states (batch, seq_len, d_model) -> LID logits (batch, seq_len, num_labels)."""
        return self.net(decoder_hidden_states)


def find_decoder_attention_target_modules(
    model: nn.Module,
    projection_names: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
    decoder_attention_only: bool = True,
) -> List[str]:
    """Identify exact decoder self-attention and cross-attention linear module names for PEFT LoRA injection."""
    proj_tuple = tuple(projection_names)
    targets: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.endswith(proj_tuple):
            continue
        if decoder_attention_only:
            if "decoder.layers." in name and (".self_attn." in name or ".encoder_attn." in name):
                targets.append(name)
        else:
            targets.append(name)
    if not targets:
        raise ValueError(
            f"No matching decoder attention modules found for projections={projection_names}."
        )
    return targets


def align_word_tags_to_subwords(
    words: List[str],
    lang_tags: List[str],
    tokenizer: Any,
    prefix_token_ids: Optional[List[int]] = None,
    eos_token_id: Optional[int] = None,
    max_length: int = 224,
) -> Tuple[List[int], List[int]]:
    """Map word-level language tags (HI, EN, OTHER) to Whisper decoder subword positions."""
    if len(words) != len(lang_tags):
        raise ValueError(
            f"Word count ({len(words)}) does not match lang_tags count ({len(lang_tags)})."
        )

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        special_ids.add(int(eos_token_id))

    if prefix_token_ids is None:
        bos_id = getattr(tokenizer, "bos_token_id", None)
        prefix_token_ids = [int(bos_id)] if bos_id is not None else []

    token_ids: List[int] = []
    lid_labels: List[int] = []

    for pid in prefix_token_ids:
        token_ids.append(int(pid))
        lid_labels.append(IGNORE_INDEX)

    for w_idx, (word, tag) in enumerate(zip(words, lang_tags)):
        surface = word if w_idx == 0 else f" {word}"
        sub_ids = tokenizer.encode(surface, add_special_tokens=False)
        tag_id = LID_TAG_TO_ID.get(tag.upper(), IGNORE_INDEX)
        for sid in sub_ids:
            token_ids.append(int(sid))
            if int(sid) in special_ids:
                lid_labels.append(IGNORE_INDEX)
            else:
                lid_labels.append(tag_id)

    max_content = max(2, max_length - 1)
    if len(token_ids) > max_content:
        token_ids = token_ids[:max_content]
        lid_labels = lid_labels[:max_content]

    if eos_token_id is not None:
        token_ids.append(int(eos_token_id))
        lid_labels.append(IGNORE_INDEX)

    return token_ids, lid_labels


class LazyWhisperLIDDataset(Dataset):
    """Memory-efficient lazy Dataset that reads one 16 kHz mono WAV file at a time on demand."""

    def __init__(
        self,
        records: List[ManifestRecord],
        feature_extractor: Any,
        tokenizer: Any,
        base_dir: Optional[Path] = None,
        prefix_token_ids: Optional[List[int]] = None,
        eos_token_id: int = 50257,
    ) -> None:
        self.records = list(records)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.base_dir = base_dir
        self.prefix_token_ids = prefix_token_ids or [50258, 50276, 50359, 50363]
        self.eos_token_id = eos_token_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        audio_p = Path(rec.audio_path)
        if not audio_p.is_absolute() and self.base_dir is not None:
            audio_p = self.base_dir / audio_p
        waveform, _ = load_audio_16k(audio_p)
        mel = self.feature_extractor(
            waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        ).input_features[0]
        t_ids, l_ids = align_word_tags_to_subwords(
            words=rec.words,
            lang_tags=rec.lang_tags,
            tokenizer=self.tokenizer,
            prefix_token_ids=self.prefix_token_ids,
            eos_token_id=self.eos_token_id,
        )
        return {
            "utterance_id": rec.utterance_id,
            "input_features": mel,
            "token_ids": t_ids,
            "lid_labels": l_ids,
        }


def collate_whisper_lid_batch(
    batch_items: List[Dict[str, Any]],
    pad_token_id: int = 50257,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of (input_features, labels, lid_labels) without unnecessary tensor copies."""
    if len(batch_items) == 1:
        item = batch_items[0]
        return {
            "input_features": item["input_features"].unsqueeze(0),
            "labels": torch.tensor([item["token_ids"]], dtype=torch.long),
            "lid_labels": torch.tensor([item["lid_labels"]], dtype=torch.long),
        }

    input_features = torch.stack([item["input_features"] for item in batch_items], dim=0)
    max_seq = max(len(item["token_ids"]) for item in batch_items)

    batch_labels: List[List[int]] = []
    batch_lid_labels: List[List[int]] = []

    for item in batch_items:
        t_ids = list(item["token_ids"])
        l_ids = list(item["lid_labels"])
        pad_len = max_seq - len(t_ids)
        batch_labels.append(t_ids + [IGNORE_INDEX] * pad_len)
        batch_lid_labels.append(l_ids + [IGNORE_INDEX] * pad_len)

    return {
        "input_features": input_features,
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "lid_labels": torch.tensor(batch_lid_labels, dtype=torch.long),
    }


def compute_joint_asr_lid_loss(
    peft_model: nn.Module,
    lid_head: Optional[AuxiliaryLIDHead],
    input_features: torch.Tensor,
    labels: torch.Tensor,
    lid_labels: torch.Tensor,
    lid_loss_weight: float = 0.3,
    use_no_grad_frozen_encoder: bool = True,
) -> Dict[str, Any]:
    """Run forward pass through Whisper + LoRA and compute combined ASR + weighted Auxiliary LID loss.

    CPU Optimization:
    - Because all Whisper encoder layers are frozen (`requires_grad=False`) and LoRA adapters are placed
      only in decoder attention layers, running `encoder(input_features)` inside `torch.no_grad()` avoids
      storing 12-layer encoder autograd activations for 1,500 frames, cutting RAM usage and CPU step time
      while producing identical decoder gradients.
    """
    need_hidden = (lid_head is not None) and (lid_loss_weight > 0.0)
    encoder_outputs = None

    if use_no_grad_frozen_encoder:
        try:
            inner_whisper = peft_model.base_model.model
            enc = inner_whisper.model.encoder
            if all(not p.requires_grad for p in enc.parameters()):
                with torch.no_grad():
                    encoder_outputs = enc(input_features, return_dict=True)
        except Exception:
            encoder_outputs = None

    if encoder_outputs is not None:
        outputs = peft_model(
            encoder_outputs=encoder_outputs,
            labels=labels,
            output_hidden_states=need_hidden,
            return_dict=True,
        )
    else:
        outputs = peft_model(
            input_features=input_features,
            labels=labels,
            output_hidden_states=need_hidden,
            return_dict=True,
        )

    asr_loss: torch.Tensor = outputs.loss

    lid_logits: Optional[torch.Tensor] = None
    if need_hidden and lid_head is not None:
        last_decoder_hidden = outputs.decoder_hidden_states[-1]
        lid_logits = lid_head(last_decoder_hidden)
        valid_mask = lid_labels.ne(IGNORE_INDEX)
        if valid_mask.any():
            lid_loss = F.cross_entropy(
                lid_logits.reshape(-1, lid_logits.size(-1)),
                lid_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        else:
            lid_loss = lid_logits.sum() * 0.0
        total_loss = asr_loss + float(lid_loss_weight) * lid_loss
    else:
        lid_loss = asr_loss.detach() * 0.0
        total_loss = asr_loss

    return {
        "total_loss": total_loss,
        "asr_loss": asr_loss,
        "lid_loss": lid_loss,
        "lid_logits": lid_logits,
        "asr_logits": outputs.logits,
    }


def filter_records_for_curriculum_stage(
    records: List[ManifestRecord],
    filter_mode: str,
) -> List[ManifestRecord]:
    """Filter training records according to the curriculum stage (strictly restricted to split == 'train')."""
    train_recs = [r for r in records if r.split == "train"]
    mode = filter_mode.strip().lower()

    if mode == "monolingual":
        filtered = [
            r
            for r in train_recs
            if (not r.is_synthetic)
            and ("synthetic" not in r.source.lower())
            and compute_cmi(r.lang_tags) <= 1e-9
        ]
        return filtered if filtered else train_recs

    if mode == "real_mixed":
        filtered = [
            r
            for r in train_recs
            if (not r.is_synthetic)
            and ("synthetic" not in r.source.lower())
            and compute_cmi(r.lang_tags) > 1e-9
        ]
        return filtered if filtered else [r for r in train_recs if not r.is_synthetic]

    if mode in {"all_with_synthetic", "synthetic_supplement", "synthetic_mixed"}:
        filtered = [
            r
            for r in train_recs
            if r.is_synthetic or ("synthetic" in r.source.lower()) or compute_cmi(r.lang_tags) > 1e-9
        ]
        return filtered if filtered else train_recs

    return train_recs


def snapshot_parameter_fingerprints(
    peft_model: nn.Module,
    lid_head: Optional[AuxiliaryLIDHead],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Capture lightweight per-parameter clone tensors for trainable LoRA/LID weights and summary norms for frozen base weights."""
    frozen_norms: Dict[str, torch.Tensor] = {}
    lora_clones: Dict[str, torch.Tensor] = {}
    lid_clones: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if p.requires_grad:
                lora_clones[name] = p.detach().cpu().clone()
            else:
                # Store compact 2-element tensor [l2_norm, mean] to avoid duplicating 967 MB of frozen weights in RAM
                frozen_norms[name] = torch.tensor(
                    [float(p.detach().norm().item()), float(p.detach().mean().item())],
                    dtype=torch.float64,
                )
        if lid_head is not None:
            for name, p in lid_head.named_parameters():
                if p.requires_grad:
                    lid_clones[name] = p.detach().cpu().clone()

    return {
        "frozen_norms": frozen_norms,
        "lora_clones": lora_clones,
        "lid_clones": lid_clones,
    }


def verify_parameter_updates(
    peft_model: nn.Module,
    lid_head: Optional[AuxiliaryLIDHead],
    before_snap: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """Verify that frozen base Whisper parameters did NOT change and trainable LoRA (and LID head) parameters DID change."""
    max_frozen_diff = 0.0
    max_lora_diff = 0.0
    max_lid_diff = 0.0
    updated_lora_tensors = 0
    updated_lid_tensors = 0

    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if p.requires_grad:
                prev = before_snap["lora_clones"][name]
                diff = float((p.detach().cpu() - prev).abs().max().item())
                if diff > 1e-12:
                    updated_lora_tensors += 1
                if diff > max_lora_diff:
                    max_lora_diff = diff
            else:
                prev_vec = before_snap["frozen_norms"][name]
                curr_vec = torch.tensor(
                    [float(p.detach().norm().item()), float(p.detach().mean().item())],
                    dtype=torch.float64,
                )
                diff = float((curr_vec - prev_vec).abs().max().item())
                if diff > max_frozen_diff:
                    max_frozen_diff = diff

        if lid_head is not None:
            for name, p in lid_head.named_parameters():
                if p.requires_grad:
                    prev = before_snap["lid_clones"][name]
                    diff = float((p.detach().cpu() - prev).abs().max().item())
                    if diff > 1e-12:
                        updated_lid_tensors += 1
                    if diff > max_lid_diff:
                        max_lid_diff = diff

    return {
        "frozen_base_params_unchanged": bool(max_frozen_diff <= 1e-9),
        "frozen_base_max_norm_diff": round(max_frozen_diff, 10),
        "trainable_lora_params_updated": bool(max_lora_diff > 1e-9),
        "trainable_lora_max_abs_diff": round(max_lora_diff, 8),
        "updated_lora_tensor_count": updated_lora_tensors,
        "total_lora_tensor_count": len(before_snap["lora_clones"]),
        "trainable_lid_head_updated": bool(max_lid_diff > 1e-9) if lid_head is not None else None,
        "trainable_lid_head_max_abs_diff": round(max_lid_diff, 8) if lid_head is not None else 0.0,
        "updated_lid_tensor_count": updated_lid_tensors if lid_head is not None else 0,
    }


def build_whisper_lora_lid_model(
    base_model_name_or_path: str = "openai/whisper-small",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    projection_names: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
    decoder_attention_only: bool = True,
    use_lid_head: bool = True,
    lid_num_labels: int = 3,
    lid_hidden_dropout: float = 0.1,
    local_tiny_fallback_if_offline: bool = False,
    device: Optional[str] = None,
    gradient_checkpointing: bool = False,
) -> Dict[str, Any]:
    """Load Whisper base model, freeze all pretrained weights, inject PEFT LoRA into decoder attention layers,
    and optionally initialize the Auxiliary Language Identification head.
    """
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor: Optional[WhisperProcessor] = None
    used_tiny_fallback = False

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir_name = "models--" + str(base_model_name_or_path).replace("/", "--")
    is_cached = (cache_root / cache_dir_name).exists() or Path(base_model_name_or_path).exists()

    try:
        if local_tiny_fallback_if_offline and not is_cached:
            raise FileNotFoundError(
                f"Base checkpoint '{base_model_name_or_path}' not in local HF cache; using compact Whisper smoke architecture."
            )
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name_or_path)
        processor = WhisperProcessor.from_pretrained(base_model_name_or_path)
    except Exception:
        if not local_tiny_fallback_if_offline:
            raise
        used_tiny_fallback = True
        tiny_cfg = WhisperConfig(
            vocab_size=51865,
            num_mel_bins=80,
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=2,
            decoder_layers=2,
            decoder_attention_heads=2,
            decoder_ffn_dim=128,
            encoder_ffn_dim=128,
            bos_token_id=50257,
            eos_token_id=50257,
            pad_token_id=50257,
            decoder_start_token_id=50258,
        )
        base_model = WhisperForConditionalGeneration(tiny_cfg)

    for param in base_model.parameters():
        param.requires_grad = False

    if gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    target_modules = find_decoder_attention_target_modules(
        base_model,
        projection_names=projection_names,
        decoder_attention_only=decoder_attention_only,
    )

    lora_config = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.to(target_device)

    lid_head: Optional[AuxiliaryLIDHead] = None
    if use_lid_head:
        d_model = int(base_model.config.d_model)
        lid_head = AuxiliaryLIDHead(
            d_model=d_model,
            num_labels=int(lid_num_labels),
            hidden_dropout=float(lid_hidden_dropout),
        ).to(target_device)

    trainable_lora_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_whisper_params = sum(p.numel() for p in peft_model.parameters())
    lid_params = sum(p.numel() for p in lid_head.parameters() if p.requires_grad) if lid_head is not None else 0

    return {
        "peft_model": peft_model,
        "lid_head": lid_head,
        "processor": processor,
        "target_modules": target_modules,
        "device": target_device,
        "used_tiny_fallback": used_tiny_fallback,
        "param_summary": {
            "total_whisper_params": total_whisper_params,
            "trainable_lora_params": trainable_lora_params,
            "trainable_lid_head_params": lid_params,
            "trainable_ratio_pct": round(
                100.0 * (trainable_lora_params + lid_params) / max(1, total_whisper_params + lid_params), 4
            ),
        },
    }


def save_finetuned_whisper_lora_lid(
    output_dir: str | Path,
    peft_model: nn.Module,
    lid_head: Optional[AuxiliaryLIDHead],
    processor: Optional[Any],
    training_config: Dict[str, Any],
    training_metrics: Optional[Dict[str, Any]] = None,
    resume_state: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save PEFT LoRA adapter, Auxiliary LID head weights, base model config, training configuration, and resume state."""
    out_root = Path(output_dir)
    adapter_dir = out_root / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    peft_model.save_pretrained(str(adapter_dir))

    base_cfg = peft_model.base_model.model.config
    base_cfg.save_pretrained(str(out_root / "base_config"))

    if processor is not None:
        try:
            processor.save_pretrained(str(out_root / "processor"))
        except Exception:
            pass

    lid_path = out_root / "lid_head.pt"
    if lid_head is not None:
        torch.save(
            {
                "state_dict": lid_head.state_dict(),
                "d_model": lid_head.d_model,
                "num_labels": lid_head.num_labels,
                "hidden_dropout": lid_head.hidden_dropout,
            },
            str(lid_path),
        )

    cfg_path = out_root / "training_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)

    if training_metrics is not None:
        metrics_path = out_root / "training_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(training_metrics, f, indent=2, ensure_ascii=False)

    if resume_state is not None:
        torch.save(resume_state, str(out_root / "resume_state.pt"))

    return out_root


def load_finetuned_whisper_lora_lid(
    checkpoint_dir: str | Path,
    base_model_override: Optional[str] = None,
    base_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Reload a saved PEFT LoRA adapter, Auxiliary LID head, and training configuration."""
    ckpt_root = Path(checkpoint_dir)
    adapter_dir = ckpt_root / "lora_adapter"
    cfg_path = ckpt_root / "training_config.json"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory not found at {adapter_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training configuration not found at {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        training_config = json.load(f)

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_name = base_model_override or training_config.get("base_model_name_or_path", "openai/whisper-small")
    processor: Optional[Any] = None

    proc_dir = ckpt_root / "processor"
    if proc_dir.exists():
        try:
            processor = WhisperProcessor.from_pretrained(str(proc_dir))
        except Exception:
            processor = None

    try:
        if base_state_dict is not None:
            base_cfg = WhisperConfig.from_pretrained(str(ckpt_root / "base_config"))
            base_model = WhisperForConditionalGeneration(base_cfg)
            base_model.load_state_dict(base_state_dict)
        else:
            base_model = WhisperForConditionalGeneration.from_pretrained(base_name)
            if processor is None:
                processor = WhisperProcessor.from_pretrained(base_name)
    except Exception:
        base_cfg = WhisperConfig.from_pretrained(str(ckpt_root / "base_config"))
        base_model = WhisperForConditionalGeneration(base_cfg)

    for p in base_model.parameters():
        p.requires_grad = False

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.to(target_device)
    peft_model.eval()

    lid_head: Optional[AuxiliaryLIDHead] = None
    lid_path = ckpt_root / "lid_head.pt"
    if lid_path.exists():
        lid_payload = torch.load(str(lid_path), map_location=target_device, weights_only=True)
        lid_head = AuxiliaryLIDHead(
            d_model=int(lid_payload["d_model"]),
            num_labels=int(lid_payload["num_labels"]),
            hidden_dropout=float(lid_payload.get("hidden_dropout", 0.1)),
        ).to(target_device)
        lid_head.load_state_dict(lid_payload["state_dict"])
        lid_head.eval()

    return {
        "peft_model": peft_model,
        "lid_head": lid_head,
        "processor": processor,
        "training_config": training_config,
        "device": target_device,
    }


@torch.no_grad()
def verify_checkpoint_reload(
    checkpoint_dir: str | Path,
    original_peft_model: nn.Module,
    original_lid_head: Optional[AuxiliaryLIDHead],
    sample_input_features: Optional[torch.Tensor] = None,
    sample_decoder_input_ids: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Verify that a saved LoRA adapter and LID head reload cleanly and match pre-save forward-pass outputs.

    Memory Optimization for 16 GB RAM CPU Machines:
    - Only extracts an in-memory `base_sd` clone when `d_model < 384` (random tiny unit-test model).
      When verifying a real pretrained Whisper model (`openai/whisper-small`), computes `out_orig` first,
      then reloads from local cache without duplicating 967 MB of `base_sd` in a Python dict.
    """
    original_peft_model.eval()
    if original_lid_head is not None:
        original_lid_head.eval()

    device = next(original_peft_model.parameters()).device
    base_cfg = original_peft_model.base_model.model.config
    d_model = int(getattr(base_cfg, "d_model", 768))

    if sample_input_features is None:
        num_mel = int(getattr(base_cfg, "num_mel_bins", 80))
        torch.manual_seed(1234)
        sample_input_features = torch.randn(1, num_mel, 3000, device=device)
    if sample_decoder_input_ids is None:
        bos = int(getattr(base_cfg, "decoder_start_token_id", 50258))
        sample_decoder_input_ids = torch.tensor([[bos, 100, 200, 300]], dtype=torch.long, device=device)

    out_orig = original_peft_model(
        input_features=sample_input_features,
        decoder_input_ids=sample_decoder_input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    orig_asr_logits = out_orig.logits.detach().cpu().clone()
    orig_lid_logits = (
        original_lid_head(out_orig.decoder_hidden_states[-1]).detach().cpu().clone()
        if original_lid_head is not None
        else None
    )
    del out_orig

    base_sd: Optional[Dict[str, torch.Tensor]] = None
    if d_model < 384:
        base_sd = {}
        for k, v in original_peft_model.base_model.model.state_dict().items():
            if "lora_" not in k:
                clean_k = k.replace(".base_layer", "")
                base_sd[clean_k] = v.detach().cpu().clone()

    reloaded = load_finetuned_whisper_lora_lid(
        checkpoint_dir=checkpoint_dir,
        base_state_dict=base_sd,
        device=str(device),
    )
    rel_peft = reloaded["peft_model"]
    rel_lid = reloaded["lid_head"]

    out_rel = rel_peft(
        input_features=sample_input_features,
        decoder_input_ids=sample_decoder_input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    rel_asr_logits = out_rel.logits.detach().cpu()

    asr_max_diff = float((orig_asr_logits - rel_asr_logits).abs().max().item())
    asr_matches = bool(torch.allclose(orig_asr_logits, rel_asr_logits, atol=1e-5))

    lid_matches = True
    lid_max_diff = 0.0
    if orig_lid_logits is not None:
        if rel_lid is None:
            lid_matches = False
        else:
            rel_lid_logits = rel_lid(out_rel.decoder_hidden_states[-1]).detach().cpu()
            lid_max_diff = float((orig_lid_logits - rel_lid_logits).abs().max().item())
            lid_matches = bool(torch.allclose(orig_lid_logits, rel_lid_logits, atol=1e-5))

    del out_rel, rel_peft, rel_lid, reloaded
    gc.collect()

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "asr_logits_match": asr_matches,
        "asr_max_abs_diff": asr_max_diff,
        "lid_logits_match": lid_matches,
        "lid_max_abs_diff": lid_max_diff,
        "reload_verified": asr_matches and lid_matches,
    }


class SimpleWordSubwordTokenizer:
    """Fallback subword tokenizer used only when offline unit testing without a cached WhisperTokenizer."""

    all_special_ids = [50257, 50258, 50259, 50276, 50359, 50363]
    bos_token_id = 50258
    eos_token_id = 50257
    pad_token_id = 50257

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        stripped = text.strip()
        if not stripped:
            return []
        b = stripped.encode("utf-8")
        if len(b) == 1:
            return [100 + (b[0] % 4000)]
        mid = len(b) // 2
        id1 = 100 + (sum(b[:mid]) % 4000)
        id2 = 100 + (sum(b[mid:]) % 4000)
        return [id1, id2]


class CurriculumTrainingRunner:
    """Runs CPU-optimized multi-stage curriculum fine-tuning (Monolingual -> Real Mixed -> Optional Synthetic)
    for Whisper Small with PEFT LoRA, Auxiliary LID Head, lazy audio loading, RAM monitoring, validation WER/Boundary WER
    tracking, early stopping, and checkpoint interruption/resume support.
    """

    def __init__(self, config_or_path: str | Path | Dict[str, Any]) -> None:
        if isinstance(config_or_path, (str, Path)):
            with Path(config_or_path).open("r", encoding="utf-8") as f:
                self.config: Dict[str, Any] = yaml.safe_load(f)
        else:
            self.config = dict(config_or_path)

    @torch.no_grad()
    def _evaluate_validation_subset(
        self,
        peft_model: nn.Module,
        lid_head: Optional[AuxiliaryLIDHead],
        processor: Optional[Any],
        feature_extractor: Any,
        tokenizer: Any,
        val_records: List[ManifestRecord],
        lid_weight: float,
        device: str,
        base_dir: Optional[Path],
        normalizer: TranscriptNormalizer,
    ) -> Dict[str, Any]:
        """Evaluate loss, overall WER, and Boundary WER on a validation subset (`split == 'val'`).
        Never touches the held-out `test` split.
        """
        if not val_records:
            return {"status": "no_val_records"}

        was_training = peft_model.training
        peft_model.eval()
        if lid_head is not None:
            lid_head.eval()

        val_ds = LazyWhisperLIDDataset(
            records=val_records,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            base_dir=base_dir,
        )
        total_val_loss = 0.0
        total_val_asr_loss = 0.0
        total_val_lid_loss = 0.0
        hyps: Dict[str, str] = {}
        times: Dict[str, float] = {}

        for idx, rec in enumerate(val_records):
            item = val_ds[idx]
            batch = collate_whisper_lid_batch([item])
            in_feat = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            lid_labels = batch["lid_labels"].to(device)

            loss_out = compute_joint_asr_lid_loss(
                peft_model=peft_model,
                lid_head=lid_head,
                input_features=in_feat,
                labels=labels,
                lid_labels=lid_labels,
                lid_loss_weight=lid_weight,
            )
            total_val_loss += float(loss_out["total_loss"].detach().cpu().item())
            total_val_asr_loss += float(loss_out["asr_loss"].detach().cpu().item())
            total_val_lid_loss += float(loss_out["lid_loss"].detach().cpu().item())

            if processor is not None:
                t0 = time.perf_counter()
                try:
                    gen_ids = peft_model.generate(
                        input_features=in_feat,
                        task="transcribe",
                        num_beams=1,
                        max_new_tokens=64,
                        no_repeat_ngram_size=3,
                    )
                except Exception:
                    gen_ids = peft_model.generate(
                        input_features=in_feat,
                        num_beams=1,
                        max_new_tokens=64,
                        no_repeat_ngram_size=3,
                    )
                import re

                raw_tx = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                clean_tx = " ".join(re.sub(r"<\|[^|>]+\|>", " ", raw_tx).strip().split())
                hyps[rec.utterance_id] = clean_tx
                times[rec.utterance_id] = round(time.perf_counter() - t0, 4)

        n_val = max(1, len(val_records))
        metrics: Dict[str, Any] = {
            "num_val_utterances": len(val_records),
            "val_total_loss": round(total_val_loss / n_val, 6),
            "val_asr_loss": round(total_val_asr_loss / n_val, 6),
            "val_lid_loss": round(total_val_lid_loss / n_val, 6),
        }

        if hyps and len(hyps) == len(val_records):
            val_summary = evaluate_corpus(
                records=val_records,
                hypotheses=hyps,
                model_name="val_eval",
                checkpoint_id="in_memory",
                run_tier="smoke_test",
                transcription_times=times,
                normalizer=normalizer,
                is_simulated=False,
            )
            metrics["val_overall_wer"] = val_summary.overall_wer
            metrics["val_boundary_wer"] = val_summary.boundary_wer
            metrics["val_non_boundary_wer"] = val_summary.non_boundary_wer

        if was_training:
            peft_model.train()
            if lid_head is not None:
                lid_head.train()
        return metrics

    def run(
        self,
        manifest_override: Optional[str | Path] = None,
        output_dir_override: Optional[str | Path] = None,
        base_dir: Optional[Path] = None,
        pilot_mode: bool = False,
        resume: bool = False,
        simulate_interrupt_after_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        run_t0 = time.perf_counter()
        hw_info = detect_cpu_hardware_info()
        train_cfg = self.config.get("training", {})
        hw_cfg = self.config.get("hardware", {})
        lora_cfg = self.config.get("lora", {})
        lid_cfg = self.config.get("auxiliary_lid", {})
        curr_cfg = self.config.get("curriculum", {})

        # Configure CPU threads to prevent hybrid-core contention
        num_threads = int(hw_cfg.get("num_threads", min(8, hw_info["physical_cores"])))
        torch.set_num_threads(num_threads)
        hw_info["configured_torch_threads"] = torch.get_num_threads()

        seed = int(train_cfg.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        manifest_p = Path(manifest_override or train_cfg.get("manifest_path", "data/full_corpus/unified_manifest.jsonl"))
        if not manifest_p.is_absolute() and base_dir is not None:
            manifest_p = base_dir / manifest_p
        records = load_manifest(manifest_p, validate=True)

        # Ensure zero test records are ever used for training or validation monitoring
        train_pool = [r for r in records if r.split == "train"]
        val_pool = [r for r in records if r.split == "val"]
        val_count = 2 if pilot_mode else int(train_cfg.get("val_subset_utterances", 4))
        val_subset = [r for r in val_pool if ("HI" in r.lang_tags and "EN" in r.lang_tags)][:val_count]
        if len(val_subset) < val_count:
            val_subset = val_pool[:val_count]

        use_lid = bool(lid_cfg.get("enabled", True)) and float(lid_cfg.get("lid_loss_weight", 0.0)) > 0.0
        lid_weight = float(lid_cfg.get("lid_loss_weight", 0.3)) if use_lid else 0.0
        grad_ckpt = bool(hw_cfg.get("gradient_checkpointing", False))

        out_dir = Path(output_dir_override or train_cfg.get("output_dir", "checkpoints/default"))
        if not out_dir.is_absolute() and base_dir is not None:
            out_dir = base_dir / out_dir

        bundle = build_whisper_lora_lid_model(
            base_model_name_or_path=self.config.get("base_model_name_or_path", "openai/whisper-small"),
            lora_r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            projection_names=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"]),
            decoder_attention_only=bool(lora_cfg.get("decoder_attention_only", True)),
            use_lid_head=use_lid,
            lid_num_labels=int(lid_cfg.get("num_labels", 3)),
            lid_hidden_dropout=float(lid_cfg.get("hidden_dropout", 0.1)),
            local_tiny_fallback_if_offline=bool(self.config.get("allow_tiny_architecture_for_cpu_smoke", False)),
            device=hw_cfg.get("device"),
            gradient_checkpointing=grad_ckpt,
        )

        peft_model = bundle["peft_model"]
        lid_head = bundle["lid_head"]
        processor = bundle["processor"]
        device = bundle["device"]

        # Snapshot initial parameter fingerprints BEFORE any updates (or before resume)
        initial_param_snap = snapshot_parameter_fingerprints(peft_model, lid_head)

        # Check for resumable checkpoint state
        completed_stage_names: List[str] = []
        stage_histories: List[Dict[str, Any]] = []
        global_steps_done = 0
        resumed_from_checkpoint = False
        resume_state_p = out_dir / "resume_state.pt"

        if (resume or bool(train_cfg.get("resume_if_exists", False))) and resume_state_p.exists() and (out_dir / "lora_adapter").exists():
            try:
                saved_resume = torch.load(str(resume_state_p), map_location=device, weights_only=False)
                completed_stage_names = list(saved_resume.get("completed_stage_names", []))
                stage_histories = list(saved_resume.get("stage_histories", []))
                global_steps_done = int(saved_resume.get("global_steps_done", 0))
                # Load adapter weights into peft_model
                reloaded_bundle = load_finetuned_whisper_lora_lid(out_dir, device=str(device))
                peft_model.load_state_dict(reloaded_bundle["peft_model"].state_dict())
                if lid_head is not None and reloaded_bundle["lid_head"] is not None:
                    lid_head.load_state_dict(reloaded_bundle["lid_head"].state_dict())
                del reloaded_bundle
                gc.collect()
                resumed_from_checkpoint = True
            except Exception:
                resumed_from_checkpoint = False

        tokenizer = processor.tokenizer if processor is not None else SimpleWordSubwordTokenizer()
        feature_extractor = processor.feature_extractor if processor is not None else WhisperFeatureExtractor()
        normalizer = TranscriptNormalizer((base_dir or Path(".")) / "configs" / "normalization_equivalents.yaml")

        stages = curr_cfg.get(
            "stages",
            [
                {"name": "stage1_monolingual", "filter_mode": "monolingual", "epochs": 1, "learning_rate": 1e-4},
                {"name": "stage2_real_mixed", "filter_mode": "real_mixed", "epochs": 1, "learning_rate": 8e-5},
                {"name": "stage3_synthetic_supplement", "filter_mode": "all_with_synthetic", "epochs": 1, "learning_rate": 5e-5},
            ],
        )
        disable_stage3 = bool(curr_cfg.get("disable_stage3_on_cpu", False))

        batch_size = 1 if pilot_mode else int(train_cfg.get("batch_size", 1))
        grad_accum_steps = 4 if pilot_mode else max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
        default_max_steps = 2 if pilot_mode else int(train_cfg.get("max_steps_per_stage", 10))
        use_fp16 = bool(train_cfg.get("fp16", False)) and str(device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16) if use_fp16 else None
        patience = int(train_cfg.get("early_stopping_patience", 3))
        best_val_loss = float("inf")
        patience_counter = 0

        peak_rss_mb = get_process_rss_mb()
        interrupted = False

        peft_model.train()
        if lid_head is not None:
            lid_head.train()

        try:
            for stage_spec in stages:
                stage_name = stage_spec["name"]
                if stage_spec.get("enabled") is False:
                    continue
                if disable_stage3 and "stage3" in stage_name.lower():
                    continue
                if stage_name in completed_stage_names:
                    continue

                filter_mode = stage_spec.get("filter_mode", "all_with_synthetic")
                epochs = int(stage_spec.get("epochs", 1))
                lr = float(stage_spec.get("learning_rate", 1e-4))
                weight_decay = float(train_cfg.get("weight_decay", 0.01))
                stage_max_steps = 2 if pilot_mode else int(stage_spec.get("max_steps", default_max_steps))

                stage_records = filter_records_for_curriculum_stage(train_pool, filter_mode)
                max_utts = (
                    (batch_size * grad_accum_steps * stage_max_steps)
                    if pilot_mode
                    else int(stage_spec.get("max_utterances", stage_max_steps * batch_size * grad_accum_steps))
                )
                active_stage_records = stage_records[:max_utts]

                lazy_dataset = LazyWhisperLIDDataset(
                    records=active_stage_records,
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    base_dir=base_dir,
                )
                loader = DataLoader(
                    lazy_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=int(hw_cfg.get("dataloader_workers", 0)),
                    collate_fn=collate_whisper_lid_batch,
                )

                params_to_opt = [p for p in peft_model.parameters() if p.requires_grad]
                if lid_head is not None:
                    params_to_opt.extend([p for p in lid_head.parameters() if p.requires_grad])
                optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=weight_decay)

                step_losses: List[Dict[str, Any]] = []
                steps_done = 0
                micro_step = 0
                accum_total_loss = 0.0
                accum_asr_loss = 0.0
                accum_lid_loss = 0.0
                accum_count = 0
                step_t0 = time.perf_counter()
                optimizer.zero_grad()

                for _ in range(epochs):
                    for batch_idx, batch in enumerate(loader):
                        if steps_done >= stage_max_steps:
                            break
                        in_feat = batch["input_features"].to(device)
                        labels = batch["labels"].to(device)
                        lid_labels = batch["lid_labels"].to(device)

                        with torch.amp.autocast("cuda", enabled=use_fp16):
                            loss_out = compute_joint_asr_lid_loss(
                                peft_model=peft_model,
                                lid_head=lid_head,
                                input_features=in_feat,
                                labels=labels,
                                lid_labels=lid_labels,
                                lid_loss_weight=lid_weight,
                            )
                            scaled_loss = loss_out["total_loss"] / float(grad_accum_steps)

                        if scaler is not None:
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()

                        accum_total_loss += float(loss_out["total_loss"].detach().cpu().item())
                        accum_asr_loss += float(loss_out["asr_loss"].detach().cpu().item())
                        accum_lid_loss += float(loss_out["lid_loss"].detach().cpu().item())
                        accum_count += 1
                        micro_step += 1

                        is_last_batch = (batch_idx + 1) >= len(loader)
                        if (micro_step % grad_accum_steps == 0) or is_last_batch:
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(params_to_opt, 1.0)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(params_to_opt, 1.0)
                                optimizer.step()
                            optimizer.zero_grad()

                            step_dur = round(time.perf_counter() - step_t0, 4)
                            curr_rss = get_process_rss_mb()
                            if curr_rss > peak_rss_mb:
                                peak_rss_mb = curr_rss

                            steps_done += 1
                            global_steps_done += 1
                            denom = max(1, accum_count)
                            step_losses.append(
                                {
                                    "step": steps_done,
                                    "global_step": global_steps_done,
                                    "micro_batches": accum_count,
                                    "total_loss": round(accum_total_loss / denom, 6),
                                    "asr_loss": round(accum_asr_loss / denom, 6),
                                    "lid_loss": round(accum_lid_loss / denom, 6),
                                    "step_duration_sec": step_dur,
                                    "process_rss_mb": curr_rss,
                                }
                            )
                            accum_total_loss = 0.0
                            accum_asr_loss = 0.0
                            accum_lid_loss = 0.0
                            accum_count = 0
                            step_t0 = time.perf_counter()

                            if simulate_interrupt_after_steps is not None and global_steps_done >= simulate_interrupt_after_steps:
                                if steps_done >= stage_max_steps:
                                    stage_histories.append(
                                        {
                                            "stage_name": stage_name,
                                            "filter_mode": filter_mode,
                                            "num_utterances_available": len(stage_records),
                                            "num_utterances_loaded": len(active_stage_records),
                                            "utterance_ids": [r.utterance_id for r in active_stage_records],
                                            "steps_executed": steps_done,
                                            "micro_steps_executed": micro_step,
                                            "initial_loss": step_losses[0]["total_loss"] if step_losses else None,
                                            "final_loss": step_losses[-1]["total_loss"] if step_losses else None,
                                            "step_losses": step_losses,
                                            "validation_metrics": {"status": "interrupted_before_val"},
                                        }
                                    )
                                    completed_stage_names.append(stage_name)
                                raise KeyboardInterrupt("Simulated training interruption after target step.")
                    if steps_done >= stage_max_steps:
                        break

                # Evaluate on validation subset (split == 'val') at end of stage/epoch
                val_metrics = self._evaluate_validation_subset(
                    peft_model=peft_model,
                    lid_head=lid_head,
                    processor=processor,
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    val_records=val_subset,
                    lid_weight=lid_weight,
                    device=str(device),
                    base_dir=base_dir,
                    normalizer=normalizer,
                )

                stage_histories.append(
                    {
                        "stage_name": stage_name,
                        "filter_mode": filter_mode,
                        "num_utterances_available": len(stage_records),
                        "num_utterances_loaded": len(active_stage_records),
                        "utterance_ids": [r.utterance_id for r in active_stage_records],
                        "steps_executed": steps_done,
                        "micro_steps_executed": micro_step,
                        "initial_loss": step_losses[0]["total_loss"] if step_losses else None,
                        "final_loss": step_losses[-1]["total_loss"] if step_losses else None,
                        "step_losses": step_losses,
                        "validation_metrics": val_metrics,
                    }
                )
                completed_stage_names.append(stage_name)

                # Save epoch/stage checkpoint + resume state
                save_finetuned_whisper_lora_lid(
                    output_dir=out_dir,
                    peft_model=peft_model,
                    lid_head=lid_head,
                    processor=processor,
                    training_config=self.config,
                    resume_state={
                        "completed_stage_names": completed_stage_names,
                        "stage_histories": stage_histories,
                        "global_steps_done": global_steps_done,
                    },
                )

                val_l = val_metrics.get("val_total_loss")
                if val_l is not None:
                    if val_l < best_val_loss - 1e-4:
                        best_val_loss = val_l
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break

        except KeyboardInterrupt:
            interrupted = True
            save_finetuned_whisper_lora_lid(
                output_dir=out_dir,
                peft_model=peft_model,
                lid_head=lid_head,
                processor=processor,
                training_config=self.config,
                resume_state={
                    "completed_stage_names": completed_stage_names,
                    "stage_histories": stage_histories,
                    "global_steps_done": global_steps_done,
                    "interrupted": True,
                },
            )

        param_update_report = verify_parameter_updates(peft_model, lid_head, initial_param_snap)
        del initial_param_snap
        gc.collect()

        total_elapsed_sec = round(time.perf_counter() - run_t0, 2)
        final_rss = get_process_rss_mb()
        if final_rss > peak_rss_mb:
            peak_rss_mb = final_rss

        training_metrics = {
            "experiment_name": self.config.get("experiment_name", "whisper_lora_experiment"),
            "run_tier": "cpu_pilot" if pilot_mode else self.config.get("run_tier", "smoke_test"),
            "status": "interrupted_checkpoint_saved" if interrupted else "completed",
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "base_model_name_or_path": self.config.get("base_model_name_or_path", "openai/whisper-small"),
            "device": str(device),
            "fp16_enabled": use_fp16,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "auxiliary_lid_enabled": use_lid,
            "lid_loss_weight": lid_weight,
            "used_tiny_fallback": bundle["used_tiny_fallback"],
            "is_real_pretrained_base": not bundle["used_tiny_fallback"],
            "usable_for_speech_recognition": not bundle["used_tiny_fallback"],
            "hardware_and_memory": {
                **hw_info,
                "peak_process_rss_mb": peak_rss_mb,
                "final_process_rss_mb": final_rss,
                "total_training_time_sec": total_elapsed_sec,
                "total_optimizer_steps": global_steps_done,
            },
            "parameter_update_verification": param_update_report,
            "param_summary": bundle["param_summary"],
            "curriculum_stages": stage_histories,
        }

        save_finetuned_whisper_lora_lid(
            output_dir=out_dir,
            peft_model=peft_model,
            lid_head=lid_head,
            processor=processor,
            training_config=self.config,
            training_metrics=training_metrics,
            resume_state={
                "completed_stage_names": completed_stage_names,
                "stage_histories": stage_histories,
                "global_steps_done": global_steps_done,
                "interrupted": interrupted,
            },
        )

        reload_check = verify_checkpoint_reload(
            checkpoint_dir=out_dir,
            original_peft_model=peft_model,
            original_lid_head=lid_head,
        )
        training_metrics["reload_verification"] = reload_check

        with (out_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(training_metrics, f, indent=2, ensure_ascii=False)

        return training_metrics

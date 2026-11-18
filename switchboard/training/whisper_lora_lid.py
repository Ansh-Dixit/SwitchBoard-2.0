"""PEFT LoRA fine-tuning for Whisper decoder attention with an auxiliary Language Identification (LID) head."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from switchboard.data.manifest import TARGET_SAMPLE_RATE, ManifestRecord, load_manifest
from switchboard.evaluation.boundary import compute_cmi
from switchboard.models.runners import load_audio_16k

LID_TAG_TO_ID: Dict[str, int] = {"HI": 0, "EN": 1, "OTHER": 2}
ID_TO_LID_TAG: Dict[int, str] = {0: "HI", 1: "EN", 2: "OTHER"}
IGNORE_INDEX: int = -100


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
    """Map word-level language tags (HI, EN, OTHER) to Whisper decoder subword positions.

    Rules:
    - All prefix special tokens (e.g., <|startoftranscript|>, <|hi|>, <|transcribe|>, <|notimestamps|>)
      receive lid_label = IGNORE_INDEX (-100).
    - Each word w_k with language tag t_k is tokenized into one or more subword token IDs; every non-special
      subword token belonging to w_k receives lid_label = LID_TAG_TO_ID[t_k].
    - Any special token IDs appearing anywhere, the final <|endoftext|> token, and any padding positions
      receive lid_label = IGNORE_INDEX (-100).
    """
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

    # 1. Prefix special tokens -> masked (-100) for LID head
    for pid in prefix_token_ids:
        token_ids.append(int(pid))
        lid_labels.append(IGNORE_INDEX)

    # 2. Word-by-word subword tokenization and tag projection
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

    # 3. Truncate before appending EOS if needed
    max_content = max(2, max_length - 1)
    if len(token_ids) > max_content:
        token_ids = token_ids[:max_content]
        lid_labels = lid_labels[:max_content]

    # 4. Append EOS token -> masked (-100) for LID head
    if eos_token_id is not None:
        token_ids.append(int(eos_token_id))
        lid_labels.append(IGNORE_INDEX)

    return token_ids, lid_labels


def collate_whisper_lid_batch(
    batch_items: List[Dict[str, Any]],
    pad_token_id: int = 50257,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of (input_features, labels, lid_labels) and mask padding positions with -100."""
    input_features = torch.stack([item["input_features"] for item in batch_items], dim=0)
    max_seq = max(len(item["token_ids"]) for item in batch_items)

    batch_labels: List[List[int]] = []
    batch_lid_labels: List[List[int]] = []

    for item in batch_items:
        t_ids = list(item["token_ids"])
        l_ids = list(item["lid_labels"])
        pad_len = max_seq - len(t_ids)
        # In WhisperForConditionalGeneration, -100 in `labels` masks padding from transcription cross-entropy loss
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
) -> Dict[str, Any]:
    """Run forward pass through Whisper + LoRA and compute combined ASR + weighted Auxiliary LID loss."""
    need_hidden = (lid_head is not None) and (lid_loss_weight > 0.0)
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
    """Filter training records according to the curriculum stage:
    1. 'monolingual': non-synthetic utterances with CMI == 0.0 (solely HI or solely EN, plus optional OTHER)
    2. 'real_mixed': non-synthetic utterances with CMI > 0.0 (real Hindi-English code-switching)
    3. 'all_with_synthetic' / 'synthetic_supplement': real mixed + synthetic mixed utterances in the train split
    """
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
        # Real mixed + synthetic mixed speech (strictly restricted to train split)
        filtered = [
            r
            for r in train_recs
            if r.is_synthetic or ("synthetic" in r.source.lower()) or compute_cmi(r.lang_tags) > 1e-9
        ]
        return filtered if filtered else train_recs

    return train_recs


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
        # Offline/CPU smoke-test fallback using a compact WhisperConfig with the exact same decoder architecture
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

    # 1. Explicitly freeze all base Whisper parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # 2. Target only decoder attention layers (self_attn & encoder_attn)
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
) -> Path:
    """Save PEFT LoRA adapter, Auxiliary LID head weights, base model config, and training configuration."""
    out_root = Path(output_dir)
    adapter_dir = out_root / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Save PEFT LoRA adapter weights and adapter_config.json
    peft_model.save_pretrained(str(adapter_dir))

    # Save base WhisperConfig so standalone/offline reloads know exact architecture dimensions
    base_cfg = peft_model.base_model.model.config
    base_cfg.save_pretrained(str(out_root / "base_config"))

    if processor is not None:
        try:
            processor.save_pretrained(str(out_root / "processor"))
        except Exception:
            pass

    # Save Auxiliary LID head if present
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
    """Verify that a saved LoRA adapter and LID head reload cleanly and match pre-save forward-pass outputs."""
    original_peft_model.eval()
    if original_lid_head is not None:
        original_lid_head.eval()

    device = next(original_peft_model.parameters()).device
    base_cfg = original_peft_model.base_model.model.config

    # Extract frozen base model state dict (excluding lora weights) for deterministic verification
    base_sd: Dict[str, torch.Tensor] = {}
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

    if sample_input_features is None:
        num_mel = int(getattr(base_cfg, "num_mel_bins", 80))
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
    out_rel = rel_peft(
        input_features=sample_input_features,
        decoder_input_ids=sample_decoder_input_ids,
        output_hidden_states=True,
        return_dict=True,
    )

    asr_max_diff = float((out_orig.logits - out_rel.logits).abs().max().item())
    asr_matches = bool(torch.allclose(out_orig.logits, out_rel.logits, atol=1e-5))

    lid_matches = True
    lid_max_diff = 0.0
    if original_lid_head is not None:
        if rel_lid is None:
            lid_matches = False
        else:
            lid_orig = original_lid_head(out_orig.decoder_hidden_states[-1])
            lid_rel = rel_lid(out_rel.decoder_hidden_states[-1])
            lid_max_diff = float((lid_orig - lid_rel).abs().max().item())
            lid_matches = bool(torch.allclose(lid_orig, lid_rel, atol=1e-5))

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "asr_logits_match": asr_matches,
        "asr_max_abs_diff": asr_max_diff,
        "lid_logits_match": lid_matches,
        "lid_max_abs_diff": lid_max_diff,
        "reload_verified": asr_matches and lid_matches,
    }


class SimpleWordSubwordTokenizer:
    """Fallback subword tokenizer used only when offline smoke testing without a cached WhisperTokenizer."""

    all_special_ids = [50257, 50258, 50259, 50359, 50363]
    bos_token_id = 50258
    eos_token_id = 50257
    pad_token_id = 50257

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        stripped = text.strip()
        if not stripped:
            return []
        # Deterministic 2-subword split per word to test multi-subword tag projection
        b = stripped.encode("utf-8")
        if len(b) == 1:
            return [100 + (b[0] % 4000)]
        mid = len(b) // 2
        id1 = 100 + (sum(b[:mid]) % 4000)
        id2 = 100 + (sum(b[mid:]) % 4000)
        return [id1, id2]


class CurriculumTrainingRunner:
    """Runs multi-stage curriculum fine-tuning (Monolingual -> Real Mixed -> Optional Synthetic Supplement)
    for Whisper Small with PEFT LoRA and configurable Auxiliary LID Head.
    """

    def __init__(self, config_or_path: str | Path | Dict[str, Any]) -> None:
        if isinstance(config_or_path, (str, Path)):
            with Path(config_or_path).open("r", encoding="utf-8") as f:
                self.config: Dict[str, Any] = yaml.safe_load(f)
        else:
            self.config = dict(config_or_path)

    def run(
        self,
        manifest_override: Optional[str | Path] = None,
        output_dir_override: Optional[str | Path] = None,
        base_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        train_cfg = self.config.get("training", {})
        lora_cfg = self.config.get("lora", {})
        lid_cfg = self.config.get("auxiliary_lid", {})
        curr_cfg = self.config.get("curriculum", {})

        seed = int(train_cfg.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        manifest_p = Path(manifest_override or train_cfg.get("manifest_path", "data/demo_fixture/manifest.jsonl"))
        if not manifest_p.is_absolute() and base_dir is not None:
            manifest_p = base_dir / manifest_p
        records = load_manifest(manifest_p, validate=True)

        use_lid = bool(lid_cfg.get("enabled", True)) and float(lid_cfg.get("lid_loss_weight", 0.0)) > 0.0
        lid_weight = float(lid_cfg.get("lid_loss_weight", 0.3)) if use_lid else 0.0

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
            local_tiny_fallback_if_offline=bool(self.config.get("allow_tiny_architecture_for_cpu_smoke", True)),
        )

        peft_model = bundle["peft_model"]
        lid_head = bundle["lid_head"]
        processor = bundle["processor"]
        device = bundle["device"]

        tokenizer = processor.tokenizer if processor is not None else SimpleWordSubwordTokenizer()
        feature_extractor = processor.feature_extractor if processor is not None else WhisperFeatureExtractor()

        stages = curr_cfg.get(
            "stages",
            [
                {"name": "stage1_monolingual", "filter_mode": "monolingual", "epochs": 1, "learning_rate": 1e-4},
                {"name": "stage2_real_mixed", "filter_mode": "real_mixed", "epochs": 1, "learning_rate": 8e-5},
                {"name": "stage3_synthetic_supplement", "filter_mode": "all_with_synthetic", "epochs": 1, "learning_rate": 5e-5},
            ],
        )

        batch_size = int(train_cfg.get("batch_size", 2))
        max_steps_per_stage = int(train_cfg.get("max_steps_per_stage", 10))
        stage_histories: List[Dict[str, Any]] = []

        peft_model.train()
        if lid_head is not None:
            lid_head.train()

        for stage_spec in stages:
            stage_name = stage_spec["name"]
            filter_mode = stage_spec.get("filter_mode", "all_with_synthetic")
            epochs = int(stage_spec.get("epochs", 1))
            lr = float(stage_spec.get("learning_rate", 1e-4))

            stage_records = filter_records_for_curriculum_stage(records, filter_mode)
            params_to_opt = [p for p in peft_model.parameters() if p.requires_grad]
            if lid_head is not None:
                params_to_opt.extend([p for p in lid_head.parameters() if p.requires_grad])
            optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=0.01)

            # Prepare acoustic features & aligned subword LID labels (slice to active training budget for efficiency)
            max_items_needed = max(batch_size, max_steps_per_stage * batch_size)
            active_stage_records = stage_records[:max_items_needed]
            prepared_items: List[Dict[str, Any]] = []
            for rec in active_stage_records:
                audio_p = Path(rec.audio_path)
                if not audio_p.is_absolute() and base_dir is not None:
                    audio_p = base_dir / audio_p
                waveform, _ = load_audio_16k(audio_p)
                mel = feature_extractor(waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt").input_features[0]
                t_ids, l_ids = align_word_tags_to_subwords(
                    words=rec.words,
                    lang_tags=rec.lang_tags,
                    tokenizer=tokenizer,
                    prefix_token_ids=[50258, 50259, 50359, 50363],
                    eos_token_id=50257,
                )
                prepared_items.append(
                    {
                        "utterance_id": rec.utterance_id,
                        "input_features": mel,
                        "token_ids": t_ids,
                        "lid_labels": l_ids,
                    }
                )

            step_losses: List[Dict[str, float]] = []
            steps_done = 0
            for _ in range(epochs):
                for start_i in range(0, len(prepared_items), batch_size):
                    if steps_done >= max_steps_per_stage:
                        break
                    batch_slice = prepared_items[start_i : start_i + batch_size]
                    batch = collate_whisper_lid_batch(batch_slice)
                    in_feat = batch["input_features"].to(device)
                    labels = batch["labels"].to(device)
                    lid_labels = batch["lid_labels"].to(device)

                    optimizer.zero_grad()
                    loss_out = compute_joint_asr_lid_loss(
                        peft_model=peft_model,
                        lid_head=lid_head,
                        input_features=in_feat,
                        labels=labels,
                        lid_labels=lid_labels,
                        lid_loss_weight=lid_weight,
                    )
                    loss_out["total_loss"].backward()
                    torch.nn.utils.clip_grad_norm_(params_to_opt, 1.0)
                    optimizer.step()

                    step_losses.append(
                        {
                            "step": steps_done + 1,
                            "total_loss": round(float(loss_out["total_loss"].detach().cpu().item()), 6),
                            "asr_loss": round(float(loss_out["asr_loss"].detach().cpu().item()), 6),
                            "lid_loss": round(float(loss_out["lid_loss"].detach().cpu().item()), 6),
                        }
                    )
                    steps_done += 1
                if steps_done >= max_steps_per_stage:
                    break

            stage_histories.append(
                {
                    "stage_name": stage_name,
                    "filter_mode": filter_mode,
                    "num_utterances": len(stage_records),
                    "utterance_ids": [r.utterance_id for r in stage_records],
                    "steps_executed": steps_done,
                    "initial_loss": step_losses[0]["total_loss"] if step_losses else None,
                    "final_loss": step_losses[-1]["total_loss"] if step_losses else None,
                    "step_losses": step_losses,
                }
            )

        out_dir = Path(output_dir_override or train_cfg.get("output_dir", "outputs/smoke_checkpoints/default"))
        if not out_dir.is_absolute() and base_dir is not None:
            out_dir = base_dir / out_dir

        training_metrics = {
            "experiment_name": self.config.get("experiment_name", "whisper_lora_experiment"),
            "run_tier": self.config.get("run_tier", "smoke_test"),
            "auxiliary_lid_enabled": use_lid,
            "lid_loss_weight": lid_weight,
            "used_tiny_fallback": bundle["used_tiny_fallback"],
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
        )

        reload_check = verify_checkpoint_reload(
            checkpoint_dir=out_dir,
            original_peft_model=peft_model,
            original_lid_head=lid_head,
        )
        training_metrics["reload_verification"] = reload_check

        # Re-save training_metrics.json with reload_verification included
        with (out_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(training_metrics, f, indent=2, ensure_ascii=False)

        return training_metrics

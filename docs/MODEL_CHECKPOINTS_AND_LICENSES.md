# Verified Model Checkpoints, Language Coverage, Access, and Licenses

This document records the verification of the baseline and comparison checkpoints used in **SwitchBoard**.

---

## Summary Table of Verified Checkpoints

| Role | Checkpoint ID | Architecture | Language Coverage | Access / Gating | License | CTC Head / Ready Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Baseline & Fine-Tuning Base** | `openai/whisper-small` | Whisper Encoder-Decoder (244M params, 12 enc / 12 dec layers) | Multilingual (99 languages including Hindi `hi` and English `en`) | Public, non-gated on Hugging Face Hub | **Apache-2.0** | Ready out-of-the-box for seq2seq transcription (`task="transcribe"`) |
| **Comparison Baseline 1a (AI4Bharat IndicWhisper)** | `ai4bharat/whisper-medium-hi` (Vistaar distribution) / `vasista22/whisper-hindi-small` | Whisper Seq2Seq fine-tuned on Indian speech corpora (Kathbath, Shrutilipi, Vistaar) | Hindi (`hi`) + Devanagari transliterated English loanwords | `ai4bharat` Vistaar models are hosted via AI4Bharat Vistaar release / HF (`vasista22/whisper-hindi-small` public on HF Hub) | **MIT / CC-BY-4.0** (AI4Bharat Vistaar) | Ready for seq2seq transcription; outputs Devanagari script (requires SwitchBoard cross-script normalization for fair scoring on Romanized English tokens) |
| **Comparison Baseline 2a (Pretrained XLS-R Base)** | `facebook/wav2vec2-xls-r-300m` | wav2vec 2.0 XLS-R Self-Supervised Encoder (300M params) | Pretrained on 436k hrs across 128 languages (including `hi` and `en`) | Public, non-gated on Hugging Face Hub | **Apache-2.0** | **Requires training a CTC head (`lm_head`) and tokenizer vocabulary.** Cannot transcribe raw speech without a fine-tuned CTC linear projection layer. |
| **Comparison Baseline 2b (Fine-Tuned XLS-R CTC — Hindi / English)** | `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` (Hindi CTC) & `jonatasgrosman/wav2vec2-large-xlsr-53-english` (English CTC) | wav2vec 2.0 / XLS-R + Fine-Tuned Linear CTC Head | Hindi (`hi` Devanagari vocab) / English (`en` Latin vocab) | Public, non-gated on Hugging Face Hub | **MIT / Apache-2.0** | **CTC head already trained.** Single-alphabet CTC heads are constrained to one character set (Devanagari or Latin), highlighting why bilingual code-switching degrades monolithic CTC decoders without a bilingual character vocabulary or transliteration normalization. |

---

## Detailed Notes on Comparison Runners

### 1. Primary Baseline: `openai/whisper-small`
- **Why Primary:** Uses the exact same pretrained base checkpoint (`openai/whisper-small`) as our PEFT LoRA and LoRA+LID fine-tuned models, guaranteeing a strictly controlled before-and-after comparison.
- **Decoding Mode:** Always run in transcription mode (`task="transcribe"`), never translation mode (`task="translate"`), so Hindi words are preserved as Hindi rather than translated into English.

### 2. AI4Bharat IndicWhisper Runner (`IndicWhisperRunner`)
- **Background:** AI4Bharat released IndicWhisper models under the **Vistaar** benchmark covering 12 Indian languages including Hindi (`whisper-medium-hi` / `whisper-large-v2-hi`). Because some AI4Bharat checkpoints are distributed as archive downloads from the Vistaar repository or mirrored on Hugging Face (e.g., `vasista22/whisper-hindi-small` / local path to extracted Vistaar Hindi checkpoint), `IndicWhisperRunner` accepts either a Hugging Face checkpoint ID or a local Vistaar directory path.
- **Behavior on Hindi–English Code-Switching:** IndicWhisper produces Devanagari output even for embedded English words (e.g., outputting `"मीटिंग"` for `"meeting"`). SwitchBoard's `TranscriptNormalizer` maps reviewed Devanagari-transliterated English equivalents (`"मीटिंग" → "meeting"`) so IndicWhisper is not unfairly penalized for script choice.

### 3. wav2vec 2.0 XLS-R CTC Runner (`Wav2Vec2CTCRunner`)
- **Explicit CTC Head Dependency Check:**
  - The base `facebook/wav2vec2-xls-r-300m` checkpoint contains only the self-supervised feature extractor and transformer encoder; it has **no trained CTC classification head (`lm_head`) or character tokenizer (`vocab.json`)**.
  - `Wav2Vec2CTCRunner` automatically inspects the checkpoint before inference:
    1. If pointed at an un-finetuned SSL checkpoint like `facebook/wav2vec2-xls-r-300m`, it detects the missing CTC head/tokenizer and explicitly reports `status="requires_ctc_head_training"` with full instructions rather than silently producing random characters from an uninitialized linear head.
    2. When pointed at a fine-tuned CTC checkpoint (e.g., `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` or a custom bilingual HI-EN CTC checkpoint trained on the unified manifest), it verifies that `lm_head` weights are present in the checkpoint and runs CTC greedy/beam decoding.

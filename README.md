# SwitchBoard — Hindi–English Code-Switching Speech Recognition and Boundary Error Analysis

**SwitchBoard** is an end-to-end speech recognition and error-analysis framework built with **Python, PyTorch, Hugging Face Transformers, PEFT (LoRA), and Streamlit**. It compares baseline and fine-tuned ASR models on **Hindi–English (Hinglish)** code-switched speech, focusing specifically on recognition errors at **language-switch boundaries** (`HI → EN` and `EN → HI` transitions).

For example, in the utterance:
> `"kal meeting ke baad main report submit kar dunga"`  
> Tags: `["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"]`

- `"kal"` (`HI`) → `"meeting"` (`EN`) is a `HI → EN` switch boundary.
- `"meeting"` (`EN`) → `"ke"` (`HI`) is an `EN → HI` switch boundary (making `"meeting"` an overlapping boundary word touched by two switches, counted **once** in the pooled boundary WER).
- `"main"` (`HI`) → `"report"` (`EN`) is a `HI → EN` switch boundary.
- `"report"` (`EN`) → `"submit"` (`EN`) stays within English (non-switch).
- `"submit"` (`EN`) → `"kar"` (`HI`) is an `EN → HI` switch boundary.

---

## 1. Project Architecture & Directory Structure

```text
SwitchBoard/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── normalization_equivalents.yaml       # Reviewed cross-script & Romanized spelling equivalents ("मीटिंग" <-> "meeting")
│   ├── train_smoke_lora_only.yaml           # Smoke-test config: PEFT LoRA alone (lid_loss_weight = 0.0)
│   ├── train_smoke_lora_lid.yaml            # Smoke-test config: PEFT LoRA + Auxiliary LID Head (lid_loss_weight = 0.30)
│   ├── train_colab_full_lora_only.yaml      # Full Colab GPU curriculum config: LoRA alone
│   └── train_colab_full_lora_lid.yaml       # Full Colab GPU curriculum config: LoRA + Auxiliary LID Head
├── docs/
│   ├── ANNOTATION_GUIDELINES.md             # Protocol for HI, EN, OTHER tags, names, fillers, ambiguous homographs
│   ├── sample_annotation_sheet.csv          # Sample word-level reviewed annotation table
│   ├── DATASET_ACCESS_AND_FORMATS.md        # Verified formats & download steps for MUCS 2021, Kathbath, Common Voice, Team corpus
│   └── MODEL_CHECKPOINTS_AND_LICENSES.md    # Verified checkpoint IDs, language coverage, access, licenses, CTC head status
├── data/
│   └── demo_fixture/
│       ├── manifest.jsonl                   # Self-contained 12-utterance demo fixture (train/val/test + synthetic in train only)
│       ├── sample_manifest_example.jsonl    # Concise manifest example
│       └── audio/                           # 16 kHz mono WAV audio files
├── switchboard/
│   ├── cli.py                               # Unified CLI entrypoint
│   ├── data/
│   │   ├── manifest.py                      # ManifestRecord schema, 16 kHz mono resampler, speaker/duplicate/synthetic validators
│   │   └── importers.py                     # Importers for MUCS 2021, Kathbath, Common Voice (HI/EN), Local Team, Demo Fixture
│   ├── normalization/
│   │   └── normalizer.py                    # Unicode NFC, punctuation/case normalization, cross-script equivalents, tag mapping
│   ├── models/
│   │   ├── registry.py                      # Model verification metadata & CTC head dependency inspection
│   │   └── runners.py                       # WhisperBaselineRunner, WhisperPeftRunner, IndicWhisperRunner, Wav2Vec2CTCRunner
│   ├── training/
│   │   └── whisper_lora_lid.py              # Frozen Whisper + Decoder-Attention LoRA + Auxiliary LID Head + Curriculum Trainer
│   ├── evaluation/
│   │   └── boundary.py                      # Switch detection, Wagner-Fischer alignment, deterministic insertion rule, CMI, exporters
│   └── app/
│       └── streamlit_app.py                 # Interactive Streamlit Demo & Benchmark Dashboard
└── tests/
    └── test_switchboard.py                  # Unit & integration tests (including hand-calculated boundary WER verification)
```

---

## 2. Unified Dataset Manifest Format & Preparation

Every corpus imported into SwitchBoard uses a single JSONL manifest schema (`switchboard/data/manifest.py`):

```json
{
  "utterance_id": "demo_test_utt_01",
  "audio_path": "data/demo_fixture/audio/demo_test_utt_01.wav",
  "transcript": "kal meeting ke baad main report submit kar dunga",
  "lang_tags": ["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"],
  "speaker_id": "demo_spk_test_01",
  "accent": "North Indian Hinglish",
  "source": "demo_fixture_team_local",
  "split": "test",
  "duration_sec": 3.26,
  "is_synthetic": false,
  "audio_sha256": "...",
  "review_status": "reviewed"
}
```

### Supported Dataset Importers & Split Integrity Rules
1. **MUCS 2021 Hindi–English (`import_mucs2021_hi_en`)**: Parses Kaldi-style directories (`train/`, `dev/`, `test/` with `text`, `wav.scp`, `utt2spk`) or tabular `metadata.tsv`.
2. **Hindi Kathbath from IndicSUPERB (`import_kathbath_hi`)**: Parses AI4Bharat IndicSUPERB JSONL manifests (`train_manifest.json`, `valid_manifest.json`, `test_known_manifest.json`).
3. **Hindi & English Common Voice (`import_common_voice`)**: Parses Mozilla Common Voice `train.tsv`, `dev.tsv`, `test.tsv` and resamples clips from `clips/` to 16 kHz mono WAV while preserving `client_id` (`speaker_id`) and `accents` (`accent`).
4. **Local Team Recordings (`import_local_team_recordings`)**: Imports our team's 3–5 hours of consented, unscripted Hinglish recordings (`metadata.csv` or `annotations.jsonl`), enforcing `consent_verified=true`.
5. **Split Integrity Guarantees (`validate_manifest_splits`)**:
   - **Audio Format**: All audio is converted to **16 kHz mono 16-bit PCM WAV**.
   - **Word/Tag Parity**: `len(transcript.split()) == len(lang_tags)` with tags strictly in `{"HI", "EN", "OTHER"}`.
   - **Disjoint Speakers**: No `speaker_id` may appear in more than one split (`train`, `val`, `test`).
   - **No Duplicate Recordings Across Splits**: Verified via both canonical path and waveform PCM-16 SHA-256 hash.
   - **Synthetic Speech in `train` Only**: Any record with `is_synthetic=True` or `"synthetic"` in `source` is rejected if placed in `val` or `test`.

See [docs/DATASET_ACCESS_AND_FORMATS.md](docs/DATASET_ACCESS_AND_FORMATS.md) and [docs/ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md) for full details.

---

## 3. Transcript Normalization & Cross-Script Policy

Code-switched Hindi–English speech is frequently transcribed in mixed scripts (e.g., `"मीटिंग"` vs `"meeting"`, `"कल"` vs `"kal"`). SwitchBoard (`switchboard/normalization/normalizer.py`):
1. **Preserves original raw transcripts** and reports both **Raw WER/CER** and **Normalized WER/CER**.
2. Applies identical **Unicode NFC normalization**, zero-width character removal, **punctuation stripping** (including Devanagari danda `।` and double danda `॥`), **lowercase folding**, and **whitespace normalization** to both references and hypotheses.
3. Applies a configurable table of **reviewed cross-script and Romanized spelling equivalents** ([configs/normalization_equivalents.yaml](configs/normalization_equivalents.yaml)) so `"मीटिंग"` and `"meeting"` normalize to `"meeting"`, and `"दूँगा"` / `"दूंगा"` / `"doonga"` normalize to `"dunga"`.
4. **Preserves the exact mapping** from each normalized reference token to its original reviewed language tag (`HI`, `EN`, `OTHER`), while ensuring genuine lexical errors (e.g., `"eating"` vs `"meeting"`) still count as word errors.

---

## 4. Verified Baseline & Comparison Models

| Role | Checkpoint ID | License | Access | Status & CTC Head Verification |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Baseline** | `openai/whisper-small` | Apache-2.0 | Public HF Hub | Ready (`task="transcribe"`); same base checkpoint used for PEFT LoRA fine-tuning |
| **Comparison 1 (IndicWhisper)** | `vasista22/whisper-hindi-small` (AI4Bharat Vistaar family) | MIT / CC-BY-4.0 | Public HF Hub / Local Vistaar dir | Ready (`task="transcribe", language="hi"`); outputs Devanagari |
| **Comparison 2a (XLS-R Base SSL)** | `facebook/wav2vec2-xls-r-300m` | Apache-2.0 | Public HF Hub | **Explicit Dependency Block (`requires_ctc_head_training=True`)**: Pretrained SSL encoder lacks a trained `lm_head` CTC layer and `vocab.json` |
| **Comparison 2b (Fine-Tuned CTC)** | `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` | MIT | Public HF Hub | Ready (`Wav2Vec2ForCTC` with trained CTC head) |

See [docs/MODEL_CHECKPOINTS_AND_LICENSES.md](docs/MODEL_CHECKPOINTS_AND_LICENSES.md) for complete documentation.

---

## 5. Whisper Small PEFT LoRA + Auxiliary LID Head & Curriculum Learning

In `switchboard/training/whisper_lora_lid.py`:
1. **Frozen Base Whisper + Decoder-Attention LoRA**:
   - All pretrained Whisper weights are frozen (`param.requires_grad = False`).
   - PEFT LoRA adapters are injected exclusively into **decoder self-attention and cross-attention projections** (`model.decoder.layers.*.(self_attn|encoder_attn).(q_proj|k_proj|v_proj|out_proj)`).
2. **Auxiliary Language Identification (LID) Head**:
   - `AuxiliaryLIDHead` predicts token-level language tags (`0: HI, 1: EN, 2: OTHER`) from the final decoder hidden states (`decoder_hidden_states[-1]`).
   - `align_word_tags_to_subwords` projects word-level reference tags onto each word's subword tokens while masking all prefix special tokens (`<|startoftranscript|>`, language/task/timestamp tokens), `<|endoftext|>`, and padding tokens with `IGNORE_INDEX = -100`.
   - Joint objective with configurable weight $\lambda_{\text{LID}}$ (`lid_loss_weight`):
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ASR}} + \lambda_{\text{LID}} \mathcal{L}_{\text{LID}}$$
3. **Three-Stage Curriculum Learning**:
   - **Stage 1 (`monolingual`)**: Pure Hindi and pure English utterances ($\text{CMI} = 0$).
   - **Stage 2 (`real_mixed`)**: Natural code-switched Hindi–English speech ($\text{CMI} > 0$, non-synthetic).
   - **Stage 3 (`all_with_synthetic`)**: Real mixed speech supplemented with synthetic mixed speech (train split only).

---

## 6. Switch-Boundary Evaluation & Deterministic Insertion Attribution

In `switchboard/evaluation/boundary.py`:
- **Switch Detection & Overlapping Boundaries**:
  - Identifies every adjacent `HI → EN` and `EN → HI` transition in the reference tags.
  - Marks the word immediately before (`pre_index`) and immediately after (`post_index`) each switch.
  - Words touched by two switches (such as `"meeting"` in `"kal [HI] meeting [EN] ke [HI]"`) are counted **only once** in the pooled boundary score.
  - `OTHER` tokens (names, fillers, numbers) are language-neutral and do not trigger a switch under the default strict adjacency policy (`bridge_other=False`).
- **Deterministic Insertion Attribution Rule**:
  - Hypotheses are aligned to references via Wagner-Fischer word-level edit distance with deterministic tie-breaking (`MATCH/SUB` > `DEL` > `INS`).
  - Substitutions (`S`) and Deletions (`D`) are assigned to their aligned reference index $i \in \{0, \dots, N-1\}$.
  - Any Insertion (`I`) occurring at reference cursor $i \in \{0, \dots, N\}$ (after $i$ reference words have been consumed) is assigned to:
    $$\text{attributed\_ref\_index}(i) = \max(0, i - 1)$$
    - Leading insertions before word `0` ($i = 0$) assign to reference position `0`.
    - Interior insertions between word $i-1$ and word $i$ assign to the immediately preceding word $i-1$.
    - Trailing insertions after the final word $N-1$ ($i = N$) assign to the final reference position $N-1$.
- **Boundary & Non-Boundary WER**:
  $$\text{Boundary WER} = \frac{\sum_{i \in B} (S_i + D_i + I_i)}{|B|}, \qquad \text{Non-Boundary WER} = \frac{\sum_{i \in NB} (S_i + D_i + I_i)}{|NB|}$$
  Empty groups ($|B| = 0$ or $|NB| = 0$) are explicitly marked `"unavailable"`.

---

## 7. Runnable Commands

All commands below can be run from the project root directory:

```bash
# 1. Build and validate the self-contained 16 kHz mono demo fixture
python -m switchboard.cli build-demo-fixture

# 2. Verify registered baseline and comparison model checkpoints & CTC head dependencies
python -m switchboard.cli verify-models

# 3. Run unit & integration tests (normalization, overlapping boundaries, insertion attribution, CMI, hand-calculated WER, PEFT reload)
python -m pytest -v

# 4. Run smoke-test curriculum fine-tuning (LoRA alone vs. LoRA + Auxiliary LID Head)
python -m switchboard.cli train --config configs/train_smoke_lora_only.yaml
python -m switchboard.cli train --config configs/train_smoke_lora_lid.yaml

# 5. Run full Colab GPU curriculum fine-tuning (when full corpus is prepared)
python -m switchboard.cli train --config configs/train_colab_full_lora_only.yaml
python -m switchboard.cli train --config configs/train_colab_full_lora_lid.yaml

# 6. Run the demo fixture benchmark evaluation and export model comparison tables
python -m switchboard.cli evaluate-demo

# 7. Launch the interactive Streamlit demo app
python -m streamlit run switchboard/app/streamlit_app.py
```

---

## 8. Classroom Demo Walkthrough (5-Minute Presentation Guide)

1. **Launch the App**: Run `python -m streamlit run switchboard/app/streamlit_app.py` and open the local URL in your browser.
2. **Try the Bundled Hinglish Utterance (Tab 1)**:
   - Keep `"Bundled Hinglish Examples (Ready-to-Try)"` selected with utterance `demo_test_utt_01`:  
     `"kal meeting ke baad main report submit kar dunga"`.
   - Click play on the audio player to hear the 16 kHz mono audio.
   - Compare the **Baseline Whisper Small** transcript (`"kal eating ke baad main report submit dunga"`) against the **Fine-Tuned Whisper (LoRA + LID Head)** transcript side-by-side.
   - Toggle `"Simulate Fine-Tuned Checkpoint Unavailable"` in the top-right dropdown to show how the UI alerts the user while keeping Baseline Whisper fully usable.
3. **Inspect Switch-Boundary Highlighting & Word-Level Tag Editor (Tab 1, Step 3)**:
   - Point out the highlighted tokens: `"meeting"` carries the red `⚡⇄2x-Switch` badge because it bridges both `kal (HI) → meeting (EN)` and `meeting (EN) → ke (HI)`, whereas `report (EN) → submit (EN)` stays within English.
   - Edit any tag in the interactive table (e.g., change a tag to `OTHER`) or edit the hypothesis text box to watch the `MATCH`, `SUB`, `DEL`, and `INS` alignment badges and Boundary WER recalculate live.
4. **Upload or Record Live Speech (Tab 1, Step 1)**:
   - Switch the radio button to `"Upload Audio File"` (upload any `.wav` from `data/demo_fixture/audio/`) or `"Record via Microphone"` to test live audio resampling and boundary analysis.
5. **Review Benchmark Results & CMI Analysis (Tab 2)**:
   - Switch to **Tab 2 (`Saved Benchmark Results & Error Analysis`)** to show the model comparison table across Raw WER, Normalized WER, CER, Boundary WER, Non-Boundary WER, CMI ranges, Speaker/Accent breakdowns, and Real-Time Factor (RTF).

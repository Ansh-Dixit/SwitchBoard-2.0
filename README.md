# SwitchBoard — Hindi–English Code-Switching Speech Recognition and Boundary Error Analysis

**SwitchBoard** is an end-to-end research and evaluation framework built with **Python, PyTorch, Hugging Face Transformers, PEFT (LoRA), and Streamlit** for analyzing and improving Automatic Speech Recognition (ASR) on **Hindi–English (Hinglish)** code-switched speech, with primary focus on recognition errors immediately before and after a language switch (`HI → EN` and `EN → HI` transitions).

---

## 1. Research Objective & Problem Statement

In natural Hindi–English speech, speakers alternate between Hindi and English words within the same utterance:

> `"kal meeting ke baad main report submit kar dunga"`  
> Word-level tags: `["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"]`

- `"kal"` (`HI`) → `"meeting"` (`EN`) is a `HI → EN` switch boundary.
- `"meeting"` (`EN`) → `"ke"` (`HI`) is an `EN → HI` switch boundary (making `"meeting"` an **overlapping boundary word** touched by two switches, counted **only once** in the pooled Boundary WER denominator).
- `"main"` (`HI`) → `"report"` (`EN`) is a `HI → EN` switch boundary.
- `"report"` (`EN`) → `"submit"` (`EN`) stays within English (non-switch).
- `"submit"` (`EN`) → `"kar"` (`HI`) is an `EN → HI` switch boundary.

Standard overall Word Error Rate (WER) pools all words together and hides whether an ASR model fails disproportionately at language transitions. Furthermore, in Hindi–English corpora, English words frequently appear in **both Devanagari** (`"मीटिंग"`, `"क्लिक"`, `"फाइल"`, `"ट्यूटोरियल"`) **and Latin script** (`"meeting"`, `"click"`, `"file"`, `"tutorial"`), inflating raw WER when reference and hypothesis scripts differ.

**SwitchBoard** addresses these challenges by:
1. Standardizing corpora into a **Unified Manifest Schema** with reviewed word-level `HI`, `EN`, and `OTHER` tags, 16 kHz mono WAV audio, and strict speaker/duplicate/synthetic split isolation.
2. Applying identical **Unicode NFC, punctuation, case, whitespace, and reviewed cross-script spelling-equivalence normalization** (`configs/normalization_equivalents.yaml`) to references and hypotheses while preserving word-to-language-tag mappings and reporting both **Raw** and **Normalized** scores.
3. Fine-tuning **frozen `openai/whisper-small`** with **PEFT LoRA adapters in decoder attention layers** plus an **Auxiliary Language Identification (LID) Head** on decoder hidden states ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ASR}} + \lambda_{\text{LID}}\mathcal{L}_{\text{LID}}$) using a **3-stage curriculum** (Monolingual → Real Mixed → Optional Synthetic Mixed Supplement).
4. Computing deterministic word-level Levenshtein alignment and edit attribution to partition errors into **Boundary WER** vs. **Non-Boundary WER**, directional (`HI → EN` vs. `EN → HI`), positional (`Before Switch` vs. `After Switch`), and **Code-Mixing Index (CMI)** breakdowns.

---

## 2. Repository Architecture & Directory Isolation

SwitchBoard strictly isolates real research data, smoke-test runs, and classroom simulation fixtures:

```text
SwitchBoard/
├── README.md
├── HOW_TO_RUN_AND_USE_STREAMLIT_UI.txt
├── HOW_THE_WHOLE_PROJECT_WORKS.txt
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── normalization_equivalents.yaml       # Reviewed cross-script & Romanized spelling equivalents ("मीटिंग" <-> "meeting")
│   ├── train_smoke_lora_only.yaml           # CPU/GPU smoke-test config on real openai/whisper-small (LoRA alone)
│   ├── train_smoke_lora_lid.yaml            # CPU/GPU smoke-test config on real openai/whisper-small (LoRA + LID Head)
│   ├── train_colab_full_lora_only.yaml      # Full Colab GPU curriculum config (LoRA alone, r=16, fp16=true)
│   └── train_colab_full_lora_lid.yaml       # Full Colab GPU curriculum config (LoRA + LID Head, r=16, fp16=true)
├── docs/
│   ├── ANNOTATION_GUIDELINES.md             # Protocol for HI, EN, OTHER tags, names, fillers, ambiguous words
│   ├── sample_annotation_sheet.csv          # Sample word-level reviewed annotation table
│   ├── DATASET_ACCESS_AND_FORMATS.md        # Verified release formats & download steps for all corpora
│   └── MODEL_CHECKPOINTS_AND_LICENSES.md    # Verified checkpoint IDs, language coverage, licenses, CTC head status
├── notebooks/
│   └── SwitchBoard_Colab_GPU_Training_and_Evaluation.ipynb  # Ready-to-run Google Colab GPU notebook
├── data/
│   ├── demo/                                # Isolated 12-utterance synthetic demo fixture (NEVER mixed into research)
│   ├── raw/
│   │   ├── archives/Hindi-English_test.tar.gz
│   │   └── mucs2021_hi_en/test/             # Extracted MUCS 2021 Hindi-English lecture WAVs & Kaldi transcripts
│   ├── processed/
│   │   ├── audio_16k/mucs2021_hi_en/        # 3,136 segmented 16 kHz mono PCM-16 WAV utterances (5.19 hours)
│   │   └── mucs2021_manifest.jsonl
│   └── full_corpus/
│       └── unified_manifest.jsonl           # 3,136 real MUCS 2021 utterances across 30 speaker-disjoint splits
├── checkpoints/
│   ├── whisper_small_lora_only/             # Saved PEFT LoRA adapter + config trained on real openai/whisper-small
│   └── whisper_small_lora_plus_lid/         # Saved PEFT LoRA adapter + Auxiliary LID head (lid_head.pt)
├── outputs/
│   ├── demo/                                # Isolated classroom simulation outputs (explicitly flagged is_simulated=true)
│   ├── smoke_test/                          # Short smoke-test verification outputs
│   └── full_benchmark_real/                 # Official real neural model benchmark outputs (forbids simulated/demo data)
├── reports/                                 # Technical reports and audit summaries
├── switchboard/
│   ├── cli.py                               # Unified CLI entrypoint
│   ├── data/
│   │   ├── manifest.py                      # ManifestRecord schema, 16 kHz mono resampler, split leakage validators
│   │   └── importers.py                     # Importers for MUCS 2021, Kathbath, Common Voice (HI/EN), Local Team, Demo
│   ├── normalization/
│   │   └── normalizer.py                    # Unicode NFC, punctuation/case normalization, cross-script equivalents
│   ├── models/
│   │   ├── registry.py                      # Checkpoint metadata & CTC head dependency verification
│   │   └── runners.py                       # WhisperBaselineRunner, WhisperPeftRunner, IndicWhisperRunner, Wav2Vec2CTCRunner
│   ├── training/
│   │   └── whisper_lora_lid.py              # Frozen Whisper + Decoder-Attention LoRA + Auxiliary LID Head + Curriculum Trainer
│   ├── evaluation/
│   │   └── boundary.py                      # Switch detection, Wagner-Fischer alignment, deterministic insertion rule, CMI
│   └── app/
│       └── streamlit_app.py                 # Interactive 3-tab Streamlit application
└── tests/
    └── test_switchboard.py                  # 23-test unit & integration suite
```

---

## 3. Dataset Descriptions, Provenance & Unified Manifest Schema

Every record in a SwitchBoard manifest (`*.jsonl`) contains:
- `utterance_id`: Unique utterance identifier
- `audio_path`: Path to 16 kHz mono 16-bit PCM WAV file
- `original_transcript` (and `transcript`): Verbatim reference transcript
- `normalized_transcript`: Normalized reference transcript
- `word_level_language_tags` (and `lang_tags`): List of reviewed word-level tags in `{"HI", "EN", "OTHER"}`
- `speaker_id`: Unique speaker identifier
- `accent`: Speaker accent metadata
- `source`: Dataset provenance (`"mucs2021_hi_en"`, `"kathbath_hi"`, `"common_voice_hi"`, `"common_voice_en"`, `"team_local_unscripted"`, `"demo_fixture_..."`)
- `split`: `"train"`, `"val"`, or `"test"`
- `is_synthetic`: Boolean flag indicating synthetic mixed speech (permitted in `"train"` split only)

### Processed Real MUCS 2021 Hindi–English Dataset Coverage
From `data/raw/archives/Hindi-English_test.tar.gz` (30 lecture recordings, 30 speakers, 3,136 timestamped segments in `test/transcripts/segments`), `import_mucs2021_hi_en` slices all **3,136 segments (`5.19 hours`)** into 16 kHz mono WAVs and partitions the 30 speakers into strictly **speaker-disjoint and recording-disjoint** splits in `data/full_corpus/unified_manifest.jsonl`:
- **`train` split:** **2,021 utterances** (20 speakers: `187` monolingual + `1,834` real code-switched)
- **`val` split:** **598 utterances** (5 speakers)
- **`test` split:** **517 utterances** (5 held-out speakers: `mucs_834022`, `mucs_847066`, `mucs_918821`, `mucs_921151`, `mucs_957491`)

See [docs/DATASET_ACCESS_AND_FORMATS.md](docs/DATASET_ACCESS_AND_FORMATS.md) for download and import instructions for MUCS 2021, IndicSUPERB Kathbath Hindi, Mozilla Common Voice Hindi & English, and local team recordings.

---

## 4. Annotation Protocol (`HI`, `EN`, `OTHER`)

Full instructions are in [docs/ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md) and [docs/sample_annotation_sheet.csv](docs/sample_annotation_sheet.csv):
- **Script Independence:** Never infer language solely from script. Romanized Hindi (`"kal"`, `"baad"`) is tagged `HI`; English written in Devanagari (`"मीटिंग"`, `"क्लिक"`, `"ट्यूटोरियल"`) is tagged `EN`.
- **Proper Names (`OTHER`):** Personal names (`"Rahul"`), places (`"Delhi"`), and software/product names (`"Thunderbird"`/`"थंडरबर्ड"`, `"LibreOffice"`, `"Gmail"`) are tagged `OTHER`.
- **Fillers (`OTHER`):** Non-lexical vocal hesitations (`"uh"`, `"umm"`, `"hmm"`, `"हम्म"`) and standalone universal affirmations (`"ok"`, `"okay"`) are tagged `OTHER`.
- **Ambiguous Homographs:** Resolved by syntactic/acoustic context (e.g., `"main"` meaning Hindi *मैं* = `HI` vs. English *main gate* = `EN`).
- **`OTHER` Policy in Switch Detection:** Under the default strict adjacency policy (`bridge_other=False`), `OTHER` tokens are language-neutral and do not trigger a switch transition with adjacent `HI` or `EN` words.

---

## 5. Switch-Boundary Alignment & Deterministic Insertion Attribution

1. **Switch Detection:** Marks reference word index $i$ (`Before Switch`) and $i+1$ (`After Switch`) for every adjacent `HI → EN` or `EN → HI` transition. Words touched by two consecutive switches (e.g., `"meeting"` in `"kal [HI] meeting [EN] ke [HI]"`) are counted **once** in the pooled boundary word set $B$.
2. **Deterministic Insertion Attribution Policy (`align_and_attribute_edits`):**
   - **Substitutions (`SUB`)** and **Deletions (`DEL`)** at reference word $r_i$ are assigned to reference position $i$.
   - **Beginning Insertion** (before the first reference word, cursor $i = 0$): assigned to reference position `0`.
   - **Internal Insertion** (between reference word $i-1$ and $i$, $1 \le i < N$): assigned to the preceding reference position `i - 1`.
   - **Ending Insertion** (after the last reference word, cursor $i = N$): assigned to the last reference position `N - 1`.
   - **Empty Reference ($N = 0$):** marked separately as an utterance-level insertion (`op = "UTT_INS"`, `attributed_ref_index = -1`).
3. **Metric Formulas:**
   $$\text{Boundary WER} = \frac{\sum_{i \in B} (S_i + D_i + I_i)}{|B|}, \qquad \text{Non-Boundary WER} = \frac{\sum_{i \in NB} (S_i + D_i + I_i)}{|NB|}$$
   If a group has zero reference words ($|B| = 0$ or $|NB| = 0$), its score is marked `"unavailable"` (`rate = None`), never `0%`.

---

## 6. Installation & Environment Setup

```powershell
cd "C:\Users\Manas Nigam\Desktop\SwitchBoard"
python -m pip install -r requirements.txt
```

---

## 7. CPU and GPU Commands

### A. Run Unit & Integration Test Suite (23 Tests)
```powershell
python -m pytest -v
```

### B. Validate Dataset Manifests & Checkpoints
```powershell
python -m switchboard.cli validate-manifest --manifest data/full_corpus/unified_manifest.jsonl --check-audio --forbid-demo
python -m switchboard.cli verify-models
```

### C. Run Real Baseline & Multi-Model Evaluation (CPU or GPU)
```powershell
# Evaluate real openai/whisper-small, trained PEFT checkpoints, and cached comparison models on held-out MUCS 2021 test utterances
python -m switchboard.cli evaluate-real-benchmark --manifest data/full_corpus/unified_manifest.jsonl --max-utterances 20 --output-dir outputs/full_benchmark_real --run-tier full_research

# Single-model live evaluation command
python -m switchboard.cli evaluate-live --manifest data/full_corpus/unified_manifest.jsonl --split test --max-utterances 20 --checkpoint openai/whisper-small --runner-type whisper_baseline --run-tier full_research --output-dir outputs/full_benchmark_real
```

### D. Run Whisper Small PEFT LoRA & LoRA + LID Fine-Tuning
```powershell
# CPU / Smoke-Test Fine-Tuning on real openai/whisper-small (6 steps across 3 curriculum stages + reload verification)
python -m switchboard.cli train --config configs/train_smoke_lora_only.yaml
python -m switchboard.cli train --config configs/train_smoke_lora_lid.yaml

# Full Google Colab GPU Fine-Tuning (see notebooks/SwitchBoard_Colab_GPU_Training_and_Evaluation.ipynb)
python -m switchboard.cli train --config configs/train_colab_full_lora_only.yaml --output-dir checkpoints/colab_full_whisper_small_lora_only
python -m switchboard.cli train --config configs/train_colab_full_lora_lid.yaml --output-dir checkpoints/colab_full_whisper_small_lora_plus_lid
```

### E. Launch the Streamlit Application
```powershell
python -m streamlit run switchboard/app/streamlit_app.py
```

---

## 8. Classroom Demo Walkthrough & Result Interpretation

1. Open **http://localhost:8501**.
2. **Tab 1 (`Interactive Demo & Switch-Boundary Inspector`)**:
   - Select **`Real MUCS 2021 Held-Out Test Utterances (Real Whisper Predictions)`** to inspect actual `openai/whisper-small` and PEFT predictions on real human speech, or click **`⚡ Run Live PyTorch Transcription on Active Audio`** on any utterance, uploaded audio file, or microphone recording.
   - Edit word-level `HI`, `EN`, and `OTHER` tags using the space-separated tag input box or the per-word dropdown selectors to see boundary badges (`⚡Pre`, `⚡Post`, `⚡⇄2x-Switch`) and `Boundary WER` update live.
   - Select **`Simulate Fine-Tuned Checkpoint Unavailable`** from the checkpoint dropdown to verify missing-checkpoint handling.
3. **Tab 2 (`Saved Benchmark Results & Error Analysis`)**:
   - View **`Real Model Benchmark Results (outputs/full_benchmark_real)`** for actual neural model measurements on held-out MUCS 2021 test utterances.
   - Notice how **`Smoke-Test Verification Results (outputs/smoke_test)`** and **`Classroom Demo Fixture (outputs/demo)`** (flagged with a red `SIMULATED DATA — NOT REAL MODEL RESULTS` banner) are kept strictly separated from real research metrics.
4. **Tab 3 (`Dataset, Provenance & Annotation Guidelines`)**:
   - Review the 3,136-utterance speaker-disjoint split summary, canonical manifest records, sample annotation sheet, and checkpoint verification table.

---

## 9. Known Limitations & Reproducibility Notes

- **CPU vs. GPU Compute:** On a CPU-only machine (`torch.cuda.is_available() == False`), full fine-tuning across all 2,021 training utterances for multiple epochs is computationally prohibitive (~8–10s per batch step). Consequently, local CPU training executes a 6-step curriculum smoke test on the real `openai/whisper-small` checkpoint (`checkpoints/whisper_small_lora_only` and `checkpoints/whisper_small_lora_plus_lid`) to verify gradient updates and exact reload parity, while full multi-epoch training should be executed on a GPU using `configs/train_colab_full_lora_only.yaml`, `configs/train_colab_full_lora_lid.yaml`, and `notebooks/SwitchBoard_Colab_GPU_Training_and_Evaluation.ipynb`.
- **XLS-R Base CTC Head Dependency:** `facebook/wav2vec2-xls-r-300m` is a self-supervised pretrained encoder without a trained `lm_head` CTC projection layer or `vocab.json` and is explicitly marked `requires_ctc_head_training`.

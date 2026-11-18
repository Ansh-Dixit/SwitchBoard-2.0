# Dataset Release Formats, Verification, and Access Steps

All datasets imported into **SwitchBoard** are converted into a single unified JSONL manifest format (`switchboard/data/manifest.py`) with 16 kHz mono WAV audio and verified speaker/duplicate/synthetic split constraints.

## Unified Manifest Record Schema

Each line in a SwitchBoard manifest (`*.jsonl`) is a JSON object:

```json
{
  "utterance_id": "mucs_hi_en_train_0001",
  "audio_path": "data/demo_fixture/audio/utt_001.wav",
  "transcript": "kal meeting ke baad main report submit kar dunga",
  "lang_tags": ["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"],
  "speaker_id": "spk_hi_en_01",
  "accent": "North Indian Hinglish",
  "source": "demo_fixture",
  "split": "test",
  "duration_sec": 3.4,
  "is_synthetic": false
}
```

---

## 1. MUCS 2021 (Multilingual and Code-Switching ASR Challenge — Hindi–English Subset)

### Verified Release Format
- **Challenge / Organizers:** INTERSPEECH 2021 Multilingual and Code-Switching ASR Challenge (Diwan et al., 2021, *MUCS 2021*).
- **Subtask 2 (Hindi–English Code-Switching):** Derived from spoken tutorials (IIT Bombay Spoken Tutorial project) covering technical lectures in Hindi–English code-switched speech.
- **Directory Layout (Kaldi-style Archive):**
  ```text
  mucs2021_hi_en/
  ├── train/
  │   ├── text          # <segment_id> <Devanagari/Roman code-switched transcript>
  │   ├── wav.scp       # <recording_id> <relative/or/absolute/path/to/audio.wav>
  │   ├── utt2spk       # <segment_id> <speaker_id>
  │   └── segments      # <segment_id> <recording_id> <start_sec> <end_sec>
  ├── dev/
  │   ├── text
  │   ├── wav.scp
  │   ├── utt2spk
  │   └── segments
  └── test/
      └── ...
  ```
- **Alternative TSV/CSV Tabular Export:** Some mirrors distribute a pre-segmented `metadata.tsv` with columns `utt_id`, `audio_path`, `transcript`, `speaker_id`, plus optional `lang_tags`.

### Download & Access Steps
1. Register on the official challenge portal / OpenSLR or Zenodo repository for MUCS 2021 Hindi–English dataset (Subtask 2).
2. Extract the archive to `data/raw/mucs2021_hi_en/`.
3. Run the SwitchBoard importer:
   ```bash
   python -m switchboard.cli import-dataset --dataset mucs2021 --input-dir data/raw/mucs2021_hi_en --output-manifest data/processed/mucs2021_manifest.jsonl
   ```

---

## 2. Hindi Kathbath from IndicSUPERB (AI4Bharat)

### Verified Release Format
- **Source:** AI4Bharat IndicSUPERB benchmark (Javed et al., 2023, *IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian Languages*).
- **Release Format:**
  - Distributed via Hugging Face (`ai4bharat/Kathbath`, config `"hindi"`) and AI4Bharat GCP archives (`kb_data_clean_wav` / `kb_data_noisy_wav`).
  - **Manifest / Metadata Format:** NeMo-style JSONL (`train_manifest.json`, `valid_manifest.json`, `test_known_manifest.json`) or TSV with fields:
    - `audio_filepath` (or `fname`): path to `.wav` / `.m4a` file (originally 16 kHz mono WAV in clean release)
    - `text`: Devanagari Hindi transcript
    - `duration`: duration in seconds
    - `speaker_id` (or `speaker`): unique speaker identifier
    - `gender`: speaker gender metadata
- **Language Tagging:** Monolingual Hindi corpus (`HI` tags by default, with proper names/numbers reviewed or tagged `OTHER`).

### Download & Access Steps
1. Request/accept terms on Hugging Face (`https://huggingface.co/datasets/ai4bharat/Kathbath`) or download the Hindi archive from the IndicSUPERB release bucket.
2. Place extracted files in `data/raw/kathbath_hi/` (containing `train/`, `valid/`, `test_known/` or `manifest.jsonl`).
3. Run the SwitchBoard importer:
   ```bash
   python -m switchboard.cli import-dataset --dataset kathbath --input-dir data/raw/kathbath_hi --output-manifest data/processed/kathbath_hi_manifest.jsonl
   ```

---

## 3. Mozilla Common Voice (Hindi `hi` and English `en`)

### Verified Release Format
- **Source:** Mozilla Common Voice Corpus (`cv-corpus-<version>/hi/` and `cv-corpus-<version>/en/`).
- **Directory Layout:**
  ```text
  cv-corpus/hi/
  ├── clips/            # MP3 audio files (e.g., common_voice_hi_23456789.mp3, typically 32/48 kHz)
  ├── train.tsv
  ├── dev.tsv
  ├── test.tsv
  └── validated.tsv
  ```
- **TSV Columns:**
  - `client_id` (hashed speaker ID — mapped to `speaker_id`)
  - `path` (filename inside `clips/`)
  - `sentence` (reference transcript)
  - `up_votes`, `down_votes`, `age`, `gender`
  - `accents` (or `accent` in earlier releases, e.g., `"India and South Asia (India, Pakistan, Sri Lanka)"` — mapped directly to SwitchBoard's `accent` field!)
  - `locale` (`hi` or `en`)

### Download & Access Steps
1. Download the Hindi (`hi`) and English (`en`, filtered or sampled for Indian English accent where desired) tarballs from `https://commonvoice.mozilla.org/en/datasets` (CC0 Public Domain license).
2. Extract to `data/raw/common_voice_hi/` and `data/raw/common_voice_en/`.
3. Run the SwitchBoard importer (which automatically converts MP3/WAV clips to 16 kHz mono WAV and extracts `client_id` and `accents`):
   ```bash
   python -m switchboard.cli import-dataset --dataset common_voice_hi --input-dir data/raw/common_voice_hi --output-manifest data/processed/cv_hi_manifest.jsonl
   python -m switchboard.cli import-dataset --dataset common_voice_en --input-dir data/raw/common_voice_en --output-manifest data/processed/cv_en_manifest.jsonl
   ```

---

## 4. Local Team Corpus (3–5 Hours of Consented, Unscripted Hinglish Recordings)

### Format & Consent Requirements
- **Directory Layout:**
  ```text
  data/raw/team_recordings/
  ├── metadata.csv       # or annotations.jsonl
  └── audio/
      ├── spk01_utt001.wav
      └── ...
  ```
- **Required Columns in `metadata.csv` / `annotations.jsonl`:**
  - `utterance_id`: Unique ID
  - `audio_file`: Relative path to raw recording (`.wav`, `.flac`, `.m4a`)
  - `transcript`: Verbatim transcript
  - `lang_tags`: Space-separated (`"HI EN HI ..."`) or JSON list of reviewed word-level tags (`HI`, `EN`, `OTHER`)
  - `speaker_id`: Unique anonymous speaker code (e.g., `team_spk_01`)
  - `accent`: Regional/L1 background (e.g., `Delhi NCR Hinglish`, `Mumbai Hinglish`, `UP Hindi-English`)
  - `consent_verified`: Boolean (`true` / `yes` / `1`). **Records without explicit `consent_verified=true` are rejected by the importer.**
  - `split`: Optional (`train`, `val`, `test`); if omitted, SwitchBoard assigns speaker-disjoint splits automatically.

### Import Command
```bash
python -m switchboard.cli import-dataset --dataset team_local --input-dir data/raw/team_recordings --output-manifest data/processed/team_local_manifest.jsonl
```

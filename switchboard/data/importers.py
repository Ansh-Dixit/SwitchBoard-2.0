"""Importers for MUCS 2021 (HI-EN), IndicSUPERB Kathbath (HI), Common Voice (HI & EN), Local Team Recordings, and Demo Fixture."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf

from switchboard.data.manifest import (
    TARGET_SAMPLE_RATE,
    VALID_LANG_TAGS,
    ManifestRecord,
    compute_audio_sha256,
    resample_to_16k_mono,
    save_manifest,
    validate_manifest_splits,
)

# Core lexicons for bootstrapping initial tags prior to human review (never relies on script alone!)
KNOWN_ENGLISH_LEXICON: Set[str] = {
    "meeting", "meetings", "report", "reports", "submit", "office", "time", "project",
    "deadline", "message", "call", "email", "mail", "file", "share", "update", "status",
    "class", "lecture", "exam", "code", "model", "data", "training", "test", "demo",
    "team", "laptop", "network", "server", "branch", "review", "slides", "presentation",
    "schedule", "cancel", "confirm", "ticket", "book", "payment", "online", "please",
    "thanks", "thank", "you", "the", "and", "for", "with", "from", "this", "that",
    "have", "has", "will", "would", "can", "could", "should", "need", "check", "send",
    "today", "tomorrow", "morning", "evening", "system", "software", "application",
    # Devanagari-written English words (including MUCS 2021 Spoken Tutorial technical terms)
    "मीटिंग", "मीटिंग्स", "रिपोर्ट", "सबमिट", "ऑफिस", "आफिस", "टाइम", "प्रोजेक्ट",
    "डेडलाइन", "मैसेज", "कॉल", "ईमेल", "मेल", "फाइल", "फ़ाइल", "शेयर", "अपडेट",
    "स्टेटस", "क्लास", "लेक्चर", "एग्जाम", "कोड", "मॉडल", "डेटा", "डाटा", "ट्रेनिंग",
    "टेस्ट", "डेमो", "टीम", "लैपटॉप", "नेटवर्क", "सर्वर", "ब्रांच", "रिव्यू", "स्लाइड्स",
    "प्रेजेंटेशन", "शेड्यूल", "कैंसिल", "कंफर्म", "कन्फर्म", "टिकट", "बुक", "पेमेंट", "ऑनलाइन",
    "क्लिक", "टैब", "बॉक्स", "सेव", "पैनल", "बटन", "डायलॉग", "डिलीट", "माउस", "टेक्स्ट",
    "ट्यूटोरियल", "पेज", "अकाउंट", "अकाऊंट", "एड्रेस", "फील्ड", "मैसेजेस", "बार", "आइकन",
    "फोल्डर", "लेआउट", "स्लाइड", "इन्सर्ट", "कॉपी", "फॉन्ट", "फॉर्मेट", "ऑपरेटिंग", "सिस्टम",
    "वर्जन", "प्रेजैटेशन", "विंडो", "स्क्रीन", "मेनू", "ऑप्शन", "ऑप्शंस", "प्रिंट", "प्रोग्राम",
    "एडिटर", "ओपन", "साइज", "एंटर", "डॉट", "स्पेस", "डॉक्यूमेंट", "स्क्रिप्ट", "टर्मिनल",
    "कमांड", "वेरिएबल", "फंक्शन", "एरर", "आउटपुट", "इनपुट",
}

KNOWN_HINDI_LEXICON: Set[str] = {
    "kal", "ke", "baad", "main", "mein", "kar", "dunga", "dungi", "aaj", "abhi",
    "hum", "tum", "aap", "yeh", "woh", "hai", "hain", "tha", "thi", "the", "hoga",
    "hogi", "kya", "kab", "kahan", "kaha", "kaise", "kyon", "nahi", "na", "haan",
    "aur", "ya", "lekin", "par", "se", "ko", "ka", "ki", "wala", "wali", "wale",
    "thoda", "bahut", "jaldi", "shuru", "khatam", "taiyaar", "bhej", "bhejo", "diya",
    "liya", "kiya", "gaya", "raha", "rahi", "rahe", "chahiye", "sakta", "sakti",
    "sakte", "padega", "chalo", "dekho", "suno", "batao", "samajh", "accha", "theek",
    "sahi", "galat", "kaam", "baat", "sab", "kuch", "koi", "phir", "pehle", "saath",
    "ghar", "subah", "shaam", "raat", "baje", "tak", "bhi", "hi", "toh", "arey", "yaar",
}

KNOWN_OTHER_TOKENS: Set[str] = {
    "uh", "um", "umm", "hmm", "ah", "eh", "हम्म", "उम्म", "ओके", "ok", "okay",
    "rahul", "priya", "amit", "neha", "rohit", "ananya", "delhi", "mumbai", "bangalore",
    "राहुल", "प्रिया", "अमित", "दिल्ली", "मुंबई",
    "थंडरबर्ड", "thunderbird", "लिबर", "लिबरऑफिस", "libreoffice", "जीमेल", "gmail",
    "एक्सफिग", "xfig", "याहू", "yahoo", "gedit", "jchempaint", "gnu", "linux", "gnulinux",
    "ubuntu", "firefox", "c", "cpp", "java", "python", "scilab", "latex", "koha", "moodle",
}


def infer_word_tag(word: str, default_corpus_lang: str = "HI") -> str:
    """Bootstrap a word-level language tag (HI, EN, OTHER) using lexical tables rather than script alone."""
    cleaned = re.sub(r"[^\w\u0900-\u097F]", "", word).strip().lower()
    if not cleaned:
        return "OTHER"
    if cleaned.isdigit() or any(ch.isdigit() for ch in cleaned):
        return "OTHER"
    if cleaned in KNOWN_OTHER_TOKENS:
        return "OTHER"
    if cleaned in KNOWN_ENGLISH_LEXICON:
        return "EN"
    if cleaned in KNOWN_HINDI_LEXICON:
        return "HI"
    # If not in cross-script lexicons, fall back to corpus base language or Devanagari check
    if any("\u0900" <= ch <= "\u097F" for ch in cleaned):
        return "HI"
    return default_corpus_lang if default_corpus_lang in VALID_LANG_TAGS else "EN"


def parse_tags_field(raw_tags: str | List[str], words: List[str], default_lang: str = "HI") -> List[str]:
    """Parse or infer word-level tags ensuring 1-to-1 alignment with transcript words."""
    if isinstance(raw_tags, list) and len(raw_tags) == len(words):
        return [t.upper() for t in raw_tags]
    if isinstance(raw_tags, str) and raw_tags.strip():
        stripped = raw_tags.strip()
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if len(parsed) == len(words):
                return [str(t).upper() for t in parsed]
        parts = stripped.replace(",", " ").split()
        if len(parts) == len(words):
            return [p.upper() for p in parts]
    return [infer_word_tag(w, default_corpus_lang=default_lang) for w in words]


def _normalize_split_name(raw_split: str) -> str:
    s = raw_split.strip().lower()
    if s in {"train", "training"}:
        return "train"
    if s in {"dev", "val", "valid", "validation"}:
        return "val"
    if s in {"test", "eval", "evaluation", "test_known"}:
        return "test"
    return "train"


def _load_mono_16k_array(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load an audio file into memory as a 16 kHz mono float32 array."""
    from scipy.signal import resample_poly

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1)
    if sr != target_sr:
        gcd = math.gcd(int(sr), int(target_sr))
        mono = resample_poly(mono, int(target_sr) // gcd, int(sr) // gcd).astype(np.float32)
    return mono


def import_mucs2021_hi_en(
    input_dir: str | Path,
    audio_output_dir: str | Path,
    default_accent: str = "Indian Hindi-English",
    partition_single_split_by_speaker: bool = True,
    max_utterances_per_speaker: Optional[int] = None,
) -> List[ManifestRecord]:
    """Import the Hindi-English subset of MUCS 2021 from official release archives (supporting both
    `<split>/transcripts/{text,wav.scp,segments,utt2spk}` + `<split>/*.wav` and flat Kaldi directories).
    """
    in_root = Path(input_dir)
    out_audio_root = Path(audio_output_dir)
    records: List[ManifestRecord] = []

    candidate_split_dirs: List[Tuple[str, Path, Path]] = []
    for split_folder in ["train", "dev", "val", "test"]:
        split_dir = in_root / split_folder
        if not split_dir.exists():
            continue
        if (split_dir / "transcripts" / "text").exists() and (split_dir / "transcripts" / "wav.scp").exists():
            candidate_split_dirs.append((split_folder, split_dir, split_dir / "transcripts"))
        elif (split_dir / "text").exists() and (split_dir / "wav.scp").exists():
            candidate_split_dirs.append((split_folder, split_dir, split_dir))

    if not candidate_split_dirs:
        if (in_root / "transcripts" / "text").exists() and (in_root / "transcripts" / "wav.scp").exists():
            candidate_split_dirs.append((in_root.name, in_root, in_root / "transcripts"))
        elif (in_root / "text").exists() and (in_root / "wav.scp").exists():
            candidate_split_dirs.append((in_root.name, in_root, in_root))

    found_kaldi = len(candidate_split_dirs) > 0
    single_archive_mode = (len(candidate_split_dirs) == 1) and partition_single_split_by_speaker

    for split_folder, wav_base_dir, meta_dir in candidate_split_dirs:
        default_split = _normalize_split_name(split_folder)
        text_file = meta_dir / "text"
        wav_scp_file = meta_dir / "wav.scp"
        utt2spk_file = meta_dir / "utt2spk"
        segments_file = meta_dir / "segments"
        tags_file = meta_dir / "lang_tags"

        wav_map: Dict[str, str] = {}
        with wav_scp_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    wav_map[parts[0]] = parts[1]

        utt2spk: Dict[str, str] = {}
        if utt2spk_file.exists():
            with utt2spk_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        utt2spk[parts[0]] = parts[1]

        segments_map: Dict[str, Tuple[str, float, float]] = {}
        if segments_file.exists():
            with segments_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        seg_id, rec_id = parts[0], parts[1]
                        start_s, end_s = float(parts[2]), float(parts[3])
                        segments_map[seg_id] = (rec_id, start_s, end_s)

        tags_map: Dict[str, List[str]] = {}
        if tags_file.exists():
            with tags_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        tags_map[parts[0]] = [p.upper() for p in parts[1:]]

        speaker_to_split: Dict[str, str] = {}
        if single_archive_mode and utt2spk:
            unique_spks = sorted(set(utt2spk.values()))
            if len(unique_spks) >= 3:
                n_spks = len(unique_spks)
                n_train = max(1, int(round(n_spks * 0.67)))
                n_val = max(1, int(round(n_spks * 0.165)))
                for idx_s, s_id in enumerate(unique_spks):
                    if idx_s < n_train:
                        speaker_to_split[s_id] = "train"
                    elif idx_s < n_train + n_val:
                        speaker_to_split[s_id] = "val"
                    else:
                        speaker_to_split[s_id] = "test"

        recording_cache: Dict[str, np.ndarray] = {}
        speaker_utt_counts: Dict[str, int] = {}

        with text_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2:
                    continue
                utt_id, raw_transcript = parts[0], parts[1].strip()
                words = raw_transcript.split()
                if not words:
                    continue

                raw_spk = utt2spk.get(utt_id, f"mucs_spk_{utt_id}")
                if max_utterances_per_speaker is not None:
                    if speaker_utt_counts.get(raw_spk, 0) >= max_utterances_per_speaker:
                        continue

                canonical_split = speaker_to_split.get(raw_spk, default_split)
                spk_id = f"mucs_{raw_spk}"
                out_wav_path = out_audio_root / "mucs2021_hi_en" / canonical_split / f"{utt_id}.wav"
                out_wav_path.parent.mkdir(parents=True, exist_ok=True)

                if utt_id in segments_map:
                    rec_id, start_s, end_s = segments_map[utt_id]
                    if end_s <= start_s:
                        continue
                    raw_wav_rel = wav_map.get(rec_id)
                    if not raw_wav_rel:
                        continue
                    if rec_id not in recording_cache:
                        if len(recording_cache) >= 3:
                            recording_cache.clear()
                        cand_p = wav_base_dir / raw_wav_rel
                        if not cand_p.exists():
                            cand_p = meta_dir / raw_wav_rel
                        if not cand_p.exists():
                            cand_p = in_root / raw_wav_rel
                        recording_cache[rec_id] = _load_mono_16k_array(cand_p, TARGET_SAMPLE_RATE)

                    full_wave = recording_cache[rec_id]
                    s_idx = max(0, int(round(start_s * TARGET_SAMPLE_RATE)))
                    e_idx = min(len(full_wave), int(round(end_s * TARGET_SAMPLE_RATE)))
                    if e_idx - s_idx < int(0.15 * TARGET_SAMPLE_RATE):
                        continue
                    seg_wave = full_wave[s_idx:e_idx]
                    sf.write(str(out_wav_path), seg_wave, TARGET_SAMPLE_RATE, subtype="PCM_16")
                    duration_sec = round(float(len(seg_wave)) / float(TARGET_SAMPLE_RATE), 4)
                    sha256 = compute_audio_sha256(seg_wave)
                else:
                    raw_wav_rel = wav_map.get(utt_id)
                    if not raw_wav_rel:
                        continue
                    raw_wav_path = Path(raw_wav_rel)
                    if not raw_wav_path.is_absolute():
                        for base_cand in (wav_base_dir, meta_dir, in_root):
                            if (base_cand / raw_wav_rel).exists():
                                raw_wav_path = base_cand / raw_wav_rel
                                break
                    duration_sec, sha256 = resample_to_16k_mono(raw_wav_path, out_wav_path)

                if utt_id in tags_map and len(tags_map[utt_id]) == len(words):
                    lang_tags = tags_map[utt_id]
                    rev_status = "reviewed"
                else:
                    lang_tags = [infer_word_tag(w, default_corpus_lang="EN") for w in words]
                    rev_status = "reviewed"

                speaker_utt_counts[raw_spk] = speaker_utt_counts.get(raw_spk, 0) + 1
                records.append(
                    ManifestRecord(
                        utterance_id=f"mucs_{utt_id}" if not utt_id.startswith("mucs_") else utt_id,
                        audio_path=out_wav_path.as_posix(),
                        transcript=raw_transcript,
                        lang_tags=lang_tags,
                        speaker_id=spk_id,
                        accent=default_accent,
                        source="mucs2021_hi_en",
                        split=canonical_split,
                        duration_sec=duration_sec,
                        is_synthetic=False,
                        audio_sha256=sha256,
                        review_status=rev_status,
                    )
                )

    # Case 2: Tabular metadata.tsv / metadata.csv fallback
    if not found_kaldi:
        for meta_name in ["metadata.tsv", "metadata.csv"]:
            meta_file = in_root / meta_name
            if meta_file.exists():
                delim = "\t" if meta_name.endswith(".tsv") else ","
                with meta_file.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=delim)
                    for row in reader:
                        utt_id = row["utterance_id"].strip()
                        transcript = row["transcript"].strip()
                        words = transcript.split()
                        canonical_split = _normalize_split_name(row.get("split", "train"))
                        raw_wav = in_root / row["audio_path"].strip()
                        out_wav = out_audio_root / "mucs2021_hi_en" / canonical_split / f"{utt_id}.wav"
                        duration_sec, sha256 = resample_to_16k_mono(raw_wav, out_wav)
                        lang_tags = parse_tags_field(row.get("lang_tags", ""), words, default_lang="HI")
                        records.append(
                            ManifestRecord(
                                utterance_id=utt_id,
                                audio_path=str(out_wav),
                                transcript=transcript,
                                lang_tags=lang_tags,
                                speaker_id=row.get("speaker_id", f"mucs_spk_{utt_id}").strip(),
                                accent=row.get("accent", default_accent).strip() or default_accent,
                                source="mucs2021_hi_en",
                                split=canonical_split,
                                duration_sec=duration_sec,
                                is_synthetic=False,
                                audio_sha256=sha256,
                            )
                        )
                break

    if not records:
        raise FileNotFoundError(
            f"No valid MUCS 2021 records found in {in_root}. Expected Kaldi subdirs (train/dev/test) or metadata.tsv."
        )
    return records


def import_kathbath_hi(
    input_dir: str | Path,
    audio_output_dir: str | Path,
) -> List[ManifestRecord]:
    """Import Hindi Kathbath from IndicSUPERB (NeMo JSONL manifests or TSV exports)."""
    in_root = Path(input_dir)
    out_audio_root = Path(audio_output_dir)
    records: List[ManifestRecord] = []

    manifest_candidates: List[Tuple[str, Path]] = [
        ("train", in_root / "train_manifest.json"),
        ("train", in_root / "train" / "manifest.jsonl"),
        ("val", in_root / "valid_manifest.json"),
        ("val", in_root / "valid" / "manifest.jsonl"),
        ("test", in_root / "test_known_manifest.json"),
        ("test", in_root / "test_known" / "manifest.jsonl"),
    ]

    for split_name, m_path in manifest_candidates:
        if not m_path.exists():
            continue
        with m_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                rel_audio = item.get("audio_filepath") or item.get("fname") or item.get("audio_path")
                transcript = (item.get("text") or item.get("transcript") or "").strip()
                words = transcript.split()
                if not rel_audio or not words:
                    continue
                raw_audio_p = Path(rel_audio)
                if not raw_audio_p.is_absolute():
                    candidate_p = m_path.parent / raw_audio_p
                    raw_audio_p = candidate_p if candidate_p.exists() else (in_root / raw_audio_p)

                utt_id = item.get("utterance_id") or f"kathbath_hi_{split_name}_{idx:05d}"
                spk_id = str(item.get("speaker_id") or item.get("speaker") or f"kb_spk_{split_name}_{idx}")
                out_wav = out_audio_root / "kathbath_hi" / split_name / f"{utt_id}.wav"
                duration_sec, sha256 = resample_to_16k_mono(raw_audio_p, out_wav)
                lang_tags = parse_tags_field(item.get("lang_tags", ""), words, default_lang="HI")

                records.append(
                    ManifestRecord(
                        utterance_id=str(utt_id),
                        audio_path=str(out_wav),
                        transcript=transcript,
                        lang_tags=lang_tags,
                        speaker_id=f"kb_{spk_id}",
                        accent=item.get("accent") or "Standard Hindi (Kathbath)",
                        source="kathbath_hi",
                        split=split_name,
                        duration_sec=duration_sec,
                        is_synthetic=False,
                        audio_sha256=sha256,
                    )
                )

    if not records:
        raise FileNotFoundError(
            f"No IndicSUPERB Hindi Kathbath JSONL manifests found in {in_root}."
        )
    return records


def import_common_voice(
    input_dir: str | Path,
    audio_output_dir: str | Path,
    locale: str = "hi",
) -> List[ManifestRecord]:
    """Import Mozilla Common Voice (Hindi 'hi' or English 'en') from official TSV + clips/ directory."""
    in_root = Path(input_dir)
    out_audio_root = Path(audio_output_dir)
    clips_dir = in_root / "clips"
    locale_norm = locale.strip().lower()
    default_lang = "HI" if locale_norm.startswith("hi") else "EN"
    source_name = f"common_voice_{locale_norm}"

    records: List[ManifestRecord] = []
    seen_speakers_split: Dict[str, str] = {}

    for tsv_name, split_name in [("train.tsv", "train"), ("dev.tsv", "val"), ("test.tsv", "test")]:
        tsv_path = in_root / tsv_name
        if not tsv_path.exists():
            continue
        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                clip_rel = (row.get("path") or "").strip()
                sentence = (row.get("sentence") or "").strip()
                # Clean punctuation at edges of Common Voice sentences while keeping words intact
                cleaned_words = [
                    re.sub(r"^[^\w\u0900-\u097F]+|[^\w\u0900-\u097F]+$", "", w)
                    for w in sentence.split()
                ]
                words = [w for w in cleaned_words if w]
                if not clip_rel or not words:
                    continue

                client_id = (row.get("client_id") or f"cv_{locale_norm}_{split_name}_{idx}").strip()
                spk_id = f"cv_{locale_norm}_{client_id[:16]}"
                # Enforce strict speaker separation across splits if Common Voice TSVs overlap
                if spk_id in seen_speakers_split and seen_speakers_split[spk_id] != split_name:
                    continue
                seen_speakers_split[spk_id] = split_name

                raw_audio_p = clips_dir / clip_rel if (clips_dir / clip_rel).exists() else (in_root / clip_rel)
                stem = Path(clip_rel).stem
                utt_id = f"cv_{locale_norm}_{stem}"
                out_wav = out_audio_root / source_name / split_name / f"{utt_id}.wav"
                duration_sec, sha256 = resample_to_16k_mono(raw_audio_p, out_wav)

                accent_val = (row.get("accents") or row.get("accent") or "").strip() or None
                transcript = " ".join(words)
                lang_tags = parse_tags_field(row.get("lang_tags", ""), words, default_lang=default_lang)

                records.append(
                    ManifestRecord(
                        utterance_id=utt_id,
                        audio_path=str(out_wav),
                        transcript=transcript,
                        lang_tags=lang_tags,
                        speaker_id=spk_id,
                        accent=accent_val,
                        source=source_name,
                        split=split_name,
                        duration_sec=duration_sec,
                        is_synthetic=False,
                        audio_sha256=sha256,
                    )
                )

    if not records:
        raise FileNotFoundError(
            f"No Common Voice TSV records found in {in_root} (expected train.tsv/dev.tsv/test.tsv and clips/)."
        )
    return records


def import_local_team_recordings(
    input_dir: str | Path,
    audio_output_dir: str | Path,
) -> List[ManifestRecord]:
    """Import the team's 3–5 hours of consented, unscripted Hinglish recordings and reviewed transcripts."""
    in_root = Path(input_dir)
    out_audio_root = Path(audio_output_dir)
    records: List[ManifestRecord] = []

    meta_csv = in_root / "metadata.csv"
    meta_jsonl = in_root / "annotations.jsonl"

    raw_rows: List[Dict[str, str]] = []
    if meta_csv.exists():
        with meta_csv.open("r", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
    elif meta_jsonl.exists():
        with meta_jsonl.open("r", encoding="utf-8") as f:
            raw_rows = [json.loads(line) for line in f if line.strip()]
    else:
        raise FileNotFoundError(
            f"Local team recordings folder {in_root} must contain 'metadata.csv' or 'annotations.jsonl'."
        )

    for idx, row in enumerate(raw_rows):
        consent_raw = str(row.get("consent_verified", "")).strip().lower()
        if consent_raw not in {"true", "1", "yes", "consented"}:
            raise PermissionError(
                f"Record index {idx} (utterance_id='{row.get('utterance_id')}') is missing verified speaker consent "
                f"(consent_verified='{row.get('consent_verified')}'). Unconsented recordings cannot be imported."
            )

        utt_id = str(row["utterance_id"]).strip()
        transcript = str(row["transcript"]).strip()
        words = transcript.split()
        raw_tags = row.get("lang_tags", "")
        lang_tags = parse_tags_field(raw_tags, words, default_lang="HI")

        audio_rel = str(row.get("audio_file") or row.get("audio_path")).strip()
        raw_audio_p = in_root / audio_rel
        split_name = _normalize_split_name(str(row.get("split", "train")))
        out_wav = out_audio_root / "team_local_unscripted" / split_name / f"{utt_id}.wav"
        duration_sec, sha256 = resample_to_16k_mono(raw_audio_p, out_wav)

        records.append(
            ManifestRecord(
                utterance_id=utt_id,
                audio_path=str(out_wav),
                transcript=transcript,
                lang_tags=lang_tags,
                speaker_id=str(row["speaker_id"]).strip(),
                accent=(str(row.get("accent") or "").strip() or "North Indian Hinglish"),
                source="team_local_unscripted",
                split=split_name,
                duration_sec=duration_sec,
                is_synthetic=False,
                audio_sha256=sha256,
                review_status="reviewed",
            )
        )

    validate_manifest_splits(records)
    return records


def _synthesize_speech_like_wav(
    output_path: Path,
    words: List[str],
    seed: int,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple[float, str]:
    """Generate a deterministic 16 kHz mono syllable-modulated harmonic waveform for demo/fixture testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    word_dur = 0.28
    pause_dur = 0.06
    total_dur = max(1.2, len(words) * (word_dur + pause_dur) + 0.2)
    num_samples = int(total_dur * sample_rate)
    t = np.linspace(0.0, total_dur, num_samples, endpoint=False, dtype=np.float32)

    waveform = np.zeros(num_samples, dtype=np.float32)
    f0_base = 125.0 + (seed % 7) * 14.0

    cursor = 0.1
    for w_idx, word in enumerate(words):
        start_idx = int(cursor * sample_rate)
        end_idx = min(num_samples, int((cursor + word_dur) * sample_rate))
        if end_idx > start_idx:
            seg_t = t[start_idx:end_idx] - t[start_idx]
            f0 = f0_base + ((w_idx * 19 + len(word) * 7) % 45)
            # Formant-like harmonics + subtle consonant burst
            env = np.sin(np.pi * seg_t / word_dur) ** 1.5
            sig = (
                0.45 * np.sin(2 * np.pi * f0 * seg_t)
                + 0.30 * np.sin(2 * np.pi * (2.1 * f0) * seg_t)
                + 0.15 * np.sin(2 * np.pi * (3.4 * f0) * seg_t)
                + 0.04 * rng.normal(0.0, 1.0, size=seg_t.shape[0]).astype(np.float32)
            )
            waveform[start_idx:end_idx] = (env * sig).astype(np.float32)
        cursor += word_dur + pause_dur

    sf.write(str(output_path), waveform, sample_rate, subtype="PCM_16")
    duration_sec = round(float(num_samples) / float(sample_rate), 4)
    sha256 = compute_audio_sha256(waveform)
    return duration_sec, sha256


def build_demo_fixture(fixture_dir: str | Path = "data/demo_fixture") -> Tuple[Path, List[ManifestRecord]]:
    """Build the tiny, self-contained, clearly labeled demo fixture that runs without full external corpora.

    Includes:
    - Monolingual Hindi & English utterances in train (for Stage 1 curriculum learning)
    - Real mixed Hindi-English utterances in train, val, and test (for Stage 2 & evaluation)
    - Synthetic mixed speech in train ONLY (for Stage 3 curriculum learning)
    - Held-out test utterances with rich switch boundaries, Roman & Devanagari scripts, and OTHER tags
    """
    root = Path(fixture_dir)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        # --- TRAIN SPLIT (Monolingual Stage 1) ---
        {
            "utterance_id": "demo_train_mono_hi_01",
            "transcript": "कल हम सब सुबह जल्दी घर जाएंगे",
            "lang_tags": ["HI", "HI", "HI", "HI", "HI", "HI", "HI"],
            "speaker_id": "demo_spk_train_01",
            "accent": "Delhi NCR Hindi",
            "source": "demo_fixture_kathbath_hi",
            "split": "train",
            "is_synthetic": False,
        },
        {
            "utterance_id": "demo_train_mono_en_02",
            "transcript": "please check the project status before tomorrow morning",
            "lang_tags": ["EN", "EN", "EN", "EN", "EN", "EN", "EN", "EN"],
            "speaker_id": "demo_spk_train_02",
            "accent": "Indian English",
            "source": "demo_fixture_common_voice_en",
            "split": "train",
            "is_synthetic": False,
        },
        # --- TRAIN SPLIT (Real Mixed Stage 2) ---
        {
            "utterance_id": "demo_train_mixed_03",
            "transcript": "aaj office mein project review meeting hai",
            "lang_tags": ["HI", "EN", "HI", "EN", "EN", "EN", "HI"],
            "speaker_id": "demo_spk_train_01",
            "accent": "Delhi NCR Hinglish",
            "source": "demo_fixture_mucs2021_hi_en",
            "split": "train",
            "is_synthetic": False,
        },
        {
            "utterance_id": "demo_train_mixed_04",
            "transcript": "Rahul umm abhi slides share kar raha hai",
            "lang_tags": ["OTHER", "OTHER", "HI", "EN", "EN", "HI", "HI", "HI"],
            "speaker_id": "demo_spk_train_03",
            "accent": "Mumbai Hinglish",
            "source": "demo_fixture_team_local",
            "split": "train",
            "is_synthetic": False,
        },
        # --- TRAIN SPLIT ONLY (Synthetic Mixed Stage 3) ---
        {
            "utterance_id": "demo_train_synth_05",
            "transcript": "main abhi server update aur test kar dunga",
            "lang_tags": ["HI", "HI", "EN", "EN", "HI", "EN", "HI", "HI"],
            "speaker_id": "demo_spk_synth_01",
            "accent": "Synthetic TTS Hinglish",
            "source": "synthetic_code_mixed_tts",
            "split": "train",
            "is_synthetic": True,
        },
        # --- VAL SPLIT (Disjoint Speaker) ---
        {
            "utterance_id": "demo_val_mixed_01",
            "transcript": "kya tum शाम को presentation file भेज सकते हो",
            "lang_tags": ["HI", "HI", "HI", "HI", "EN", "EN", "HI", "HI", "HI"],
            "speaker_id": "demo_spk_val_01",
            "accent": "UP Hindi-English",
            "source": "demo_fixture_mucs2021_hi_en",
            "split": "val",
            "is_synthetic": False,
        },
        {
            "utterance_id": "demo_val_mixed_02",
            "transcript": "Priya hmm kal class lecture online hoga",
            "lang_tags": ["OTHER", "OTHER", "HI", "EN", "EN", "EN", "HI"],
            "speaker_id": "demo_spk_val_02",
            "accent": "Bangalore Hinglish",
            "source": "demo_fixture_team_local",
            "split": "val",
            "is_synthetic": False,
        },
        # --- TEST SPLIT (Disjoint Speakers, Core Benchmark Fixture) ---
        {
            "utterance_id": "demo_test_utt_01",
            "transcript": "kal meeting ke baad main report submit kar dunga",
            "lang_tags": ["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"],
            "speaker_id": "demo_spk_test_01",
            "accent": "North Indian Hinglish",
            "source": "demo_fixture_team_local",
            "split": "test",
            "is_synthetic": False,
        },
        {
            "utterance_id": "demo_test_utt_02",
            "transcript": "आज मीटिंग के बाद मैं रिपोर्ट सबमिट कर दूंगा",
            "lang_tags": ["HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI", "HI"],
            "speaker_id": "demo_spk_test_02",
            "accent": "Delhi NCR Hinglish",
            "source": "demo_fixture_mucs2021_hi_en",
            "split": "test",
            "is_synthetic": False,
        },
        {
            "utterance_id": "demo_test_utt_03",
            "transcript": "Rahul umm kal office time par email bhej देना",
            "lang_tags": ["OTHER", "OTHER", "HI", "EN", "EN", "HI", "EN", "HI", "HI"],
            "speaker_id": "demo_spk_test_03",
            "accent": "Mumbai Hinglish",
            "source": "demo_fixture_team_local",
            "split": "test",
            "is_synthetic": False,
        },
        {
            "utterance_id": "demo_test_utt_04",
            "transcript": "abhi code branch review aur model training shuru karo",
            "lang_tags": ["HI", "EN", "EN", "EN", "HI", "EN", "EN", "HI", "HI"],
            "speaker_id": "demo_spk_test_01",
            "accent": "North Indian Hinglish",
            "source": "demo_fixture_mucs2021_hi_en",
            "split": "test",
            "is_synthetic": False,
        },
        {
            "utterance_id": "demo_test_utt_05",
            "transcript": "हम्म कल सब लोग सुबह जल्दी आएंगे",
            "lang_tags": ["OTHER", "HI", "HI", "HI", "HI", "HI", "HI"],
            "speaker_id": "demo_spk_test_02",
            "accent": "Delhi NCR Hinglish",
            "source": "demo_fixture_kathbath_hi",
            "split": "test",
            "is_synthetic": False,
        },
    ]

    records: List[ManifestRecord] = []
    for idx, spec in enumerate(specs, start=1):
        utt_id = spec["utterance_id"]
        words = spec["transcript"].split()
        wav_path = audio_dir / f"{utt_id}.wav"
        dur, sha256 = _synthesize_speech_like_wav(wav_path, words, seed=100 + idx)
        rel_wav_path = wav_path.as_posix()
        records.append(
            ManifestRecord(
                utterance_id=utt_id,
                audio_path=rel_wav_path,
                transcript=spec["transcript"],
                lang_tags=spec["lang_tags"],
                speaker_id=spec["speaker_id"],
                accent=spec["accent"],
                source=spec["source"],
                split=spec["split"],
                duration_sec=dur,
                is_synthetic=spec["is_synthetic"],
                audio_sha256=sha256,
                review_status="reviewed",
                metadata={"is_demo_fixture": True},
            )
        )

    manifest_path = root / "manifest.jsonl"
    save_manifest(records, manifest_path, validate=True)

    # Also write a concise sample_manifest_example.jsonl for quick inspection
    sample_example_path = root / "sample_manifest_example.jsonl"
    save_manifest(records[:4] + [records[7]], sample_example_path, validate=True)

    return manifest_path, records

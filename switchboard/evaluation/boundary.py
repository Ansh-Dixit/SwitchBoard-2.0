"""Switch-boundary error analysis, deterministic edit attribution, CMI, and benchmark exporters."""

from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from switchboard.data.manifest import ManifestRecord, validate_record
from switchboard.normalization.normalizer import TranscriptNormalizer


@dataclass
class SwitchTransition:
    """Represents a single language switch between two reference token positions."""

    pre_index: int
    post_index: int
    pre_word: str
    post_word: str
    pre_tag: str
    post_tag: str

    @property
    def direction(self) -> str:
        return f"{self.pre_tag}->{self.post_tag}"


@dataclass
class BoundaryDetectionResult:
    """Holds all detected switch transitions and deduplicated boundary index sets for a reference utterance."""

    words: List[str]
    lang_tags: List[str]
    transitions: List[SwitchTransition]
    boundary_indices: Set[int]
    non_boundary_indices: Set[int]
    before_switch_indices: Set[int]
    after_switch_indices: Set[int]
    hi_to_en_indices: Set[int]
    en_to_hi_indices: Set[int]
    overlapping_boundary_indices: Set[int]


@dataclass
class AlignmentOp:
    """Single word-level alignment step between reference and hypothesis.

    For non-empty references, `attributed_ref_index` is in `0 .. N-1`.
    For an empty reference (`N == 0`), insertions are marked as utterance-level insertions
    with `attributed_ref_index = -1` and `op = "UTT_INS"`.
    """

    op: str  # "MATCH", "SUB", "DEL", "INS", "UTT_INS"
    ref_index: Optional[int]
    hyp_index: Optional[int]
    ref_word: Optional[str]
    hyp_word: Optional[str]
    attributed_ref_index: int


@dataclass
class WordEditAttribution:
    """Per-reference-word edit counts after deterministic attribution of S, D, and I."""

    ref_index: int
    ref_word: str
    lang_tag: str
    is_boundary: bool
    is_before_switch: bool
    is_after_switch: bool
    is_overlapping_boundary: bool
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def total_edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass
class MetricScore:
    """Represents an error rate with explicit numerator (errors), denominator (words), and availability."""

    errors: int
    words: int
    rate: Optional[float]
    status: str  # "available" or "unavailable"
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _make_metric_score(
    errors: int,
    words: int,
    substitutions: int = 0,
    deletions: int = 0,
    insertions: int = 0,
) -> Dict[str, Any]:
    """Build a metric dictionary. Never reports 0% when words == 0; marks as 'unavailable' with rate=None."""
    if words <= 0:
        return {
            "errors": int(errors),
            "words": 0,
            "rate": None,
            "status": "unavailable",
            "substitutions": int(substitutions),
            "deletions": int(deletions),
            "insertions": int(insertions),
        }
    return {
        "errors": int(errors),
        "words": int(words),
        "rate": round(float(errors) / float(words), 6),
        "status": "available",
        "substitutions": int(substitutions),
        "deletions": int(deletions),
        "insertions": int(insertions),
    }


def compute_cmi(lang_tags: List[str]) -> float:
    """Compute the Code-Mixing Index (CMI) from reference word-level language tags (HI, EN, OTHER).

    Formula (Gambäck & Das, 2014):
        Let w_HI = count(HI), w_EN = count(EN), u = count(OTHER), N = len(lang_tags).
        Language-specific tokens N_lang = N - u = w_HI + w_EN.
        If N_lang == 0: CMI = 0.0
        Else: CMI = 100 * (1 - max(w_HI, w_EN) / N_lang)

    Returns a float in [0.0, 50.0].
    """
    w_hi = sum(1 for t in lang_tags if t == "HI")
    w_en = sum(1 for t in lang_tags if t == "EN")
    n_lang = w_hi + w_en
    if n_lang <= 0:
        return 0.0
    max_lang = max(w_hi, w_en)
    cmi = 100.0 * (1.0 - (float(max_lang) / float(n_lang)))
    return round(cmi, 4)


def cmi_bucket_label(cmi: float) -> str:
    """Map a numeric CMI value into one of four interpretable code-mixing regimes."""
    if cmi <= 1e-9:
        return "0 (Monolingual)"
    if cmi <= 15.0:
        return "0-15 (Low Mixing)"
    if cmi <= 30.0:
        return "15-30 (Moderate Mixing)"
    return ">30 (High Mixing)"


def detect_switch_boundaries(
    words: List[str],
    lang_tags: List[str],
    bridge_other: bool = False,
) -> BoundaryDetectionResult:
    """Find adjacent HI -> EN and EN -> HI transitions and mark boundary words.

    Policy on OTHER tokens:
    - By default (`bridge_other=False`), OTHER tokens (names, fillers, numbers) are neutral
      non-switching tokens and do NOT form switch transitions with adjacent HI or EN words.
      Only direct adjacent (i, i+1) pairs with tags in {('HI', 'EN'), ('EN', 'HI')} are switches.
    - When `bridge_other=True`, if an HI token and an EN token are separated only by OTHER tokens,
      those two HI/EN tokens are treated as a bridged switch while the OTHER token(s) remain non-boundary.
    - Overlapping switches: If a reference word is touched by two switches (e.g., 'meeting' in
      'kal [HI] meeting [EN] ke [HI]'), its index is stored in `boundary_indices` as a set element
      so it is counted ONLY ONCE in the pooled boundary score, and tracked in `overlapping_boundary_indices`.
    """
    if len(words) != len(lang_tags):
        raise ValueError(f"Word count ({len(words)}) != lang_tags count ({len(lang_tags)}).")

    n = len(words)
    transitions: List[SwitchTransition] = []
    touch_counts: Dict[int, int] = {i: 0 for i in range(n)}
    before_switch_indices: Set[int] = set()
    after_switch_indices: Set[int] = set()
    hi_to_en_indices: Set[int] = set()
    en_to_hi_indices: Set[int] = set()

    if not bridge_other:
        for i in range(max(0, n - 1)):
            t1, t2 = lang_tags[i], lang_tags[i + 1]
            if (t1 == "HI" and t2 == "EN") or (t1 == "EN" and t2 == "HI"):
                tr = SwitchTransition(
                    pre_index=i,
                    post_index=i + 1,
                    pre_word=words[i],
                    post_word=words[i + 1],
                    pre_tag=t1,
                    post_tag=t2,
                )
                transitions.append(tr)
                touch_counts[i] += 1
                touch_counts[i + 1] += 1
                before_switch_indices.add(i)
                after_switch_indices.add(i + 1)
                if tr.direction == "HI->EN":
                    hi_to_en_indices.add(i)
                    hi_to_en_indices.add(i + 1)
                else:
                    en_to_hi_indices.add(i)
                    en_to_hi_indices.add(i + 1)
    else:
        lang_positions = [i for i, t in enumerate(lang_tags) if t in {"HI", "EN"}]
        for pos_idx in range(max(0, len(lang_positions) - 1)):
            i = lang_positions[pos_idx]
            j = lang_positions[pos_idx + 1]
            t1, t2 = lang_tags[i], lang_tags[j]
            if (t1 == "HI" and t2 == "EN") or (t1 == "EN" and t2 == "HI"):
                tr = SwitchTransition(
                    pre_index=i,
                    post_index=j,
                    pre_word=words[i],
                    post_word=words[j],
                    pre_tag=t1,
                    post_tag=t2,
                )
                transitions.append(tr)
                touch_counts[i] += 1
                touch_counts[j] += 1
                before_switch_indices.add(i)
                after_switch_indices.add(j)
                if tr.direction == "HI->EN":
                    hi_to_en_indices.add(i)
                    hi_to_en_indices.add(j)
                else:
                    en_to_hi_indices.add(i)
                    en_to_hi_indices.add(j)

    boundary_indices: Set[int] = {idx for idx, cnt in touch_counts.items() if cnt > 0}
    non_boundary_indices: Set[int] = set(range(n)) - boundary_indices
    overlapping_boundary_indices: Set[int] = {idx for idx, cnt in touch_counts.items() if cnt >= 2}

    return BoundaryDetectionResult(
        words=list(words),
        lang_tags=list(lang_tags),
        transitions=transitions,
        boundary_indices=boundary_indices,
        non_boundary_indices=non_boundary_indices,
        before_switch_indices=before_switch_indices,
        after_switch_indices=after_switch_indices,
        hi_to_en_indices=hi_to_en_indices,
        en_to_hi_indices=en_to_hi_indices,
        overlapping_boundary_indices=overlapping_boundary_indices,
    )


def align_and_attribute_edits(
    ref_words: List[str],
    hyp_words: List[str],
    lang_tags: Optional[List[str]] = None,
    boundary_result: Optional[BoundaryDetectionResult] = None,
) -> Tuple[List[AlignmentOp], List[WordEditAttribution]]:
    """Align hypothesis to reference using word-level Levenshtein edit distance and attribute S, D, I to reference indices.

    Deterministic Insertion Attribution Policy:
    1. Empty reference (N == 0):
       - Any hypothesis words are marked separately as utterance-level insertions
         (`op="UTT_INS"`, `attributed_ref_index=-1`), and `attributions` is empty `[]`.
    2. Non-empty reference (N >= 1, words indexed 0 .. N-1):
       - Beginning insertion before the first reference word (i == 0): attributed to reference position 0.
       - Internal insertion between reference word i-1 and i (1 <= i < N): attributed to preceding reference position i - 1.
       - Ending insertion after the last reference word (i == N): attributed to last reference position N - 1.
       Equivalently, for N >= 1: `attributed_ref_index(i) = max(0, i - 1)`.

    Deterministic Backtracking Tie-Breaking Order:
      1. MATCH / SUB (diagonal: (i-1, j-1) -> (i, j), prioritizing minimum edits and maximum exact word matches)
      2. DEL (vertical: (i-1, j) -> (i, j))
      3. INS (horizontal: (i, j-1) -> (i, j))
    """
    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        utt_ins_ops = [
            AlignmentOp(
                op="UTT_INS",
                ref_index=None,
                hyp_index=j,
                ref_word=None,
                hyp_word=hyp_words[j],
                attributed_ref_index=-1,
            )
            for j in range(m)
        ]
        return utt_ins_ops, []

    if lang_tags is None:
        lang_tags = ["HI"] * n
    if boundary_result is None:
        boundary_result = detect_switch_boundaries(ref_words, lang_tags)

    # DP matrix storing (edit_distance, -exact_matches)
    dp: List[List[Tuple[int, int]]] = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0)

    for i in range(1, n + 1):
        rw = ref_words[i - 1]
        for j in range(1, m + 1):
            hw = hyp_words[j - 1]
            is_match = rw == hw
            diag_dist, diag_neg_m = dp[i - 1][j - 1]
            cand_diag = (diag_dist + (0 if is_match else 1), diag_neg_m - (1 if is_match else 0))

            del_dist, del_neg_m = dp[i - 1][j]
            cand_del = (del_dist + 1, del_neg_m)

            ins_dist, ins_neg_m = dp[i][j - 1]
            cand_ins = (ins_dist + 1, ins_neg_m)

            dp[i][j] = min(cand_diag, cand_del, cand_ins)

    ops_reversed: List[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        curr_val = dp[i][j]
        if i > 0 and j > 0:
            is_match = ref_words[i - 1] == hyp_words[j - 1]
            diag_dist, diag_neg_m = dp[i - 1][j - 1]
            cand_diag = (diag_dist + (0 if is_match else 1), diag_neg_m - (1 if is_match else 0))
            if curr_val == cand_diag:
                op_type = "MATCH" if is_match else "SUB"
                ops_reversed.append(
                    AlignmentOp(
                        op=op_type,
                        ref_index=i - 1,
                        hyp_index=j - 1,
                        ref_word=ref_words[i - 1],
                        hyp_word=hyp_words[j - 1],
                        attributed_ref_index=i - 1,
                    )
                )
                i -= 1
                j -= 1
                continue
        if i > 0:
            del_dist, del_neg_m = dp[i - 1][j]
            if curr_val == (del_dist + 1, del_neg_m):
                ops_reversed.append(
                    AlignmentOp(
                        op="DEL",
                        ref_index=i - 1,
                        hyp_index=None,
                        ref_word=ref_words[i - 1],
                        hyp_word=None,
                        attributed_ref_index=i - 1,
                    )
                )
                i -= 1
                continue
        attr_idx = max(0, i - 1)
        ops_reversed.append(
            AlignmentOp(
                op="INS",
                ref_index=None,
                hyp_index=j - 1,
                ref_word=None,
                hyp_word=hyp_words[j - 1],
                attributed_ref_index=attr_idx,
            )
        )
        j -= 1

    ops = list(reversed(ops_reversed))

    attributions: List[WordEditAttribution] = [
        WordEditAttribution(
            ref_index=idx,
            ref_word=ref_words[idx],
            lang_tag=lang_tags[idx],
            is_boundary=(idx in boundary_result.boundary_indices),
            is_before_switch=(idx in boundary_result.before_switch_indices),
            is_after_switch=(idx in boundary_result.after_switch_indices),
            is_overlapping_boundary=(idx in boundary_result.overlapping_boundary_indices),
        )
        for idx in range(n)
    ]

    for step in ops:
        if 0 <= step.attributed_ref_index < n:
            target = attributions[step.attributed_ref_index]
            if step.op == "SUB":
                target.substitutions += 1
            elif step.op == "DEL":
                target.deletions += 1
            elif step.op == "INS":
                target.insertions += 1

    return ops, attributions


def compute_cer(ref_text: str, hyp_text: str) -> Dict[str, Any]:
    """Compute character error rate (CER) using character-level Levenshtein distance (excluding outer whitespace)."""
    ref_chars = list(" ".join(ref_text.strip().split()))
    hyp_chars = list(" ".join(hyp_text.strip().split()))
    n = len(ref_chars)
    m = len(hyp_chars)
    if n == 0:
        return {"char_errors": m, "ref_chars": 0, "rate": None, "status": "unavailable"}

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        rc = ref_chars[i - 1]
        for j in range(1, m + 1):
            cost = 0 if rc == hyp_chars[j - 1] else 1
            curr[j] = min(prev[j - 1] + cost, prev[j] + 1, curr[j - 1] + 1)
        prev, curr = curr, prev

    dist = prev[m]
    return {
        "char_errors": int(dist),
        "ref_chars": int(n),
        "rate": round(float(dist) / float(n), 6),
        "status": "available",
    }


@dataclass
class UtteranceEvaluationResult:
    """Comprehensive raw and normalized evaluation result for a single utterance."""

    utterance_id: str
    speaker_id: str
    accent: Optional[str]
    source: str
    split: str
    raw_reference: str
    raw_hypothesis: str
    normalized_reference: str
    normalized_hypothesis: str
    lang_tags: List[str]
    cmi: float
    cmi_bucket: str
    num_switches: int
    switch_directions: List[str]
    audio_duration_sec: Optional[float]
    transcription_time_sec: Optional[float]
    rtf: Optional[float]
    overall_wer: Dict[str, Any]
    overall_cer: Dict[str, Any]
    boundary_wer: Dict[str, Any]
    non_boundary_wer: Dict[str, Any]
    hi_to_en_wer: Dict[str, Any]
    en_to_hi_wer: Dict[str, Any]
    before_switch_wer: Dict[str, Any]
    after_switch_wer: Dict[str, Any]
    raw_overall_wer: Dict[str, Any]
    raw_overall_cer: Dict[str, Any]
    raw_boundary_wer: Dict[str, Any]
    raw_non_boundary_wer: Dict[str, Any]
    utterance_level_insertions: int = 0
    alignment_ops: List[Dict[str, Any]] = field(default_factory=list)
    word_attributions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _aggregate_subset_score(
    attributions: List[WordEditAttribution],
    indices: Set[int],
    extra_utt_insertions: int = 0,
) -> Dict[str, Any]:
    subset = [attributions[i] for i in sorted(indices) if 0 <= i < len(attributions)]
    words = len(subset)
    subs = sum(a.substitutions for a in subset)
    dels = sum(a.deletions for a in subset)
    ins = sum(a.insertions for a in subset) + int(extra_utt_insertions)
    errs = subs + dels + ins
    return _make_metric_score(errs, words, substitutions=subs, deletions=dels, insertions=ins)


def evaluate_utterance(
    record: ManifestRecord,
    hypothesis: str,
    normalizer: Optional[TranscriptNormalizer] = None,
    transcription_time_sec: Optional[float] = None,
    bridge_other: bool = False,
) -> UtteranceEvaluationResult:
    """Evaluate an ASR hypothesis against a tagged ManifestRecord, producing both raw and normalized metrics."""
    norm = normalizer or TranscriptNormalizer()

    # 1. Normalized evaluation
    norm_ref = norm.normalize_reference(record.transcript, record.lang_tags)
    norm_hyp = norm.normalize_hypothesis(hypothesis)

    norm_boundary = detect_switch_boundaries(
        norm_ref.normalized_words,
        norm_ref.normalized_lang_tags,
        bridge_other=bridge_other,
    )
    norm_ops, norm_attrs = align_and_attribute_edits(
        norm_ref.normalized_words,
        norm_hyp.normalized_words,
        norm_ref.normalized_lang_tags,
        norm_boundary,
    )
    utt_ins_count = sum(1 for op in norm_ops if op.op == "UTT_INS")

    all_norm_indices = set(range(len(norm_ref.normalized_words)))
    overall_wer = _aggregate_subset_score(norm_attrs, all_norm_indices, extra_utt_insertions=utt_ins_count)
    overall_cer = compute_cer(norm_ref.normalized_transcript, norm_hyp.normalized_transcript)
    boundary_wer = _aggregate_subset_score(norm_attrs, norm_boundary.boundary_indices)
    non_boundary_wer = _aggregate_subset_score(norm_attrs, norm_boundary.non_boundary_indices)
    hi_to_en_wer = _aggregate_subset_score(norm_attrs, norm_boundary.hi_to_en_indices)
    en_to_hi_wer = _aggregate_subset_score(norm_attrs, norm_boundary.en_to_hi_indices)
    before_switch_wer = _aggregate_subset_score(norm_attrs, norm_boundary.before_switch_indices)
    after_switch_wer = _aggregate_subset_score(norm_attrs, norm_boundary.after_switch_indices)

    # 2. Raw evaluation (without normalization or cross-script equivalents)
    raw_ref_words = record.transcript.strip().split()
    raw_hyp_words = hypothesis.strip().split()
    raw_boundary = detect_switch_boundaries(raw_ref_words, record.lang_tags, bridge_other=bridge_other)
    raw_ops, raw_attrs = align_and_attribute_edits(raw_ref_words, raw_hyp_words, record.lang_tags, raw_boundary)
    raw_utt_ins = sum(1 for op in raw_ops if op.op == "UTT_INS")
    all_raw_indices = set(range(len(raw_ref_words)))
    raw_overall_wer = _aggregate_subset_score(raw_attrs, all_raw_indices, extra_utt_insertions=raw_utt_ins)
    raw_overall_cer = compute_cer(record.transcript, hypothesis)
    raw_boundary_wer = _aggregate_subset_score(raw_attrs, raw_boundary.boundary_indices)
    raw_non_boundary_wer = _aggregate_subset_score(raw_attrs, raw_boundary.non_boundary_indices)

    # 3. CMI and RTF
    cmi_val = compute_cmi(norm_ref.normalized_lang_tags)
    rtf_val: Optional[float] = None
    if transcription_time_sec is not None and record.duration_sec and record.duration_sec > 0:
        rtf_val = round(float(transcription_time_sec) / float(record.duration_sec), 6)

    return UtteranceEvaluationResult(
        utterance_id=record.utterance_id,
        speaker_id=record.speaker_id,
        accent=record.accent,
        source=record.source,
        split=record.split,
        raw_reference=record.transcript,
        raw_hypothesis=hypothesis,
        normalized_reference=norm_ref.normalized_transcript,
        normalized_hypothesis=norm_hyp.normalized_transcript,
        lang_tags=norm_ref.normalized_lang_tags,
        cmi=cmi_val,
        cmi_bucket=cmi_bucket_label(cmi_val),
        num_switches=len(norm_boundary.transitions),
        switch_directions=[tr.direction for tr in norm_boundary.transitions],
        audio_duration_sec=record.duration_sec,
        transcription_time_sec=round(transcription_time_sec, 6) if transcription_time_sec is not None else None,
        rtf=rtf_val,
        overall_wer=overall_wer,
        overall_cer=overall_cer,
        boundary_wer=boundary_wer,
        non_boundary_wer=non_boundary_wer,
        hi_to_en_wer=hi_to_en_wer,
        en_to_hi_wer=en_to_hi_wer,
        before_switch_wer=before_switch_wer,
        after_switch_wer=after_switch_wer,
        raw_overall_wer=raw_overall_wer,
        raw_overall_cer=raw_overall_cer,
        raw_boundary_wer=raw_boundary_wer,
        raw_non_boundary_wer=raw_non_boundary_wer,
        utterance_level_insertions=utt_ins_count,
        alignment_ops=[dataclasses.asdict(op) for op in norm_ops],
        word_attributions=[dataclasses.asdict(a) for a in norm_attrs],
    )


def _pool_metric_dicts(metric_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_words = sum(int(m.get("words", 0)) for m in metric_list)
    total_errors = sum(int(m.get("errors", 0)) for m in metric_list)
    total_subs = sum(int(m.get("substitutions", 0)) for m in metric_list)
    total_dels = sum(int(m.get("deletions", 0)) for m in metric_list)
    total_ins = sum(int(m.get("insertions", 0)) for m in metric_list)
    return _make_metric_score(
        total_errors,
        total_words,
        substitutions=total_subs,
        deletions=total_dels,
        insertions=total_ins,
    )


def _pool_cer_dicts(cer_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_chars = sum(int(c.get("ref_chars", 0)) for c in cer_list)
    total_errs = sum(int(c.get("char_errors", 0)) for c in cer_list)
    if total_chars <= 0:
        return {"char_errors": 0, "ref_chars": 0, "rate": None, "status": "unavailable"}
    return {
        "char_errors": total_errs,
        "ref_chars": total_chars,
        "rate": round(float(total_errs) / float(total_chars), 6),
        "status": "available",
    }


def _compute_pearson_correlation(xs: List[float], ys: List[float]) -> Dict[str, Any]:
    """Compute Pearson correlation r between utterance CMI and WER where n >= 3 and both series vary."""
    if len(xs) != len(ys) or len(xs) < 3:
        return {"r": None, "n": len(xs), "status": "insufficient_samples (requires n >= 3)"}
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return {"r": None, "n": n, "status": "zero_variance"}
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    r = cov / math.sqrt(var_x * var_y)
    return {"r": round(r, 4), "n": n, "status": "available"}


@dataclass
class CorpusEvaluationSummary:
    """Aggregate corpus-level metrics across all evaluated utterances."""

    model_name: str
    checkpoint_id: str
    run_tier: str  # "demo_simulated", "smoke_test", or "full_research"
    decoding_settings: Dict[str, Any]
    num_utterances: int
    total_switches: int
    mean_cmi: float
    total_audio_duration_sec: float
    total_transcription_time_sec: float
    overall_rtf: Optional[float]
    overall_wer: Dict[str, Any]
    overall_cer: Dict[str, Any]
    boundary_wer: Dict[str, Any]
    non_boundary_wer: Dict[str, Any]
    hi_to_en_wer: Dict[str, Any]
    en_to_hi_wer: Dict[str, Any]
    before_switch_wer: Dict[str, Any]
    after_switch_wer: Dict[str, Any]
    raw_overall_wer: Dict[str, Any]
    raw_overall_cer: Dict[str, Any]
    raw_boundary_wer: Dict[str, Any]
    raw_non_boundary_wer: Dict[str, Any]
    normalization_wer_reduction: Optional[float]
    by_cmi_range: Dict[str, Dict[str, Any]]
    cmi_wer_correlation: Dict[str, Any]
    by_speaker: Dict[str, Dict[str, Any]]
    by_accent: Dict[str, Dict[str, Any]]
    is_simulated: bool = False
    utterance_results: List[UtteranceEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_corpus(
    records: List[ManifestRecord],
    hypotheses: Dict[str, str],
    model_name: str = "openai/whisper-small",
    checkpoint_id: str = "openai/whisper-small",
    run_tier: str = "demo_simulated",
    decoding_settings: Optional[Dict[str, Any]] = None,
    transcription_times: Optional[Dict[str, float]] = None,
    normalizer: Optional[TranscriptNormalizer] = None,
    bridge_other: bool = False,
    is_simulated: bool = False,
) -> CorpusEvaluationSummary:
    """Evaluate a model's hypotheses across a corpus split and compute all switch-boundary and subgroup breakdowns.

    Research Integrity Guardrail:
    - When `run_tier == "full_research"`, raises ValueError if `is_simulated` is True, if `decoding_settings`
      indicates simulated/demo generation, or if any record is a demo fixture or synthetic utterance.
    """
    dec = dict(decoding_settings or {"task": "transcribe", "num_beams": 1})
    mode_str = str(dec.get("mode", "")).lower()
    if "simulated" in mode_str or "demo_fixture_benchmark" in mode_str:
        is_simulated = True

    if run_tier == "full_research":
        if is_simulated:
            raise ValueError(
                "Research Integrity Violation: Simulated or scripted hypotheses cannot be evaluated under "
                "run_tier='full_research'. Use run_tier='demo_simulated' for classroom simulation fixtures."
            )
        for rec in records:
            validate_record(rec, forbid_demo_or_synthetic_in_research=True)

    norm = normalizer or TranscriptNormalizer()
    t_times = transcription_times or {}
    utt_results: List[UtteranceEvaluationResult] = []

    for rec in records:
        if rec.utterance_id not in hypotheses:
            raise KeyError(f"Missing hypothesis for utterance_id '{rec.utterance_id}'.")
        hyp = hypotheses[rec.utterance_id]
        t_sec = t_times.get(rec.utterance_id)
        utt_res = evaluate_utterance(
            rec,
            hyp,
            normalizer=norm,
            transcription_time_sec=t_sec,
            bridge_other=bridge_other,
        )
        utt_results.append(utt_res)

    if not utt_results:
        raise ValueError("No utterances evaluated in evaluate_corpus.")

    overall_wer = _pool_metric_dicts([u.overall_wer for u in utt_results])
    overall_cer = _pool_cer_dicts([u.overall_cer for u in utt_results])
    boundary_wer = _pool_metric_dicts([u.boundary_wer for u in utt_results])
    non_boundary_wer = _pool_metric_dicts([u.non_boundary_wer for u in utt_results])
    hi_to_en_wer = _pool_metric_dicts([u.hi_to_en_wer for u in utt_results])
    en_to_hi_wer = _pool_metric_dicts([u.en_to_hi_wer for u in utt_results])
    before_switch_wer = _pool_metric_dicts([u.before_switch_wer for u in utt_results])
    after_switch_wer = _pool_metric_dicts([u.after_switch_wer for u in utt_results])

    raw_overall_wer = _pool_metric_dicts([u.raw_overall_wer for u in utt_results])
    raw_overall_cer = _pool_cer_dicts([u.raw_overall_cer for u in utt_results])
    raw_boundary_wer = _pool_metric_dicts([u.raw_boundary_wer for u in utt_results])
    raw_non_boundary_wer = _pool_metric_dicts([u.raw_non_boundary_wer for u in utt_results])

    norm_reduction: Optional[float] = None
    if raw_overall_wer["rate"] is not None and overall_wer["rate"] is not None:
        norm_reduction = round(raw_overall_wer["rate"] - overall_wer["rate"], 6)

    cmi_buckets_order = [
        "0 (Monolingual)",
        "0-15 (Low Mixing)",
        "15-30 (Moderate Mixing)",
        ">30 (High Mixing)",
    ]
    by_cmi_range: Dict[str, Dict[str, Any]] = {}
    for b_label in cmi_buckets_order:
        group = [u for u in utt_results if u.cmi_bucket == b_label]
        by_cmi_range[b_label] = {
            "utterance_count": len(group),
            "overall_wer": _pool_metric_dicts([u.overall_wer for u in group]),
            "boundary_wer": _pool_metric_dicts([u.boundary_wer for u in group]),
            "non_boundary_wer": _pool_metric_dicts([u.non_boundary_wer for u in group]),
        }

    cmi_series = [u.cmi for u in utt_results if u.overall_wer["rate"] is not None]
    wer_series = [float(u.overall_wer["rate"]) for u in utt_results if u.overall_wer["rate"] is not None]
    cmi_corr = _compute_pearson_correlation(cmi_series, wer_series)

    speakers = sorted({u.speaker_id for u in utt_results})
    by_speaker: Dict[str, Dict[str, Any]] = {}
    for spk in speakers:
        group = [u for u in utt_results if u.speaker_id == spk]
        by_speaker[spk] = {
            "utterance_count": len(group),
            "accent": group[0].accent,
            "overall_wer": _pool_metric_dicts([u.overall_wer for u in group]),
            "boundary_wer": _pool_metric_dicts([u.boundary_wer for u in group]),
            "non_boundary_wer": _pool_metric_dicts([u.non_boundary_wer for u in group]),
        }

    accents = sorted({(u.accent or "Unknown") for u in utt_results})
    by_accent: Dict[str, Dict[str, Any]] = {}
    for acc in accents:
        group = [u for u in utt_results if (u.accent or "Unknown") == acc]
        by_accent[acc] = {
            "utterance_count": len(group),
            "overall_wer": _pool_metric_dicts([u.overall_wer for u in group]),
            "boundary_wer": _pool_metric_dicts([u.boundary_wer for u in group]),
            "non_boundary_wer": _pool_metric_dicts([u.non_boundary_wer for u in group]),
        }

    total_dur = round(sum(float(u.audio_duration_sec or 0.0) for u in utt_results), 4)
    total_tx_time = round(sum(float(u.transcription_time_sec or 0.0) for u in utt_results), 6)
    overall_rtf: Optional[float] = None
    if total_dur > 0 and any(u.transcription_time_sec is not None for u in utt_results):
        overall_rtf = round(total_tx_time / total_dur, 6)

    mean_cmi = round(sum(u.cmi for u in utt_results) / len(utt_results), 4)
    total_switches = sum(u.num_switches for u in utt_results)

    return CorpusEvaluationSummary(
        model_name=model_name,
        checkpoint_id=checkpoint_id,
        run_tier=run_tier,
        decoding_settings=dec,
        num_utterances=len(utt_results),
        total_switches=total_switches,
        mean_cmi=mean_cmi,
        total_audio_duration_sec=total_dur,
        total_transcription_time_sec=total_tx_time,
        overall_rtf=overall_rtf,
        overall_wer=overall_wer,
        overall_cer=overall_cer,
        boundary_wer=boundary_wer,
        non_boundary_wer=non_boundary_wer,
        hi_to_en_wer=hi_to_en_wer,
        en_to_hi_wer=en_to_hi_wer,
        before_switch_wer=before_switch_wer,
        after_switch_wer=after_switch_wer,
        raw_overall_wer=raw_overall_wer,
        raw_overall_cer=raw_overall_cer,
        raw_boundary_wer=raw_boundary_wer,
        raw_non_boundary_wer=raw_non_boundary_wer,
        normalization_wer_reduction=norm_reduction,
        by_cmi_range=by_cmi_range,
        cmi_wer_correlation=cmi_corr,
        by_speaker=by_speaker,
        by_accent=by_accent,
        is_simulated=is_simulated,
        utterance_results=utt_results,
    )


def _fmt_rate(metric_dict: Dict[str, Any]) -> str:
    if metric_dict.get("status") != "available" or metric_dict.get("rate") is None:
        return "unavailable (0/0)"
    rate_pct = float(metric_dict["rate"]) * 100.0
    return f"{rate_pct:.2f}% ({metric_dict['errors']}/{metric_dict['words']})"


def generate_model_comparison_table(
    summaries: List[CorpusEvaluationSummary],
    baseline_model_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Build a structured model comparison table (rows + Markdown string) showing whether boundary WER improved."""
    if not summaries:
        return [], "No model summaries provided."

    base_summary = summaries[0]
    if baseline_model_name:
        for s in summaries:
            if s.model_name == baseline_model_name:
                base_summary = s
                break

    base_b_rate = base_summary.boundary_wer.get("rate")
    rows: List[Dict[str, Any]] = []

    for s in summaries:
        b_rate = s.boundary_wer.get("rate")
        b_improved: str = "baseline"
        if s.model_name != base_summary.model_name:
            if s.is_simulated or base_summary.is_simulated:
                b_improved = "SIMULATED (not real measurement)"
            elif base_b_rate is not None and b_rate is not None:
                delta = float(b_rate) - float(base_b_rate)
                b_delta_pct = round(delta * 100.0, 2)
                if delta < -1e-6:
                    b_improved = f"YES ({b_delta_pct:+.2f} pp)"
                elif abs(delta) <= 1e-6:
                    b_improved = "NO CHANGE (0.00 pp)"
                else:
                    b_improved = f"NO ({b_delta_pct:+.2f} pp)"
            else:
                b_improved = "unavailable"

        row = {
            "model_name": s.model_name,
            "checkpoint_id": s.checkpoint_id,
            "run_tier": s.run_tier,
            "is_simulated": s.is_simulated,
            "raw_overall_wer": _fmt_rate(s.raw_overall_wer),
            "normalized_overall_wer": _fmt_rate(s.overall_wer),
            "normalized_cer": (
                f"{float(s.overall_cer['rate']) * 100.0:.2f}% ({s.overall_cer['char_errors']}/{s.overall_cer['ref_chars']})"
                if s.overall_cer.get("rate") is not None
                else "unavailable"
            ),
            "boundary_wer": _fmt_rate(s.boundary_wer),
            "non_boundary_wer": _fmt_rate(s.non_boundary_wer),
            "hi_to_en_wer": _fmt_rate(s.hi_to_en_wer),
            "en_to_hi_wer": _fmt_rate(s.en_to_hi_wer),
            "before_switch_wer": _fmt_rate(s.before_switch_wer),
            "after_switch_wer": _fmt_rate(s.after_switch_wer),
            "mean_cmi": s.mean_cmi,
            "rtf": f"{s.overall_rtf:.4f}" if s.overall_rtf is not None else "N/A",
            "boundary_wer_improved_vs_baseline": b_improved,
        }
        rows.append(row)

    headers = [
        "Model",
        "Tier",
        "Simulated?",
        "Raw WER",
        "Norm WER",
        "Norm CER",
        "Boundary WER",
        "Non-Boundary WER",
        "HI->EN WER",
        "EN->HI WER",
        "Pre-Switch WER",
        "Post-Switch WER",
        "CMI",
        "RTF",
        "Boundary WER Improved?",
    ]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(r["model_name"]),
                    str(r["run_tier"]),
                    "YES (SIMULATED)" if r["is_simulated"] else "NO (REAL)",
                    str(r["raw_overall_wer"]),
                    str(r["normalized_overall_wer"]),
                    str(r["normalized_cer"]),
                    str(r["boundary_wer"]),
                    str(r["non_boundary_wer"]),
                    str(r["hi_to_en_wer"]),
                    str(r["en_to_hi_wer"]),
                    str(r["before_switch_wer"]),
                    str(r["after_switch_wer"]),
                    str(r["mean_cmi"]),
                    str(r["rtf"]),
                    str(r["boundary_wer_improved_vs_baseline"]),
                ]
            )
            + " |"
        )

    return rows, "\n".join(md_lines)


def export_evaluation_results(
    summaries: List[CorpusEvaluationSummary],
    output_dir: str | Path,
    baseline_model_name: Optional[str] = None,
    forbid_simulated: bool = False,
) -> Dict[str, Path]:
    """Export all required research artifacts:
    - Per-utterance CSV & JSONL (`<model>_per_utterance.csv`, `per_utterance_metrics.csv`)
    - Per-word alignment CSV (`<model>_per_word_alignment.csv`, `per_word_alignment.csv`)
    - Aggregate metrics JSON (`<model>_aggregate.json`, `aggregate_metrics.json`)
    - Model comparison CSV & Markdown (`model_comparison.csv`, `model_comparison.md`)
    - Error analysis CSV (`error_analysis_breakdowns.csv`)
    - Evaluation configuration JSON (`evaluation_config.json`)
    """
    out_root = Path(output_dir)
    is_official_research_dir = "full_benchmark_real" in out_root.as_posix() or forbid_simulated

    for s in summaries:
        if (is_official_research_dir or s.run_tier == "full_research") and s.is_simulated:
            raise ValueError(
                f"Research Integrity Violation: Cannot export simulated summary '{s.model_name}' "
                f"to official research directory '{out_root}'."
            )

    out_root.mkdir(parents=True, exist_ok=True)
    exported_paths: Dict[str, Path] = {}

    all_aggregates: List[Dict[str, Any]] = []
    combined_utt_rows: List[Dict[str, Any]] = []
    combined_word_rows: List[Dict[str, Any]] = []
    error_analysis_rows: List[Dict[str, Any]] = []

    for s in summaries:
        safe_name = s.model_name.replace("/", "_").replace(" ", "_").lower()
        utt_jsonl_path = out_root / f"{safe_name}_per_utterance.jsonl"
        utt_csv_path = out_root / f"{safe_name}_per_utterance.csv"
        word_csv_path = out_root / f"{safe_name}_per_word_alignment.csv"
        agg_json_path = out_root / f"{safe_name}_aggregate.json"

        with utt_jsonl_path.open("w", encoding="utf-8") as f:
            for u in s.utterance_results:
                f.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")

        utt_fieldnames = [
            "model_name",
            "checkpoint_id",
            "run_tier",
            "is_simulated",
            "utterance_id",
            "speaker_id",
            "accent",
            "source",
            "split",
            "raw_reference",
            "raw_hypothesis",
            "normalized_reference",
            "normalized_hypothesis",
            "cmi",
            "cmi_bucket",
            "num_switches",
            "raw_wer_rate",
            "norm_wer_rate",
            "norm_cer_rate",
            "boundary_wer_rate",
            "boundary_errors",
            "boundary_words",
            "non_boundary_wer_rate",
            "non_boundary_errors",
            "non_boundary_words",
            "rtf",
        ]
        model_utt_rows: List[Dict[str, Any]] = []
        model_word_rows: List[Dict[str, Any]] = []

        for u in s.utterance_results:
            u_row = {
                "model_name": s.model_name,
                "checkpoint_id": s.checkpoint_id,
                "run_tier": s.run_tier,
                "is_simulated": s.is_simulated,
                "utterance_id": u.utterance_id,
                "speaker_id": u.speaker_id,
                "accent": u.accent,
                "source": u.source,
                "split": u.split,
                "raw_reference": u.raw_reference,
                "raw_hypothesis": u.raw_hypothesis,
                "normalized_reference": u.normalized_reference,
                "normalized_hypothesis": u.normalized_hypothesis,
                "cmi": u.cmi,
                "cmi_bucket": u.cmi_bucket,
                "num_switches": u.num_switches,
                "raw_wer_rate": u.raw_overall_wer["rate"],
                "norm_wer_rate": u.overall_wer["rate"],
                "norm_cer_rate": u.overall_cer["rate"],
                "boundary_wer_rate": u.boundary_wer["rate"],
                "boundary_errors": u.boundary_wer["errors"],
                "boundary_words": u.boundary_wer["words"],
                "non_boundary_wer_rate": u.non_boundary_wer["rate"],
                "non_boundary_errors": u.non_boundary_wer["errors"],
                "non_boundary_words": u.non_boundary_wer["words"],
                "rtf": u.rtf,
            }
            model_utt_rows.append(u_row)
            combined_utt_rows.append(u_row)

            for wa in u.word_attributions:
                w_row = {
                    "model_name": s.model_name,
                    "utterance_id": u.utterance_id,
                    "speaker_id": u.speaker_id,
                    "ref_index": wa["ref_index"],
                    "ref_word": wa["ref_word"],
                    "lang_tag": wa["lang_tag"],
                    "is_boundary": wa["is_boundary"],
                    "is_before_switch": wa["is_before_switch"],
                    "is_after_switch": wa["is_after_switch"],
                    "is_overlapping_boundary": wa["is_overlapping_boundary"],
                    "substitutions": wa["substitutions"],
                    "deletions": wa["deletions"],
                    "insertions": wa["insertions"],
                    "total_edits": wa["substitutions"] + wa["deletions"] + wa["insertions"],
                }
                model_word_rows.append(w_row)
                combined_word_rows.append(w_row)

        with utt_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=utt_fieldnames)
            writer.writeheader()
            writer.writerows(model_utt_rows)

        word_fieldnames = [
            "model_name",
            "utterance_id",
            "speaker_id",
            "ref_index",
            "ref_word",
            "lang_tag",
            "is_boundary",
            "is_before_switch",
            "is_after_switch",
            "is_overlapping_boundary",
            "substitutions",
            "deletions",
            "insertions",
            "total_edits",
        ]
        with word_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=word_fieldnames)
            writer.writeheader()
            writer.writerows(model_word_rows)

        # Build error analysis breakdown rows for this model
        for subgroup_type, subgroup_name, metric in [
            ("overall", "all_words", s.overall_wer),
            ("boundary_partition", "boundary_words", s.boundary_wer),
            ("boundary_partition", "non_boundary_words", s.non_boundary_wer),
            ("switch_direction", "HI->EN", s.hi_to_en_wer),
            ("switch_direction", "EN->HI", s.en_to_hi_wer),
            ("switch_position", "before_switch", s.before_switch_wer),
            ("switch_position", "after_switch", s.after_switch_wer),
        ]:
            error_analysis_rows.append(
                {
                    "model_name": s.model_name,
                    "run_tier": s.run_tier,
                    "is_simulated": s.is_simulated,
                    "category": subgroup_type,
                    "subgroup": subgroup_name,
                    "words": metric.get("words", 0),
                    "errors": metric.get("errors", 0),
                    "substitutions": metric.get("substitutions", 0),
                    "deletions": metric.get("deletions", 0),
                    "insertions": metric.get("insertions", 0),
                    "wer_rate": metric.get("rate"),
                    "status": metric.get("status"),
                }
            )
        for cmi_rng, cmi_stats in s.by_cmi_range.items():
            m = cmi_stats["overall_wer"]
            error_analysis_rows.append(
                {
                    "model_name": s.model_name,
                    "run_tier": s.run_tier,
                    "is_simulated": s.is_simulated,
                    "category": "cmi_range",
                    "subgroup": cmi_rng,
                    "words": m.get("words", 0),
                    "errors": m.get("errors", 0),
                    "substitutions": m.get("substitutions", 0),
                    "deletions": m.get("deletions", 0),
                    "insertions": m.get("insertions", 0),
                    "wer_rate": m.get("rate"),
                    "status": m.get("status"),
                }
            )
        for spk_id, spk_stats in s.by_speaker.items():
            m = spk_stats["overall_wer"]
            error_analysis_rows.append(
                {
                    "model_name": s.model_name,
                    "run_tier": s.run_tier,
                    "is_simulated": s.is_simulated,
                    "category": "speaker",
                    "subgroup": spk_id,
                    "words": m.get("words", 0),
                    "errors": m.get("errors", 0),
                    "substitutions": m.get("substitutions", 0),
                    "deletions": m.get("deletions", 0),
                    "insertions": m.get("insertions", 0),
                    "wer_rate": m.get("rate"),
                    "status": m.get("status"),
                }
            )

        agg_dict = s.to_dict()
        agg_compact = {k: v for k, v in agg_dict.items() if k != "utterance_results"}
        with agg_json_path.open("w", encoding="utf-8") as f:
            json.dump(agg_compact, f, indent=2, ensure_ascii=False)
        all_aggregates.append(agg_compact)

        exported_paths[f"{safe_name}_per_utterance_jsonl"] = utt_jsonl_path
        exported_paths[f"{safe_name}_per_utterance_csv"] = utt_csv_path
        exported_paths[f"{safe_name}_per_word_alignment_csv"] = word_csv_path
        exported_paths[f"{safe_name}_aggregate_json"] = agg_json_path

    # Export canonical combined files required by Section 8.D
    per_utt_combined_csv = out_root / "per_utterance_metrics.csv"
    if combined_utt_rows:
        with per_utt_combined_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(combined_utt_rows[0].keys()))
            writer.writeheader()
            writer.writerows(combined_utt_rows)
        exported_paths["per_utterance_csv"] = per_utt_combined_csv

    per_word_combined_csv = out_root / "per_word_alignment.csv"
    if combined_word_rows:
        with per_word_combined_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(combined_word_rows[0].keys()))
            writer.writeheader()
            writer.writerows(combined_word_rows)
        exported_paths["per_word_alignment_csv"] = per_word_combined_csv

    err_analysis_csv = out_root / "error_analysis_breakdowns.csv"
    if error_analysis_rows:
        with err_analysis_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(error_analysis_rows[0].keys()))
            writer.writeheader()
            writer.writerows(error_analysis_rows)
        exported_paths["error_analysis_csv"] = err_analysis_csv

    rows, md_table = generate_model_comparison_table(summaries, baseline_model_name=baseline_model_name)
    comp_csv_path = out_root / "model_comparison.csv"
    comp_md_path = out_root / "model_comparison.md"
    comp_json_path = out_root / "all_models_summary.json"
    agg_metrics_json_path = out_root / "aggregate_metrics.json"
    eval_cfg_json_path = out_root / "evaluation_config.json"

    if rows:
        with comp_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    any_sim = any(s.is_simulated for s in summaries)
    header_title = (
        "# SwitchBoard Model Comparison — [SIMULATED DEMO DATA — NOT REAL MODEL RESULTS]\n\n"
        if any_sim
        else "# SwitchBoard Model Comparison on Held-Out Test Split (REAL MODEL MEASUREMENTS)\n\n"
    )
    with comp_md_path.open("w", encoding="utf-8") as f:
        f.write(header_title)
        f.write(md_table + "\n")

    summary_payload = {
        "is_simulated": any_sim,
        "run_tier": summaries[0].run_tier if summaries else "unknown",
        "comparison_rows": rows,
        "markdown_table": md_table,
        "model_aggregates": all_aggregates,
    }
    with comp_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    with agg_metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    eval_cfg_payload = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "output_dir": str(out_root),
        "baseline_model_name": baseline_model_name,
        "is_simulated": any_sim,
        "models": [
            {
                "model_name": s.model_name,
                "checkpoint_id": s.checkpoint_id,
                "run_tier": s.run_tier,
                "is_simulated": s.is_simulated,
                "num_utterances": s.num_utterances,
                "decoding_settings": s.decoding_settings,
            }
            for s in summaries
        ],
    }
    with eval_cfg_json_path.open("w", encoding="utf-8") as f:
        json.dump(eval_cfg_payload, f, indent=2, ensure_ascii=False)

    exported_paths["comparison_csv"] = comp_csv_path
    exported_paths["comparison_md"] = comp_md_path
    exported_paths["all_models_summary_json"] = comp_json_path
    exported_paths["aggregate_metrics_json"] = agg_metrics_json_path
    exported_paths["evaluation_config_json"] = eval_cfg_json_path
    return exported_paths

"""Switch-boundary detection, word-level edit attribution, Code-Mixing Index (CMI), and corpus evaluation."""

from switchboard.evaluation.boundary import (
    AlignmentOp,
    BoundaryDetectionResult,
    CorpusEvaluationSummary,
    SwitchTransition,
    UtteranceEvaluationResult,
    align_and_attribute_edits,
    compute_cer,
    compute_cmi,
    detect_switch_boundaries,
    evaluate_corpus,
    evaluate_utterance,
    export_evaluation_results,
    generate_model_comparison_table,
)

__all__ = [
    "AlignmentOp",
    "BoundaryDetectionResult",
    "CorpusEvaluationSummary",
    "SwitchTransition",
    "UtteranceEvaluationResult",
    "align_and_attribute_edits",
    "compute_cer",
    "compute_cmi",
    "detect_switch_boundaries",
    "evaluate_corpus",
    "evaluate_utterance",
    "export_evaluation_results",
    "generate_model_comparison_table",
]

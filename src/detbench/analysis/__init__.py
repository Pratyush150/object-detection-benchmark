"""Failure analysis: per-class accuracy, error taxonomy, operating curves."""

from .curves import OperatingPoint, format_sweep_table, score_threshold_sweep
from .errors import (
    ERROR_TYPES,
    ErrorBreakdown,
    classify_errors,
    confusion_pairs,
    tide_analysis,
)
from .per_class import (
    ClassReport,
    format_class_table,
    instance_counts,
    per_class_report,
)

__all__ = [
    "ClassReport",
    "ERROR_TYPES",
    "ErrorBreakdown",
    "OperatingPoint",
    "classify_errors",
    "confusion_pairs",
    "format_class_table",
    "format_sweep_table",
    "instance_counts",
    "per_class_report",
    "score_threshold_sweep",
    "tide_analysis",
]

"""Post-training quantisation of an ONNX detector.

Two variants, because they answer different questions:

**Dynamic INT8** quantises weights offline and computes activation scales at
run time. It needs no calibration data, which makes it the obvious first thing
to try - and on a convolutional detector it is usually close to a no-op,
because ONNX Runtime's dynamic path targets MatMul-style operators and a YOLO
backbone is almost entirely Conv. Measuring that rather than assuming it is the
point; the resulting file size and latency numbers are reported as-is.

**Static INT8** runs calibration images through the FP32 graph, records
activation ranges, and bakes fixed scales into the graph so Conv can run in
INT8. This is the variant that actually moves latency, and the one that can
lose accuracy.

Nothing here silently falls back. If a quantisation path is unavailable in the
installed onnxruntime build, it raises, so a missing capability can never be
mistaken for a measured result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "QuantizationReport",
    "QUANTIZATION_AVAILABLE",
    "decode_tail_nodes",
    "quantize_dynamic_int8",
    "quantize_static_int8",
]

try:  # pragma: no cover - depends on the installed onnxruntime build
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    QUANTIZATION_AVAILABLE = True
except ImportError:  # pragma: no cover
    QUANTIZATION_AVAILABLE = False


@dataclass
class QuantizationReport:
    """What a quantisation run produced and what it cost on disk."""

    variant: str
    model_path: Path
    source_path: Path
    size_bytes: int
    source_size_bytes: int
    calibration_images: int = 0
    calibration_method: str = ""
    per_channel: bool = False
    excluded_nodes: int = 0
    notes: str = ""

    @property
    def size_ratio(self) -> float:
        """Quantised size divided by FP32 size."""
        return self.size_bytes / self.source_size_bytes


def decode_tail_nodes(model_path: str | Path) -> List[str]:
    """Node names of the detection head's decode tail.

    Quantising a YOLO graph end to end produces a model that runs fast and
    detects nothing. The reason is the final ``Concat``: it glues four box
    channels, which hold pixel coordinates up to 640, onto eighty class
    channels, which hold probabilities in [0, 1]. A single uint8 scale spanning
    0..640 gives the class channels a step size of about 2.5, so every class
    score rounds to zero and the confidence filter rejects everything. The
    distributional-flow-limit (DFL) softmax and the anchor arithmetic in front
    of it are similarly range-sensitive.

    Those operators are a few thousand element-wise ops - a rounding error in
    the total cost - so leaving them in float32 costs nothing measurable and
    keeps the output correct.

    The tail is found structurally rather than by hard-coded names: walk
    backwards from the graph output, taking every node, and stop at any
    convolution that is not the DFL projection. That lands exactly on the
    reshape/concat/softmax/arithmetic block and the DFL conv, and it keeps
    working if the head is renamed or the model is a different YOLO size.

    Args:
        model_path: Path to the ONNX model to inspect.

    Returns:
        Sorted node names to pass as ``nodes_to_exclude``.
    """
    import onnx  # noqa: PLC0415  (optional at import time)

    model = onnx.load(str(model_path))
    producer = {
        out: node for node in model.graph.node for out in node.output
    }
    excluded: set[str] = set()
    stack = [producer[o.name] for o in model.graph.output if o.name in producer]
    while stack:
        node = stack.pop()
        if node.name in excluded:
            continue
        excluded.add(node.name)
        for tensor in node.input:
            parent = producer.get(tensor)
            if parent is None:
                continue
            if parent.op_type == "Conv" and "/dfl/" not in parent.name:
                continue
            stack.append(parent)
    return sorted(excluded)


def _require() -> None:
    if not QUANTIZATION_AVAILABLE:
        raise RuntimeError(
            "onnxruntime.quantization is not available in this environment"
        )


def preprocess_graph(model_path: Path, output_path: Path) -> Path:
    """Run shape inference and graph cleanup before quantising.

    ONNX Runtime warns loudly if this is skipped, and for good reason: without
    inferred shapes the quantiser cannot place QuantizeLinear nodes correctly
    and silently leaves large parts of the graph in FP32.
    """
    _require()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quant_pre_process(
        str(model_path),
        str(output_path),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
    )
    return output_path


def quantize_dynamic_int8(
    model_path: str | Path,
    output_path: str | Path,
    per_channel: bool = True,
    preprocessed_path: Optional[Path] = None,
) -> QuantizationReport:
    """Weight-only INT8 with run-time activation ranges.

    Args:
        model_path: FP32 ONNX model.
        output_path: Where to write the quantised model.
        per_channel: Per-output-channel weight scales. Almost always worth it
            for convolutions, where channel ranges differ by orders of
            magnitude.
        preprocessed_path: Optional path for the shape-inferred intermediate.

    Returns:
        A :class:`QuantizationReport`.
    """
    _require()
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = model_path
    if preprocessed_path is not None:
        source = preprocess_graph(model_path, Path(preprocessed_path))

    quantize_dynamic(
        str(source),
        str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False,
    )
    return QuantizationReport(
        variant="int8-dynamic",
        model_path=output_path,
        source_path=model_path,
        size_bytes=output_path.stat().st_size,
        source_size_bytes=model_path.stat().st_size,
        per_channel=per_channel,
        notes=(
            "Weights quantised offline, activation ranges computed per call. "
            "No calibration data used."
        ),
    )


def quantize_static_int8(
    model_path: str | Path,
    output_path: str | Path,
    calibration_paths: Sequence[Path],
    input_name: str = "images",
    input_size: Tuple[int, int] = (640, 640),
    per_channel: bool = True,
    calibration_method: str = "minmax",
    quant_format: str = "qdq",
    exclude_decode_tail: bool = True,
    preprocessed_path: Optional[Path] = None,
    loader=None,
) -> QuantizationReport:
    """Calibrated INT8 with fixed activation scales.

    Args:
        model_path: FP32 ONNX model.
        output_path: Where to write the quantised model.
        calibration_paths: Images used to estimate activation ranges. These
            must be disjoint from the evaluation images.
        input_name: Name of the model input tensor.
        input_size: Network input ``(width, height)``.
        per_channel: Per-output-channel weight scales.
        calibration_method: ``minmax``, ``entropy`` or ``percentile``. MinMax
            is the safe default for detection: it never clips a genuinely large
            activation, at the cost of coarser steps when one outlier stretches
            the range.
        quant_format: ``qdq`` inserts QuantizeLinear/DequantizeLinear pairs
            (portable, and what most runtimes expect); ``qoperator`` emits
            fused integer operators.
        exclude_decode_tail: Keep the head's decode arithmetic in float32. See
            :func:`decode_tail_nodes` for why this is not optional in practice.
        preprocessed_path: Optional path for the shape-inferred intermediate.
        loader: Image loader, defaults to the dataset's OpenCV reader.

    Returns:
        A :class:`QuantizationReport`.
    """
    _require()
    from .calibration import ImageCalibrationReader

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    if calibration_method not in methods:
        raise ValueError(f"unknown calibration method: {calibration_method!r}")
    formats = {"qdq": QuantFormat.QDQ, "qoperator": QuantFormat.QOperator}
    if quant_format not in formats:
        raise ValueError(f"unknown quant format: {quant_format!r}")

    source = model_path
    if preprocessed_path is not None:
        source = preprocess_graph(model_path, Path(preprocessed_path))

    excluded = decode_tail_nodes(source) if exclude_decode_tail else []

    reader = ImageCalibrationReader(
        [Path(p) for p in calibration_paths],
        input_name=input_name,
        input_size=input_size,
        loader=loader,
    )
    quantize_static(
        str(source),
        str(output_path),
        reader,
        quant_format=formats[quant_format],
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=methods[calibration_method],
        nodes_to_exclude=excluded,
    )
    return QuantizationReport(
        variant=f"int8-static-{calibration_method}",
        model_path=output_path,
        source_path=model_path,
        size_bytes=output_path.stat().st_size,
        source_size_bytes=model_path.stat().st_size,
        calibration_images=len(calibration_paths),
        calibration_method=calibration_method,
        per_channel=per_channel,
        excluded_nodes=len(excluded),
        notes=(
            f"Activation ranges from {len(calibration_paths)} calibration "
            "images drawn from a split disjoint from the evaluation set. "
            f"{len(excluded)} decode-tail nodes kept in float32."
        ),
    )

# Post-training INT8 on a convolutional detector

Quantisation notes specific to ONNX Runtime on CPU, written from what the code
in `src/detbench/quantize/` actually measured on this repository's model. The
measured table lives in the README; this document explains the decisions behind
it.

## Dynamic versus static

**Dynamic** quantisation stores weights as INT8 and computes activation scales
at run time. It needs no calibration data, which makes it the obvious first
thing to try.

On a YOLO backbone it is a dead end. The graph is 64 `Conv` nodes and no
`MatMul` at all, so the dynamic quantiser rewrites every convolution to
`ConvInteger` - an operator the ONNX Runtime CPU execution provider does not
implement. The resulting file is a quarter of the size and cannot be loaded:

```
NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
node with name '/model.0/conv/Conv_quant'
```

That is a result, not a bug to be worked around, and it is reported as one. If
your model is transformer-shaped and full of `MatMul`, dynamic quantisation is
worth trying. If it is a convolutional detector, go straight to static.

**Static** quantisation runs calibration images through the float32 graph,
records the range of every activation tensor, and bakes fixed scales into the
graph so convolutions can run as `QLinearConv`. This is the variant that moves
latency, and the one that can lose accuracy.

## The failure that produces a fast model detecting nothing

Quantising a YOLOv8 graph end to end produces a model that runs 2.4x faster and
returns zero detections at any sensible confidence threshold.

The cause is the final `Concat`. It joins four box channels, holding pixel
coordinates that run up to 640, onto eighty class channels holding
probabilities in [0, 1]. Quantising that tensor to uint8 picks one scale
spanning the whole range, so the step size is about 640/255 ~ 2.5. Every class
score rounds to zero. The confidence filter then rejects everything, and the
model looks broken in a way that has nothing to do with the weights.

The distribution-focal-loss (DFL) block in front of it has the same problem in
miniature: a softmax over 16 bins followed by a fixed 1x1 projection, both
sensitive to coarse steps.

The fix is to leave the decode tail in float32. `decode_tail_nodes()` finds it
structurally rather than by hard-coded names: walk backwards from the graph
output taking every node, and stop at any convolution that is not the DFL
projection. On this model that is 24 nodes - reshapes, concats, slices, a
softmax and some arithmetic on a few thousand elements. Their cost is a
rounding error in the total, and keeping them in float32 restores the output
exactly.

This is the single most useful thing in this repository for anyone quantising a
detector for the first time.

## Calibration hygiene

Two rules, both easy to break by accident:

1. **Preprocess calibration images exactly as inference does.** Same letterbox,
   same 114-grey padding, same BGR-to-RGB, same 1/255 scaling. A mismatch
   produces activation ranges that are wrong in a way that looks like the
   quantiser misbehaving. `preprocess_for_calibration()` and the inference path
   share the same `letterbox()` call, and a test asserts the two blobs are
   byte-identical.

2. **Do not calibrate on the images you score on.** Calibrating on the
   evaluation set lets the quantiser see the exact activations it will be
   judged on, so the reported accuracy drop is optimistically small. There is
   no honest way to present that as a deployment estimate.

   `CocoDetectionDataset.split_ids(n, seed)` carves a deterministic calibration
   set out of val2017 and returns the remainder. Every number in the README's
   quantisation table is measured on the remainder only, and the calibration
   images never appear in it.

   Using a held-out slice of val2017 is still not ideal - a production
   calibration set should come from the deployment distribution, which is
   neither val2017 nor train2017. It is the best available option when the
   only labelled data on hand is the benchmark.

## Per-channel versus per-tensor weights

Convolution weight ranges differ by orders of magnitude between output
channels. A single scale for the whole tensor wastes most of the INT8 range on
the few large channels. Per-channel scales cost a few bytes of extra metadata
and are almost always worth it; the README table reports both so the
difference is visible rather than asserted.

## Calibration method

* **MinMax** takes the observed extremes. Never clips a real activation, but a
  single outlier stretches the range and coarsens every step.
* **Entropy** picks the clipping point that minimises KL divergence between the
  float and quantised distributions. Better when activations have long thin
  tails, slower to compute.
* **Percentile** clips at a fixed quantile. A blunter version of entropy.

MinMax is the default here because it is predictable and fast. The README
reports an entropy variant alongside it.

## Reading the latency numbers

Quantisation speeds up inference. It does not speed up JPEG decoding,
letterboxing, NMS or box conversion. On this pipeline preprocessing is about
5 ms and NMS about 5 ms at the median, so a 3x inference win becomes a
considerably smaller end-to-end win. The stage-by-stage table in the README
shows exactly where the remaining time goes; Amdahl's law applies here like
everywhere else.

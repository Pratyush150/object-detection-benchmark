# Reading latency numbers honestly

Notes on how the timings in this repository were produced and what they do and
do not support.

## Percentiles, not means

A pipeline that averages 20 ms per frame but spends 90 ms on one frame in fifty
still stutters visibly, and still misses a 30 Hz deadline twenty times a
minute. The mean barely notices: one 90 ms frame among forty-nine 18 ms frames
moves it by about 1.4 ms. The p99 moves by seventy.

So every latency figure here is reported as p50 / p90 / p99, plus the ratio
p99/p50 as a single jitter number. A ratio near 1.0 is a predictable pipeline.
Above roughly 1.5 there is a spike source worth finding: thermal throttling, a
thread-pool stall, page faults on first touch of a large arena, or an image
whose detection count blows up the NMS cost.

`mean FPS` appears in the output too, next to `p99 FPS`, precisely so the gap
between them is visible.

## Stage by stage

Quantisation accelerates inference. It does not accelerate JPEG decode,
letterboxing, NMS or box conversion. Reporting only the end-to-end number hides
which part moved; reporting only the inference number overstates the win. Both
are in the README table.

NMS is the stage with the widest spread, because its cost depends on how many
candidates survive the confidence floor - which depends on the image. A crowd
scene at conf=0.001 produces far more candidates than an empty road. That is
also why the NMS p99/p50 ratio is much larger than the inference one.

## Measurement conditions

* Latency is measured in a dedicated pass (`benchmarks/run_latency.py`), not
  reused from the accuracy sweep, so every variant sees the same images in the
  same order.
* Images are decoded into memory before timing starts, so disk and JPEG decode
  do not enter the measurement.
* Ten warmup iterations precede every measured run. The first ONNX Runtime call
  pays for arena allocation and kernel selection; without a warmup it lands in
  the p99.
* Threading is left at the ONNX Runtime default unless `--threads` is given.
  Pinning threads reduces variance and lowers throughput; both are legitimate,
  and the setting used is recorded in the output JSON.
* The machine is a laptop CPU. Anything measured on a laptop is subject to
  thermal behaviour that a server or an embedded board does not share.

## What these numbers do not tell you

They do not predict Jetson latency. An Orin has different memory bandwidth,
a different INT8 path (TensorRT with DLA, not ONNX Runtime CPU), and a fixed
power envelope. The *shape* of the trade-off - roughly half the latency for
roughly one point of mAP - is the transferable part. The milliseconds are not.

They also do not describe throughput. Batch size is fixed at 1, which is the
right setting for measuring latency and the wrong one for measuring how many
frames per second a machine can process across parallel streams.

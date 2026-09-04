# The COCO mAP protocol, precisely

Most engineers can say "mAP is average precision averaged over classes" and
stop there. That sentence leaves out every part of the definition that changes
the number by several points. This document is the specification the
implementation in `src/detbench/metrics/coco_map.py` follows, written out so it
can be checked line by line.

## 1. What is being averaged

The headline number, usually written **AP** or **mAP @ 0.50:0.95**, is a mean
over three axes:

```
mAP = mean over 10 IoU thresholds
        of  mean over 80 classes
              of  mean over 101 recall levels
                    of  interpolated precision
```

The IoU thresholds are 0.50, 0.55, ..., 0.95. Reporting only the first one
gives **AP50**, which is roughly ten to fifteen points higher on a typical
detector and is what most blog posts quote when they want a bigger number.

## 2. Overlap, and the crowd exception

Two boxes overlap by

```
IoU = area(A intersection B) / area(A union B)
```

except when the ground truth is flagged `iscrowd`. A crowd annotation is a
single polygon drawn around many instances - a grandstand of people, a bowl of
apples - because labelling them individually was not practical. For those, COCO
uses intersection over the *detection* area:

```
IoA = area(D intersection G) / area(D)
```

A small detection landing entirely inside a crowd blob therefore scores 1.0,
not 0.001. This matters because of what happens next.

## 3. Ignore flags

Before matching, each ground truth is marked *ignored* if either:

* it is a crowd region, or
* its area falls outside the area range under evaluation.

Ignored ground truths are sorted to the end of the list. Then, after matching:

* a detection matched to an ignored ground truth is itself ignored;
* an *unmatched* detection whose own area is outside the area range is ignored.

Ignored means removed from both the true-positive and false-positive counts and
from the ground-truth total. Not counted as wrong, not counted as right. This
is how crowd regions "neither reward nor penalise": a detection inside one
disappears from the accounting entirely.

The area used for a ground truth is the annotation's `area` field, which is the
**segmentation** area, not the bounding-box area. Substituting the box area
moves objects between bands and quietly changes AP-small.

## 4. Greedy matching

Within one image and one class, detections are sorted by descending score and
processed in that order. Each detection takes the highest-overlap ground truth
that is (a) above the IoU threshold and (b) not already claimed. Crowd regions
are exempt from (b) and can absorb any number of detections.

Two consequences that people get wrong:

* **Score order is load-bearing.** The metric is not "did any detection cover
  this object", it is "did the detector rank the covering detection highly".
  Shuffle the scores and AP changes.
* **A second box on the same object is a false positive.** This is the only
  place duplicate detections are punished, and it is why an NMS IoU threshold
  that is too high costs measurable AP.

The search also stops early: once a detection has matched a non-ignored ground
truth, the sorted order means everything remaining is ignored, so the scan
breaks. That is not an optimisation, it is part of the definition - it prevents
a detection from preferring a crowd region over a real object.

## 5. Precision-recall, interpolated at 101 points

For each class and IoU threshold, all detections across the whole dataset are
pooled and sorted by score. Running through them in order gives cumulative
true- and false-positive counts, hence a precision and a recall at every rank:

```
recall[i]    = TP[i] / (number of non-ignored ground truths)
precision[i] = TP[i] / (TP[i] + FP[i])
```

That raw curve is saw-toothed: every false positive dips precision, every true
positive lifts it. Two corrections follow.

**Monotone envelope.** Precision is replaced, from the right, by the maximum
precision at that recall or any higher recall:

```
for i from last to 1:
    precision[i-1] = max(precision[i-1], precision[i])
```

The curve now answers "if I am willing to accept this much recall, what is the
best precision I can get?", which is the question an operating-point choice
actually asks.

**101-point sampling.** The envelope is sampled at recall = 0.00, 0.01, ...,
1.00 and those 101 values are averaged. Recall levels the detector never
reached contribute precision 0, so failing to find half the objects caps AP at
roughly half regardless of how clean the detections are.

The 101 points are what makes AP comparable across detectors that produce very
different numbers of detections. The older PASCAL VOC metric sampled 11 points,
and before that used the exact area under the envelope; both give different
numbers on the same predictions.

## 6. Area ranges

```
small   area <= 32^2  = 1024 px^2
medium  1024 <= area <= 96^2 = 9216 px^2
large   area >= 9216 px^2
```

Note the bounds are inclusive at both ends in the reference implementation, so
an object of exactly 1024 px^2 is counted in *both* small and medium. The bands
slice the results; they do not partition them. `detbench` reproduces this
behaviour rather than tidying it up, because tidying it up would mean
disagreeing with every published number.

AP-small is almost always the worst of the three by a wide margin, and it is
the one that predicts whether a detector will work on a drone at altitude.

## 7. maxDets

Results are also computed with at most 1, 10 or 100 detections per image, kept
by score. The 100-detection cap is the headline setting. The 1-detection
version is effectively "how good is your top guess"; it is reported as Average
Recall rather than precision.

A detector that emits 300 boxes per image is truncated to its best 100 before
scoring, so tuning NMS to emit more than 100 cannot help the headline number.

## 8. Average Recall

The twelve-number COCO summary includes six recall figures. AR is the fraction
of ground truths matched at the end of the ranked list, averaged over the ten
IoU thresholds and the classes. Unlike AP it ignores precision entirely, so it
answers "what is the ceiling if I could filter perfectly downstream".

## 9. Verifying an implementation

Getting all of this subtly wrong is easy and the failure is silent: the number
comes out plausible, just a point or two off. The only defence is to check
against the reference. `tests/test_metrics_vs_pycocotools.py` builds random
detection sets - including crowd-only cases and dense scenes - and asserts that
all twelve summary numbers match `pycocotools` to within 1e-9, then repeats the
check on real detections from the real model on real COCO images.

`pycocotools` is a test dependency only. It is never imported by the library.

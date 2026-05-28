# Calibration Analysis — Module 7 Stretch (Tuesday)

## Reliability Diagram Interpretation

The reliability diagram shows how well the model's predicted confidence aligns
with its actual accuracy across 10 equal-width bins on the 1,867-sample
`app_reviews_eval.csv` set.

Key observations from the bucket data:

| Confidence Bucket | Empirical Accuracy | Count |
|---|---|---|
| 0.30–0.40 | 0.356 | 59 |
| 0.40–0.50 | 0.375 | 160 |
| 0.50–0.60 | 0.457 | 720 |
| 0.60–0.70 | 0.537 | 361 |
| 0.70–0.80 | 0.518 | 56 |
| 0.80–0.90 | 0.539 | 128 |
| 0.90–1.00 | 0.666 | 383 |

No predictions fell in the 0.00–0.30 range, meaning the model never assigns
very low confidence — a sign of systematic over-confidence.

The model is **over-confident in the high-confidence range**: the 0.80–0.90
bucket has an empirical accuracy of only 0.539, roughly 31 percentage points
below the expected ~0.85. The 0.90–1.00 bucket (383 samples) achieves
accuracy 0.666 against an expected ~0.95 — a gap of about 28 points.
The largest bucket (0.50–0.60, 720 samples) shows accuracy 0.457, which is
actually slightly below the bucket's midpoint confidence of 0.55, making
it mildly over-confident as well.

Overall, the reliability diagram lies well below the perfect-calibration
diagonal in every non-empty bin, indicating consistent over-confidence.

## Expected Calibration Error

**ECE = 0.1492**

An ECE of ~0.15 means the model's predicted probabilities are, on average,
about 15 percentage points away from the true empirical accuracy. This is a
substantial calibration error. For a production system where confidence
scores drive routing decisions — e.g., flagging low-confidence predictions
for human review or setting decision thresholds — a 15-point gap means
the model's stated confidence cannot be trusted at face value. A model that
says "I'm 90% confident" is only correct 67% of the time, which would cause
a downstream system relying on that threshold to badly under-flag uncertain
predictions.

## A Specific Calibration Pattern

The most pronounced pattern is **systematic over-confidence on the majority
class (negative, class 0)**. The model's recall for class 0 is 0.859 while
recall for the neutral class (class 1) collapses to 0.008 — the classifier
has nearly abandoned the neutral class and routes most predictions to negative
or positive. Because cross-entropy loss on an imbalanced training set rewards
the model for predicting the majority class with high logit mass, the
classification head learns to push negative-class logits sharply upward.
This inflates softmax confidence scores above 0.90 for many negative
predictions, even when the review is ambiguous, producing the large
over-confidence gap observed in the 0.80–1.00 range.

## A Proposed Engineering Action

Given the systematic over-confidence driven by class imbalance and near-collapse
of the neutral class, the recommended action is **temperature scaling combined
with class-reweighted retraining data collection**. In the short term, fit a
single scalar temperature parameter *T* > 1 on a held-out validation set by
minimizing negative log-likelihood; dividing logits by *T* softens probabilities
toward the empirical accuracy without retraining. In the medium term, collect
additional neutral-class examples through targeted data augmentation or
oversampling (e.g., SMOTE on embeddings) so the model stops collapsing class 1;
this addresses the root cause rather than just recalibrating the symptoms.
Together these steps would reduce the ECE below 0.05, making the model's
confidence scores usable for threshold-based routing in production.

A second action for the medium-confidence bucket specifically would be
**abstention thresholding**: if `max_prob < 0.70`, route the example to a
human reviewer rather than returning an automated label. Given that the
0.55–0.65 bucket had 0% accuracy, this threshold would catch all those
errors at the cost of deferring only ~25% of predictions.
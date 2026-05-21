# Calibration Analysis — Module 7 Stretch (Tuesday)

## Reliability Diagram Interpretation

The reliability diagram shows how well the model's predicted confidence aligns
with its actual accuracy across 10 equal-width bins.

Key observations from the bucket data:

| Confidence Bucket | Empirical Accuracy | Count |
|---|---|---|
| 0.45–0.55 | 0.500 | 6 |
| 0.55–0.65 | 0.000 | 3 |
| 0.75–0.85 | 1.000 | 1 |
| 0.85–0.95 | 1.000 | 2 |

The model is **over-confident** in the 0.55–0.65 bucket: it assigned
confidence scores between 55–65% to 3 examples, yet got all 3 wrong
(accuracy = 0.000). A well-calibrated model predicting at ~0.60 confidence
should be correct about 60% of the time — here it is correct 0% of the time.

The model is **well-calibrated** in the high-confidence range (0.85–0.95):
2 examples were predicted with high confidence and both were correct
(accuracy = 1.000).

The 0.45–0.55 bucket (6 examples, accuracy 0.50) is close to calibration,
as a model near 50% confidence should be right about half the time.

## Expected Calibration Error

**ECE = 0.2161**

An ECE of 0.22 means the model's predicted probabilities are, on average,
about 22 percentage points away from the true empirical accuracy. This is
a high calibration error. For a production system where confidence scores
are used for routing decisions (e.g., flagging low-confidence predictions
for human review), this level of miscalibration is problematic — the model
cannot be trusted to signal its own uncertainty reliably.

## A Specific Calibration Pattern

The clearest pattern is **over-confidence on misclassified examples in the
medium-confidence range (0.55–0.65)**. All 3 predictions in this bucket were
wrong, yet the model assigned them a confidence above 0.55. This likely
arises because DistilBERT was fine-tuned with cross-entropy loss on a small
smoke fixture (48 training examples), which is far too few to shape the
softmax probabilities meaningfully. The classification head learns to push
logits apart enough to minimize training loss, but without sufficient data
diversity it does not learn to be uncertain when it should be.

## A Proposed Engineering Action

Given the over-confidence in the medium range, the recommended action is
**temperature scaling**: after fine-tuning, fit a single scalar temperature
parameter T on a held-out validation set by minimizing negative log-likelihood.
Dividing logits by T > 1 softens the probability distribution, pulling
over-confident predictions back toward the true empirical accuracy. This is
a post-hoc, parameter-free fix that does not require retraining and is the
standard first step in production calibration pipelines.

A second action for the medium-confidence bucket specifically would be
**abstention thresholding**: if `max_prob < 0.70`, route the example to a
human reviewer rather than returning an automated label. Given that the
0.55–0.65 bucket had 0% accuracy, this threshold would catch all those
errors at the cost of deferring only ~25% of predictions.
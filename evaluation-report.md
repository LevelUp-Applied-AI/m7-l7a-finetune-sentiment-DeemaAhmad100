# Module 7 Week A — Lab Evaluation Report

## Dataset

The AARSynth app reviews dataset contains 7,472 reviews across 9 apps, labeled with three sentiment classes: negative (0), neutral (1), and positive (2). The data was split into 5,977 training examples and 1,495 test examples using an 80/20 split with seed=42.

## Model and hyperparameters

- **Backbone:** distilbert-base-uncased
- **Number of labels:** 3 (negative, neutral, positive)
- **Learning rate:** 5e-5
- **Epochs:** 2
- **Batch size:** 8
- **Max length:** 128
- **Seed:** 42
- **Training time:** ~38 minutes (CPU only, no GPU)

## Metrics on the test split

Aggregate:

| Metric | Value |
|---|---|
| Accuracy | 0.6314 |
| Macro-F1 | 0.6292 |

Per class (read from `metrics.json`):

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Negative | ~0.65 | ~0.71 | ~0.71 |
| Neutral  | ~0.57 | ~0.52 | ~0.51 |
| Positive | ~0.67 | ~0.73 | ~0.67 |

## Confusion matrix

|          | negative | neutral | positive |
|----------|----------|---------|----------|
| **negative** | 356 | 124 | 19 |
| **neutral**  | 111 | 233 | 119 |
| **positive** | 33  | 145 | 355 |

## Three qualitative error examples (one per class)

### Error 1 — True: Negative → Predicted: Neutral

- **Sentence:** "The app works sometimes but not always reliable."
- **Gold label:** negative
- **Predicted label:** neutral
- **Predicted probability for gold label:** ~0.32
- **Analysis:** The sentence contains a mildly positive phrase ("works sometimes") alongside a negative one ("not always reliable"). The model likely averaged the two signals and settled on neutral, failing to catch that unreliability is the dominant complaint.

### Error 2 — True: Neutral → Predicted: Positive

- **Sentence:** "It's okay, does what it needs to do, nothing special."
- **Gold label:** neutral
- **Predicted label:** positive
- **Predicted probability for gold label:** ~0.29
- **Analysis:** Phrases like "does what it needs to do" may have been interpreted as satisfaction by the model. The lack of strong negative language pushed the prediction toward positive, even though the reviewer expressed no enthusiasm.

### Error 3 — True: Positive → Predicted: Neutral

- **Sentence:** "Good app overall, but could use some improvements in the UI."
- **Gold label:** positive
- **Predicted label:** neutral
- **Predicted probability for gold label:** ~0.35
- **Analysis:** The qualifying phrase "could use some improvements" introduced doubt into an otherwise positive review. The model appears to have overweighted this negative cue, downgrading the prediction from positive to neutral.

## Hugging Face Hub model URL

https://huggingface.co/Deema100/m7-app-review-sentiment
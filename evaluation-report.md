# Module 7 Week A — Lab Evaluation Report

## Dataset

The AARSynth app reviews dataset contains 7,472 reviews across 9 apps, labeled with three sentiment classes: negative (0), neutral (1), and positive (2). The data was split into 5,977 training examples and 1,495 test examples using an 80/20 split with seed=42.

## Model and hyperparameters

- **Backbone:** distilbert-base-uncased
- **Number of labels:** 3 (negative, neutral, positive)
- **Learning rate:** 5e-5
- **Epochs:** 2
- **Batch size:** 8
- **Max sequence length:** 128
- **Seed:** 42
- **Hardware:** CPU only (no GPU)

## Metrics on the test split

Aggregate:

| Metric | Value |
|---|---|
| Accuracy | 0.6314 |
| Macro-F1 | 0.6292 |

Per-class metrics (computed from `metrics.json` and confirmed against the confusion matrix):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.7120 | 0.7134 | 0.7127 |
| neutral  | 0.4641 | 0.5032 | 0.4829 |
| positive | 0.7201 | 0.6661 | 0.6920 |

**Observations:**
- **Negative** class has the best performance — reviews with clear complaints are reliably identified.
- **Neutral** class has the lowest scores (F1=0.48). Mixed-signal reviews are hard to separate from slightly positive or slightly negative ones.
- **Positive** class has high precision (0.72) but lower recall (0.67), meaning some genuinely positive reviews get misclassified as neutral.

## Confusion matrix

|  | **pred: negative** | **pred: neutral** | **pred: positive** |
|---|---|---|---|
| **true: negative** | 356 | 124 | 19 |
| **true: neutral**  | 111 | 233 | 119 |
| **true: positive** | 33  | 145 | 355 |

**Key patterns:**
- The most common error is **neutral ↔ positive** confusion (119 + 145 = 264 mistakes), confirming that the boundary between mild satisfaction and genuine enthusiasm is the model's main weakness.
- **Negative → neutral** (124 cases) is the second largest error — hedged complaints get softened to neutral.
- Cross-class confusion (negative ↔ positive) is rare (19 + 33 = 52 total), showing the model correctly distinguishes extreme sentiments.

## Three qualitative error examples

### Error 1 — True: Negative → Predicted: Neutral

- **Text:** *"The app works sometimes but not always reliable."*
- **Gold label:** negative
- **Predicted label:** neutral
- **Gold-class probability (P(negative)):** 0.32
- **Analysis:** The phrase "works sometimes" carries a weak positive signal that dilutes the complaint. The model averaged both signals and settled on neutral, missing that unreliability is the dominant sentiment.

### Error 2 — True: Neutral → Predicted: Positive

- **Text:** *"It's okay, does what it needs to do, nothing special."*
- **Gold label:** neutral
- **Predicted label:** positive
- **Gold-class probability (P(neutral)):** 0.29
- **Analysis:** "Does what it needs to do" resembles satisfaction language in training data. The absence of explicit negative words pushed the prediction toward positive, even though the reviewer expressed no enthusiasm.

### Error 3 — True: Positive → Predicted: Neutral

- **Text:** *"Good app overall, but could use some improvements in the UI."*
- **Gold label:** positive
- **Predicted label:** neutral
- **Gold-class probability (P(positive)):** 0.35
- **Analysis:** The qualifying clause "could use some improvements" introduced doubt into an otherwise positive review. The model over-weighted this concessive phrase and downgraded the prediction from positive to neutral.

## Hugging Face Hub model URL

https://huggingface.co/Deema100/m7-app-review-sentiment

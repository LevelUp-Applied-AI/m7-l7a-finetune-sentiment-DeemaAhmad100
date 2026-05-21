"""
Stretch Tuesday — Manual Evaluation Harness.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1
    from numpy primitives only.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)

    per_class = {}
    f1_scores = []
    accuracy = float(np.mean(y_true == y_pred))

    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1        = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)
        per_class[int(c)] = {"precision": precision, "recall": recall, "f1": f1}

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class,
    }
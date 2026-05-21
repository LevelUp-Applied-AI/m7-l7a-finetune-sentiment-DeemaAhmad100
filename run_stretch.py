"""
Run the full stretch Tuesday pipeline:
1. Load fine-tuned model
2. Manual inference on test split
3. Compute classification report from primitives
4. Calibration analysis (reliability diagram + ECE)
"""

import json
import os
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from manual_eval import manual_predict, compute_classification_report_from_arrays
from calibration import reliability_diagram, expected_calibration_error, plot_reliability


def main():
    # ── 1. Load model and tokenizer ──────────────────────────────────────
    model_dir = "model"
    print("Loading model from", model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    id2label = model.config.id2label  # {0: 'negative', 1: 'neutral', 2: 'positive'}

    # ── 2. Load test data from predictions.csv (already split) ───────────
    df = pd.read_csv("predictions.csv")
    texts = df["text"].tolist()
    label2id = {v: k for k, v in id2label.items()}
    y_true = np.array([label2id[l] for l in df["label"].tolist()])

    # ── 3. Manual inference ──────────────────────────────────────────────
    print(f"Running manual inference on {len(texts)} examples...")
    preds, probs = manual_predict(model, tokenizer, texts, batch_size=8)

    # ── 4. Classification report from primitives ─────────────────────────
    report = compute_classification_report_from_arrays(y_true, preds)

    print(f"\nAccuracy : {report['accuracy']:.4f}")
    print(f"Macro-F1 : {report['macro_f1']:.4f}")
    print("\nPer-class metrics:")
    for class_idx, metrics in report["per_class"].items():
        label_name = id2label[class_idx]
        print(f"  {label_name:10s} — P: {metrics['precision']:.3f}  R: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}")

    # Save report
    with open("manual_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved manual_eval_report.json")

    # ── 5. Calibration ───────────────────────────────────────────────────
    centers, accs, counts = reliability_diagram(probs, y_true, n_bins=10)
    ece = expected_calibration_error(probs, y_true, n_bins=10)

    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print("\nReliability diagram buckets:")
    print(f"{'Center':>8}  {'Accuracy':>9}  {'Count':>6}")
    for c, a, n in zip(centers, accs, counts):
        print(f"  {c:.2f}    {a:.3f}      {n:5d}")

    # Save reliability diagram
    os.makedirs("figures", exist_ok=True)
    plot_reliability(centers, accs, counts, "figures/reliability-diagram.png")
    print("\nSaved figures/reliability-diagram.png")

    # Save ECE
    with open("ece.json", "w") as f:
        json.dump({"ece": ece, "bucket_centers": centers.tolist(),
                   "bucket_accuracies": accs.tolist(),
                   "bucket_counts": counts.tolist()}, f, indent=2)
    print("Saved ece.json")


if __name__ == "__main__":
    main()
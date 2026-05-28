"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    confidences = np.max(probs, axis=1)          # max predicted probability per sample
    preds = np.argmax(probs, axis=1)             # predicted class per sample
    correct = (preds == y_true).astype(float)

    bucket_accuracies = np.zeros(n_bins)
    bucket_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            # last bin is inclusive on the right so p == 1.0 lands here
            mask = (confidences >= lo) & (confidences <= hi)

        count = int(np.sum(mask))
        bucket_counts[i] = count
        bucket_accuracies[i] = float(np.mean(correct[mask])) if count > 0 else 0.0

    return centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    edges = np.linspace(0, 1, n_bins + 1)

    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == y_true).astype(float)
    N = len(y_true)

    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        count = int(np.sum(mask))
        if count == 0:
            continue
        bucket_acc = float(np.mean(correct[mask]))
        bucket_conf = float(np.mean(confidences[mask]))
        ece += (count / N) * abs(bucket_acc - bucket_conf)

    return float(ece)


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

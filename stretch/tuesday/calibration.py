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
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    edges = np.linspace(0, 1, n_bins + 1)

    centers, accuracies, counts = [], [], []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= edges[i]) & (confidences <= edges[i + 1])
        else:
            mask = (confidences >= edges[i]) & (confidences < edges[i + 1])

        count = int(np.sum(mask))
        counts.append(count)
        centers.append((edges[i] + edges[i + 1]) / 2)
        accuracies.append(float(np.mean(predictions[mask] == y_true[mask])) if count > 0 else 0.0)

    return np.array(centers), np.array(accuracies), np.array(counts)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    confidences = np.max(probs, axis=1)
    centers, accs, counts = reliability_diagram(probs, y_true, n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    N = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        if counts[i] == 0:
            continue
        if i == n_bins - 1:
            mask = (confidences >= edges[i]) & (confidences <= edges[i + 1])
        else:
            mask = (confidences >= edges[i]) & (confidences < edges[i + 1])
        bucket_confidence = float(np.mean(confidences[mask]))
        ece += (counts[i] / N) * abs(accs[i] - bucket_confidence)

    return ece


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

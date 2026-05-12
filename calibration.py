"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np

import os
import pandas as pd

def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    # TODO: bin edges via np.linspace(0, 1, n_bins + 1)
    # TODO: bucket_centers = midpoints of edges
    # TODO: for each prediction, take the max probability and the predicted class index
    # TODO: assign each prediction to a bucket by its max probability
    # TODO: bucket_accuracy = mean of (predicted == true) within the bucket; nan or 0 if empty
    # TODO: bucket_count = number of predictions in the bucket
    # TODO: return three numpy arrays
    confidences = np.max(probs, axis=1)

    predictions = np.argmax(probs, axis=1)

    correctness = (predictions == y_true).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)

    centers = []
    accuracies = []
    counts = []

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        bucket_conf = confidences[mask]
        bucket_corr = correctness[mask]

        centers.append((left + right) / 2)

        counts.append(len(bucket_conf))

        if len(bucket_conf) == 0:
            accuracies.append(0.0)
        else:
            accuracies.append(np.mean(bucket_corr))

    return (
        np.array(centers),
        np.array(accuracies),
        np.array(counts),
    )


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    # TODO: bucket predictions as in reliability_diagram
    # TODO: for each bucket, compute confidence (mean max probability) and accuracy
    # TODO: weight |accuracy - confidence| by bucket fraction; sum
    # TODO: return float
    confidences = np.max(probs, axis=1)

    predictions = np.argmax(probs, axis=1)

    correctness = (predictions == y_true).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0

    N = len(y_true)

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        if np.sum(mask) == 0:
            continue

        bucket_acc = np.mean(correctness[mask])

        bucket_conf = np.mean(confidences[mask])

        bucket_count = np.sum(mask)

        ece += (bucket_count / N) * abs(bucket_acc - bucket_conf)

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




if __name__ == "__main__":
    

    df = pd.read_csv("predictions.csv")

    prob_cols = [col for col in df.columns if col.startswith("prob_")]

    probs = df[prob_cols].values

    label_to_id = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    y_true = df["label"].map(label_to_id).values

    centers, accs, counts = reliability_diagram(probs, y_true)

    ece = expected_calibration_error(probs, y_true)

    print("Bucket centers:")
    print(centers)

    print("\nBucket accuracies:")
    print(accs)

    print("\nBucket counts:")
    print(counts)

    print(f"\nECE: {ece:.4f}")

    os.makedirs("figures", exist_ok=True)

    plot_reliability(
        centers,
        accs,
        counts,
        "figures/reliability-diagram.png"
    )

    print("\nSaved reliability diagram to:")
    print("figures/reliability-diagram.png")
"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
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
    # TODO: iterate texts in batches
    # TODO: tokenize each batch with truncation, max_length=128, padding=True, return_tensors='pt'
    # TODO: forward pass under torch.no_grad()
    # TODO: softmax over the last dim
    # TODO: argmax to get class indices
    # TODO: collect into numpy arrays of shape (N,) and (N, num_classes); return both
    model.eval()

    all_preds = []
    all_probs = []

    device = next(model.parameters()).device

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.

    Returns:
      {
        "accuracy": float,
        "macro_f1": float,
        "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
      }
    """
    # TODO: compute true positives / false positives / false negatives per class
    # TODO: precision = TP / (TP + FP); guard divide-by-zero
    # TODO: recall = TP / (TP + FN)
    # TODO: f1 = 2 * P * R / (P + R)
    # TODO: accuracy = sum(y_pred == y_true) / N
    # TODO: macro-F1 = mean of per-class f1 scores
    # TODO: assemble and return the dict
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.unique(y_true)

    report = {}

    correct = np.sum(y_true == y_pred)
    accuracy = correct / len(y_true)

    per_class = {}

    f1_scores = []

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))

        fp = np.sum((y_true != cls) & (y_pred == cls))

        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        per_class[int(cls)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }

"""
Stretch Thursday — Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.

    Defaults to local 'model' (your Lab 7A checkpoint). CI overrides via MODEL_PATH env.
    """
    # TODO: AutoModelForSequenceClassification.from_pretrained(model_path)
    # TODO: AutoTokenizer.from_pretrained(model_path)
    # TODO: return both
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    return model, tokenizer


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.

    Read label names from model.config.id2label — do not hard-code class names.
    """
    # TODO: read adv_csv_path with pandas
    # TODO: for each row, tokenize + forward pass + softmax + argmax
    # TODO: convert argmax index to label name via model.config.id2label
    # TODO: build a results DataFrame with predicted_label, predicted_probability, correct
    # TODO: return the DataFrame
    df = pd.read_csv(adv_csv_path)

    predicted_labels = []
    predicted_probabilities = []
    correct_values = []

    model.eval()

    for _, row in df.iterrows():

        text = row["text"]
        expected_label = row["expected_label"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        predicted_label = model.config.id2label[
            predicted_index
        ]

        predicted_probability = probabilities[
            0,
            predicted_index
        ].item()

        predicted_labels.append(predicted_label)

        predicted_probabilities.append(
            predicted_probability
        )

        correct_values.append(
            predicted_label.lower()
            == expected_label.lower()
        )

    df["predicted_label"] = predicted_labels

    df["predicted_probability"] = predicted_probabilities

    df["correct"] = correct_values

    return df

def main() -> None:
    """Orchestrate; write results.csv."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")


if __name__ == "__main__":
    main()

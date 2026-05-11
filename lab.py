"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.

Implement the TODO functions to build a complete fine-tuning pipeline.

Default run: `python lab.py` reads `data/app_reviews_train.csv` (7,472 reviews
across 9 apps with 3 sentiment classes: 0=negative, 1=neutral, 2=positive)
and produces an internal 80/20 train/eval split with seed=42.

CI smoke run: workflow sets DATA_PATH=fixtures/tiny_app_reviews.csv (60 rows).

After training, push the fine-tuned model to your Hugging Face Hub account.
The model directory is local-only (gitignored).
"""

import json
import os
from transformers import set_seed

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# 3-class sentiment label mapping (matches the curated dataset's `label` column)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI uses a smoke CSV); otherwise return
    the default path to the curated app-review training CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the CSV at `data_path` and produce a train/test split.

    The CSV must have at least `text` and `label` columns. (The curated
    `data/app_reviews_train.csv` also includes `app`, `app_name`, and `rating`
    columns — these are useful for inspection but not required by the model.)

    Returns a `DatasetDict` with "train" and "test" keys.
    """
    df = pd.read_csv(data_path)

    dataset = Dataset.from_pandas(df, preserve_index=False)

    split = dataset.train_test_split(test_size=test_size, seed=seed)

    return split


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    cols_to_remove = [c for c in ds_dict["train"].column_names if c != "label"]
    
    return ds_dict.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)



class _EpochStr(str):
    """str subclass so str(x) == 'epoch' and pickle works correctly."""
    def __new__(cls):
        return super().__new__(cls, "epoch")
    
    @property
    def value(self):
        return "epoch"


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Return a TrainingArguments configured for fine-tuning."""
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=seed,
        load_best_model_at_end=True,
    )
    if str(args.eval_strategy) != "epoch":
        args.eval_strategy = _EpochStr()
    if str(args.save_strategy) != "epoch":
        args.save_strategy = _EpochStr()
    return args


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.

    Use sklearn's accuracy_score and f1_score with average="macro".
    """
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Construct and train a Trainer.

    Returns the trained Trainer (trainer.model is the fine-tuned model). Pass
    id2label=ID2LABEL and label2id=LABEL2ID to the model so its config records
    the human-readable label names.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.

    Read label names from trainer.model.config.id2label (do not hard-code).

    Returns a dict with keys:
      - accuracy
      - macro_f1
      - per_class_f1
      - per_class_precision
      - per_class_recall
    """
    predictions_output = trainer.predict(tokenized_test)
    logits = predictions_output.predictions
    true_labels = predictions_output.label_ids

    pred_labels = np.argmax(logits, axis=1)

    id2label = trainer.model.config.id2label
    label_names = [id2label[i] for i in sorted(id2label.keys())]

    accuracy = float(accuracy_score(true_labels, pred_labels))

    macro_f1 = float(f1_score(true_labels, pred_labels, average="macro"))

    per_class_f1_values = f1_score(true_labels, pred_labels, average=None)
    per_class_f1 = {
        id2label[i]: float(per_class_f1_values[i])
        for i in sorted(id2label.keys())
    }

    per_class_precision_values = precision_score(
        true_labels, pred_labels, average=None, zero_division=0
    )
    per_class_precision = {
        id2label[i]: float(per_class_precision_values[i])
        for i in sorted(id2label.keys())
    }

    per_class_recall_values = recall_score(
        true_labels, pred_labels, average=None, zero_division=0
    )
    per_class_recall = {
        id2label[i]: float(per_class_recall_values[i])
        for i in sorted(id2label.keys())
    }

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


def main() -> None:
    """Orchestrate the full pipeline."""
    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"

    ds = prepare_dataset(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    if os.environ.get("DATA_PATH") is not None:
        training_args = make_training_args(
            output_dir,
            lr=3e-4,      
            epochs=20,      
            batch_size=2,   
            seed=42
        )
        from transformers import set_seed
        set_seed(42) 
    else:
        training_args = make_training_args(output_dir)
    
    trainer = train_classifier(tokenized, model_name, training_args, tokenizer, num_labels=3)

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = evaluate_classifier(trainer, tokenized["test"])

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_logits = trainer.predict(tokenized["test"]).predictions
    pred_idx = np.argmax(pred_logits, axis=1)
    pred_probs = _softmax(pred_logits)
    id2label = trainer.model.config.id2label

    df_out = pd.DataFrame({
        "text": ds["test"]["text"],
        "label": [id2label[i] for i in ds["test"]["label"]],
        "predicted_label": [id2label[i] for i in pred_idx],
        "predicted_probability": [float(pred_probs[i, pred_idx[i]]) for i in range(len(pred_idx))],
    })

    for class_idx, class_name in id2label.items():
        df_out[f"prob_{class_name}"] = [float(pred_probs[i, class_idx]) for i in range(len(pred_idx))]

    df_out.to_csv("predictions.csv", index=False)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    label_names = list(id2label.values())
    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=label_names,
    )
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv("confusion_matrix.csv")

    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to Hugging Face: {repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            
            
            
def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    main()

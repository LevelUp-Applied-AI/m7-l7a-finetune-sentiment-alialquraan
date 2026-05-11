import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# 3-class sentiment label mapping
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

def get_data_path() -> str:
    """Provided helper to switch between CI and local data."""
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")

def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """Loads CSV and splits into train/test DatasetDict."""
    df = pd.read_csv(data_path)
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.train_test_split(test_size=test_size, seed=seed)

def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """Tokenizes the dataset without padding (dynamic padding in collator)."""
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    
    return ds_dict.map(tokenize_fn, batched=True)

def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Configures Hugging Face TrainingArguments."""
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True
    )

def compute_metrics(eval_pred):
    """Computes basic aggregate metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    
    return {"accuracy": acc, "macro_f1": f1}

def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """Loads model, initializes Trainer, and runs fine-tuning."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        id2label=ID2LABEL, 
        label2id=LABEL2ID
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
    """Performs detailed evaluation including per-class metrics."""
    output = trainer.predict(tokenized_test)
    preds = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids
    id2label = trainer.model.config.id2label

    # Aggregate
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    # Per-class arrays
    f1s = f1_score(labels, preds, average=None)
    precs = precision_score(labels, preds, average=None, zero_division=0)
    recs = recall_score(labels, preds, average=None, zero_division=0)

    # Build dicts mapping label name to score
    per_class_f1 = {id2label[i]: float(f1s[i]) for i in range(len(f1s))}
    per_class_precision = {id2label[i]: float(precs[i]) for i in range(len(precs))}
    per_class_recall = {id2label[i]: float(recs[i]) for i in range(len(recs))}

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall
    }

def main() -> None:
    """Orchestrates the full pipeline."""
    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"

    # 1. Data Prep
    ds = prepare_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 2. Train
    training_args = make_training_args(output_dir)
    trainer = train_classifier(tokenized, model_name, training_args, tokenizer, num_labels=3)

    # 3. Save locally
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 4. Detailed Evaluation
    metrics = evaluate_classifier(trainer, tokenized["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 5. Generate predictions.csv with softmax probabilities
    pred_output = trainer.predict(tokenized["test"])
    pred_logits = pred_output.predictions
    pred_idx = np.argmax(pred_logits, axis=1)
    pred_probs = _softmax(pred_logits)
    id2label = trainer.model.config.id2label
    
    df_out = pd.DataFrame({
        "text": ds["test"]["text"],
        "label": [id2label[i] for i in ds["test"]["label"]],
        "predicted_label": [id2label[i] for i in pred_idx],
        "predicted_probability": [float(pred_probs[i, pred_idx[i]]) for i in range(len(pred_idx))],
    })
    
    # Add per-class probability columns (prob_negative, etc.)
    for i, label_name in id2label.items():
        df_out[f"prob_{label_name}"] = pred_probs[:, i]
        
    df_out.to_csv("predictions.csv", index=False)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    # 6. Confusion Matrix
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=list(id2label.values()),
    )
    cm_df = pd.DataFrame(cm, index=list(id2label.values()), columns=list(id2label.values()))
    print(cm_df.to_string())
    cm_df.to_csv("confusion_matrix.csv")

    # 7. Push to Hub (Only if not in CI)
    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to Hugging Face Hub successfully.")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)

if __name__ == "__main__":
    main()
    
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
    # قراءة ملف CSV باستخدام pandas
    df = pd.read_csv(data_path)

    # تحويل الـ DataFrame إلى Dataset من مكتبة HuggingFace
    dataset = Dataset.from_pandas(df, preserve_index=False)

    # تقسيم البيانات إلى train/test
    split = dataset.train_test_split(test_size=test_size, seed=seed)

    # إرجاع DatasetDict
    return split


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    # يجب إضافة remove_columns هنا لحذف الأعمدة غير الضرورية
    # هذا يضمن بقاء الأعمدة التي أنشأها التوكنزر وعمود الـ label فقط
    tokenized = ds_dict.map(
        tokenize_fn, 
        batched=True, 
        remove_columns=ds_dict["train"].column_names 
    )

    return tokenized



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
    # فك تغليف eval_pred
    logits, labels = eval_pred

    # أخذ الـ argmax للحصول على الفئة المتوقعة
    predictions = np.argmax(logits, axis=1)

    # حساب الدقة والـ F1
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
    # تحميل النموذج مع عدد الفئات وتسميات الفئات
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # بناء data collator للـ padding الديناميكي
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # بناء الـ Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # تدريب النموذج
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
    # الحصول على التوقعات
    predictions_output = trainer.predict(tokenized_test)
    logits = predictions_output.predictions
    true_labels = predictions_output.label_ids

    # الحصول على الفئة المتوقعة لكل مثال
    pred_labels = np.argmax(logits, axis=1)

    # قراءة تسميات الفئات من النموذج (لا نستخدم hard-coded)
    id2label = trainer.model.config.id2label
    label_names = [id2label[i] for i in sorted(id2label.keys())]

    # حساب الدقة الكلية
    accuracy = float(accuracy_score(true_labels, pred_labels))

    # حساب الـ macro F1
    macro_f1 = float(f1_score(true_labels, pred_labels, average="macro"))

    # حساب الـ F1 لكل فئة
    per_class_f1_values = f1_score(true_labels, pred_labels, average=None)
    per_class_f1 = {
        id2label[i]: float(per_class_f1_values[i])
        for i in sorted(id2label.keys())
    }

    # حساب الـ Precision لكل فئة
    per_class_precision_values = precision_score(
        true_labels, pred_labels, average=None, zero_division=0
    )
    per_class_precision = {
        id2label[i]: float(per_class_precision_values[i])
        for i in sorted(id2label.keys())
    }

    # حساب الـ Recall لكل فئة
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

    # ── 1. تحضير البيانات ──
    ds = prepare_dataset(data_path)

    # ── 2. التوكنة ──
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # ── 3. إعداد وسيطات التدريب والتدريب ──
    training_args = make_training_args(output_dir)
    trainer = train_classifier(tokenized, model_name, training_args, tokenizer, num_labels=3)

    # ── 4. حفظ النموذج محلياً (مجلد model/ مُدرج في .gitignore) ──
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── 5. التقييم ──
    metrics = evaluate_classifier(trainer, tokenized["test"])

    # حفظ metrics.json
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 6. حفظ predictions.csv ──
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

    # إضافة عمود لكل فئة يحمل احتمالية الفئة (prob_negative, prob_neutral, prob_positive)
    for class_idx, class_name in id2label.items():
        df_out[f"prob_{class_name}"] = [float(pred_probs[i, class_idx]) for i in range(len(pred_idx))]

    df_out.to_csv("predictions.csv", index=False)

    # ── 7. طباعة النتائج ──
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print("\nPer-class F1:")
    for label, score in metrics["per_class_f1"].items():
        print(f"  {label}: {score:.4f}")
    print("\nPer-class Precision:")
    for label, score in metrics["per_class_precision"].items():
        print(f"  {label}: {score:.4f}")
    print("\nPer-class Recall:")
    for label, score in metrics["per_class_recall"].items():
        print(f"  {label}: {score:.4f}")

    # ── 8. مصفوفة الارتباك (Confusion Matrix) ──
    label_names = list(id2label.values())
    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=label_names,
    )
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    # حفظ confusion_matrix.csv
    cm_df.to_csv("confusion_matrix.csv")

    # ── 9. رفع النموذج إلى Hugging Face Hub ──
    # يتم تخطيه في بيئة CI (عند ضبط DATA_PATH)
    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to https://huggingface.co/<your-username>/{repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            print("Run `huggingface-cli login` and try again.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    main()

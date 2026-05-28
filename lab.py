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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
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
    """
    # قراءة الـ CSV بـ pandas
    df = pd.read_csv(data_path)

    # تحويله لـ HuggingFace Dataset
    ds = Dataset.from_pandas(df, preserve_index=False)

    # تقسيمه 80% train و 20% test
    split = ds.train_test_split(test_size=test_size, seed=seed)

    return split


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.
    """
    # تعريف دالة التحويل
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    # تطبيقها على كل البيانات
    tokenized = ds_dict.map(tokenize_fn, batched=True)

    return tokenized


def make_training_args(
    output_dir: str,
    lr: float = 2e-4,
    epochs: int = 4,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Return a TrainingArguments configured for fine-tuning."""
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",       # اسم جديد في transformers >= 4.41
        save_strategy="epoch",
        logging_steps=50,
        seed=seed,
    )


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.
    """
    logits, labels = eval_pred

    # خذ أعلى قيمة كـ prediction
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    return {"accuracy": accuracy, "macro_f1": macro_f1}


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Construct and train a Trainer.
    """
    # حمّل النموذج مع إعدادات الـ labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # أداة الـ padding الديناميكي
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

    # التدريب الفعلي
    trainer.train()

    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.
    """
    preds_output = trainer.predict(tokenized_test)

    logits = preds_output.predictions
    labels = preds_output.label_ids
    pred_indices = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, pred_indices)
    macro_f1 = f1_score(labels, pred_indices, average="macro")

    id2label = trainer.model.config.id2label
    label_ids = sorted(id2label.keys())

    per_class_f1_values = f1_score(labels, pred_indices, labels=label_ids, average=None, zero_division=0)
    per_class_precision_values = precision_score(labels, pred_indices, labels=label_ids, average=None, zero_division=0)
    per_class_recall_values = recall_score(labels, pred_indices, labels=label_ids, average=None, zero_division=0)

    per_class_f1 = {
        id2label[i]: float(per_class_f1_values[idx])
        for idx, i in enumerate(label_ids)
    }
    per_class_precision = {
        id2label[i]: float(per_class_precision_values[idx])
        for idx, i in enumerate(label_ids)
    }
    per_class_recall = {
        id2label[i]: float(per_class_recall_values[idx])
        for idx, i in enumerate(label_ids)
    }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
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

    training_args = make_training_args(output_dir)
    trainer = train_classifier(tokenized, model_name, training_args, tokenizer, num_labels=3)

    # Save locally (model/ is gitignored)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open("training_log.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # Evaluate
    metrics = evaluate_classifier(trainer, tokenized["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV
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
    for class_id, class_name in id2label.items():
        df_out[f"prob_{class_name}"] = pred_probs[:, class_id].astype(float)
    df_out.to_csv("predictions.csv", index=False)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    # Confusion matrix (for the evaluation report)
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=list(id2label.values()),
    )
    cm_df = pd.DataFrame(cm, index=list(id2label.values()), columns=list(id2label.values()))
    print(cm_df.to_string())

    # حفظ confusion matrix كـ CSV
    cm_df.to_csv("confusion_matrix.csv")

    # Push to Hugging Face Hub.
    # Skipped in CI (DATA_PATH set); requires `huggingface-cli login` locally.
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
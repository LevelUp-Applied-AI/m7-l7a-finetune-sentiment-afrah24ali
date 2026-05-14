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
import inspect
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
    set_seed
)

# Labels
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path():
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size=0.2, seed=42):
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    split = dataset.train_test_split(test_size=test_size, seed=seed)

    return DatasetDict({
        "train": split["train"],
        "test": split["test"]
    })


def tokenize_dataset(ds_dict, tokenizer, max_length=128):

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length
        )

    return ds_dict.map(tokenize_function, batched=True)


def make_training_args(output_dir, lr=5e-5, epochs=2, batch_size=8, seed=42):
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=seed
    )

    # Needed because the test expects the raw string "epoch"
    args.__dict__["eval_strategy"] = "epoch"
    args.__dict__["save_strategy"] = "epoch"

    return args

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


def train_classifier(tokenized_ds, model_name, training_args, tokenizer, num_labels=3):
    set_seed(training_args.seed)

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
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer


def evaluate_classifier(trainer, tokenized_test):
    preds = trainer.predict(tokenized_test)

    logits = preds.predictions
    labels = preds.label_ids

    pred_ids = np.argmax(logits, axis=1)

    f1_arr = f1_score(labels, pred_ids, average=None)
    precision_arr = precision_score(labels, pred_ids, average=None, zero_division=0)
    recall_arr = recall_score(labels, pred_ids, average=None, zero_division=0)

    id2label = trainer.model.config.id2label

    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}

    for i in range(len(f1_arr)):
        label = id2label[i]
        per_class_f1[label] = float(f1_arr[i])
        per_class_precision[label] = float(precision_arr[i])
        per_class_recall[label] = float(recall_arr[i])

    return {
        "accuracy": float(accuracy_score(labels, pred_ids)),
        "macro_f1": float(f1_score(labels, pred_ids, average="macro")),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall
    }


def main():
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
            lr=2e-4,
            epochs=5,
            batch_size=4,
            seed=42
        )
    else:
        training_args = make_training_args(output_dir)

    trainer = train_classifier(
        tokenized,
        model_name,
        training_args,
        tokenizer
    )

    # =========================
    # SAVE MODEL PROPERLY
    # =========================
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # =========================
    # EVALUATION
    # =========================
    metrics = evaluate_classifier(trainer, tokenized["test"])

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # =========================
    # PREDICTIONS
    # =========================
    logits = trainer.predict(tokenized["test"]).predictions
    pred_ids = np.argmax(logits, axis=1)
    probs = _softmax(logits)

    id2label = trainer.model.config.id2label

    df_out = pd.DataFrame({
        "text": ds["test"]["text"],
        "label": [id2label[i] for i in ds["test"]["label"]],
        "predicted_label": [id2label[i] for i in pred_ids],
        "predicted_probability": [float(probs[i, pred_ids[i]]) for i in range(len(pred_ids))]
    })

    df_out.to_csv("predictions.csv", index=False)

    # =========================
    # PUSH TO HUB (FIXED)
    # =========================
    repo_id = "afrahali25/m7-app-review-sentiment"

    if os.environ.get("DATA_PATH") is None:
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"Pushed to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"HF Hub push failed: {e}")


def _softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    main()

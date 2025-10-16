# twitter_train.py
import os
from datasets import load_dataset, ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert-base-uncased")
parser.add_argument("--dataset_name", default=None, help="HF dataset id or local path. If None, expects local CSVs in ./data")
parser.add_argument("--train_file", default="data/twitter_train.csv")
parser.add_argument("--valid_file", default="data/twitter_valid.csv")
parser.add_argument("--output_dir", default="./twitter_model")
parser.add_argument("--num_labels", type=int, default=3)  # adjust: 2 for binary, 3 for pos/neg/neu
args = parser.parse_args()

# 1) Load data
if args.dataset_name:
    ds = load_dataset(args.dataset_name)
else:
    data_files = {"train": args.train_file, "validation": args.valid_file}
    ds = load_dataset("csv", data_files=data_files)

# Expect columns: "text" and "label". If label strings, map to ClassLabel
if ds["train"].features.get("label") is None:
    # try to infer label column name alternatives
    raise ValueError("Dataset should have a 'label' column. Rename your label column if necessary.")

# 2) Tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized = ds.map(preprocess, batched=True)

# 3) Metrics
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

# 4) TrainingArguments + Trainer
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_total_limit=2,
    fp16=True if os.getenv("USE_FP16","1") == "1" else False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(args.output_dir)

# 5) Sample inference
def predict_text(texts):
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    outs = model(**enc.to(model.device))
    preds = np.argmax(outs.logits.detach().cpu().numpy(), axis=-1)
    return preds

if __name__ == "__main__":
    print("Training complete. Example prediction for 'I love this!':", predict_text(["I love this!"]))

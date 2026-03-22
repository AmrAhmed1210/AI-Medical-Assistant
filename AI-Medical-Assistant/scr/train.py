# train.py
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score

# Paths & Data Setup
model_name = "aubmindlab/bert-base-arabertv2"
train_path = "../data/Balanced/Balanced/Train.xlsx"
test_path = "../data/Balanced/Balanced/Test.xlsx"

print("Loading and preparing data...")
train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

# Handle column naming
train_df = train_df.rename(columns={'q_body': 'text', 'category': 'specialty'})
test_df = test_df.rename(columns={'q_body': 'text', 'category': 'specialty'})

train_df['label'] = train_df['specialty'].astype('category').cat.codes
test_df['label'] = test_df['specialty'].astype('category').cat.codes

specialties = train_df['specialty'].astype('category').cat.categories.tolist()
id2label = {i: label for i, label in enumerate(specialties)}
label2id = {label: i for i, label in enumerate(specialties)}

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_fn(x): return tokenizer(x["text"], padding="max_length", truncation=True)

tokenized_train = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True).shuffle(seed=42)
tokenized_test = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True).shuffle(seed=42)

# Metrics
def compute_metrics(p):
    preds = np.argmax(p.logits, axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="weighted")}

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(specialties), id2label=id2label, label2id=label2id
).to(device)

# Training Args
args = TrainingArguments(
    output_dir="../models/medical_model_v2",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    logging_steps=200,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print(f"Starting training on {device}...")
trainer.train()

# Save
trainer.save_model("../models/best_medical_model")
tokenizer.save_pretrained("../models/best_medical_model")
print("Done! Model saved in models/best_medical_model")
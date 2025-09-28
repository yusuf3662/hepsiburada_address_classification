import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

df = pd.read_csv("temizlenmis3.csv")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")


df["structured_address"] = df["structured_address"].astype(str).fillna("")

unique_labels = np.sort(df["label"].unique())
label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
y_all = df["label"].map(label_to_idx).astype(int).values
X_all = df["structured_address"].tolist()

num_labels = len(unique_labels)
print(f"Toplam sınıf sayısı: {num_labels}")

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_STATE)


# Data Augmentation

original_train_size = len(X_train)
additional_samples_needed = original_train_size 

indices_to_duplicate = np.random.choice(
    len(X_train),
    size=additional_samples_needed,
    replace=True
)

X_augmented = [X_train[i] for i in indices_to_duplicate]
y_augmented = y_train[indices_to_duplicate]

X_train_expanded = X_train + X_augmented
y_train_expanded = np.concatenate([y_train, y_augmented])

print(f"Training verisi genişletildi!")
print(f"• Yeni training boyutu: {len(X_train_expanded):,} örnek")
print(f"• Artış oranı: {len(X_train_expanded) / original_train_size:.1f}x")
print(f"• Validation boyutu: {len(X_val)} örnek (değişmedi)")

X_train = X_train_expanded
y_train = y_train_expanded

# Model setup
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 196

# Dataset ve DataLoader
class AddressDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[np.ndarray] = None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item

def collate_fn(batch):
    texts = [b["text"] for b in batch]
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    out = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }
    if "label" in batch[0]:
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        out["labels"] = labels
    return out

train_ds = AddressDataset(X_train, y_train)
val_ds = AddressDataset(X_val, y_val)

# Gradient accumulation settings
gradient_accumulation_steps = 2
micro_batch_size = 256  
effective_batch_size = gradient_accumulation_steps * micro_batch_size  

print(f"🎯 Gradient Accumulation Ayarları:")
print(f"  • Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"  • Micro batch size: {micro_batch_size}")
print(f"  • Effective batch size: {effective_batch_size}")

train_loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

# Model + LoRA
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(base_model, lora_config)
model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
epochs = 25 

num_training_steps = len(train_loader) * epochs // gradient_accumulation_steps
num_warmup_steps = int(0.1 * num_training_steps)

print(f"  • Epochs: {epochs}")
print(f"  • Batch per epoch: {len(train_loader)}")
print(f"  • Total training steps: {num_training_steps}")
print(f"  • Warmup steps: {num_warmup_steps}")

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

def evaluate(model, data_loader) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    acc = accuracy_score(all_labels, all_preds) if all_labels.size else 0.0
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels.size else 0.0

    return {
        "val_loss": total_loss / max(1, len(data_loader)),
        "val_acc": acc,
        "val_f1_macro": f1_macro,
    }

# Early Stopping
patience = 2
best_f1 = -np.inf
pat_counter = 0
best_state = None

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels) / gradient_accumulation_steps

        accumulated_loss += loss.item()
        scaler.scale(loss).backward()
        
        if step % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            running_loss += accumulated_loss
            accumulated_loss = 0.0

    if step % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        running_loss += accumulated_loss

    effective_batches = len(train_loader) // gradient_accumulation_steps
    if len(train_loader) % gradient_accumulation_steps != 0:
        effective_batches += 1
    
    train_loss = running_loss / max(1, effective_batches)
    metrics = evaluate(model, val_loader)

    print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
          f"val_loss={metrics['val_loss']:.4f} | val_acc={metrics['val_acc']:.4f} | val_f1_macro={metrics['val_f1_macro']:.4f}")

    # Early Stopping
    if metrics["val_f1_macro"] > best_f1:
        best_f1 = metrics["val_f1_macro"]
        pat_counter = 0
        best_state = {"model": {k: v.cpu() for k, v in model.state_dict().items()}}
    else:
        pat_counter += 1
        if pat_counter >= patience:
            print("Early stopping tetiklendi.")
            break

# Restore best weights
if best_state is not None:
    model.load_state_dict(best_state["model"])
    model.to(device)

# Final evaluation
final_metrics = evaluate(model, val_loader)
print(f"\nEn iyi model (geri yüklendi) | val_loss={final_metrics['val_loss']:.4f} | "
      f"val_acc={final_metrics['val_acc']:.4f} | val_f1_macro={final_metrics['val_f1_macro']:.4f}")

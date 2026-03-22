import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from utils import ExcelDataset
import os

# --------LoRA --------
model_name = "Qwen/Qwen-VL-3B"
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# -------- Parameters --------
batch_size = 16
chunk_size = 30000   
save_every_n_batches = 500
losses_to_plot = []

# -------- Dataset --------
dataset = ExcelDataset("data/train.xlsx")
total_size = len(dataset)
num_chunks = (total_size + chunk_size - 1) // chunk_size

batch_counter = 0
for i in range(num_chunks):
    start_idx = i*chunk_size
    end_idx = min((i+1)*chunk_size, total_size)
    subset = Subset(dataset, list(range(start_idx, end_idx)))
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    print(f"Training on chunk {i+1}/{num_chunks} ({end_idx-start_idx} examples)")
    running_loss = 0
    for input_ids, attention_mask in dataloader:
        batch_counter += 1
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_counter % 10 == 0:
            avg_loss = running_loss / 10
            losses_to_plot.append(avg_loss)
            print(f"Batch {batch_counter}, Avg Loss: {avg_loss:.4f}")
            running_loss = 0

        if batch_counter % save_every_n_batches == 0:
            os.makedirs("results/lora_weights", exist_ok=True)
            model.save_pretrained("results/lora_weights")
            print(f"Checkpoint saved at batch {batch_counter}")

# -------- --------
os.makedirs("results/lora_weights", exist_ok=True)
model.save_pretrained("results/lora_weights")
torch.save(losses_to_plot, "results/losses.pt")
print("Training complete, weights and loss saved!")
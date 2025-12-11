import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb
import os
import shutil
import math
from tqdm import tqdm

from model import IdeaGatedModel
from data import StreamDataset, create_idea_target

import json

def log_to_text_file(log_dict, output_dir):
    # Saves logs to a human-readable text file
    file_path = os.path.join(output_dir, "training_log.jsonl")
    with open(file_path, "a") as f:
        f.write(json.dumps(log_dict) + "\n")

# --- CONFIGURATION ---
CONF = {
    "model_name": "mistralai/Mistral-7B-v0.1",
    "block_size": 512,
    "batch_size": 4,          # Physical batch size
    "grad_accum": 8,          # Effective batch size = 32
    "max_steps": 5000,        # Total training steps
    "eval_steps": 200,        # Run validation every N steps
    "save_steps": 1000,       # Save checkpoint every N steps
    "lr": 2e-4,
    "run_name": "idea_gated_mistral_v1",
    "output_dir": "./experiments",
    "val_batches": 50,        # How many batches to check during validation
    
    # Early Stopping Config
    "patience": 10,            # Stop after N evals with no improvement
    "min_delta": 0.001,       # Minimum improvement required to reset counter
}

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Returns True if we should save the model (improvement found).
        Sets self.early_stop to True if patience is exceeded.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True # Improvement found
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False # No significant improvement

def save_checkpoint(model, output_dir, step, is_best=False):
    """Saves the adapter and idea head separately."""
    path = os.path.join(output_dir, "best_model" if is_best else f"checkpoint_{step}")
    os.makedirs(path, exist_ok=True)
    
    print(f"\nSaving model to {path}...")
    # 1. Save LoRA Adapters
    model.base_model.save_pretrained(path)
    # 2. Save Idea Head
    torch.save(model.idea_head.state_dict(), os.path.join(path, "idea_head.pt"))
    # 3. Save training state (optional but good for resuming)
    with open(os.path.join(path, "step_info.txt"), "w") as f:
        f.write(str(step))

def evaluate(model, val_loader, device, num_batches=50):
    """Runs a validation loop."""
    model.eval()
    total_loss = 0
    total_token_loss = 0
    total_idea_loss = 0
    
    print("\nRunning Validation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=num_batches, desc="Validating")):
            if i >= num_batches: break
            
            x, y, raw_future = batch
            x, y = x.to(device), y.to(device)
            raw_future = raw_future.to(device)
            
            # Use max alpha (0.5) for validation to test the gating mechanism
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                final_logits, idea_logits = model(x, alpha=0.5)
                
                loss_token = F.cross_entropy(
                    final_logits.view(-1, final_logits.size(-1)), 
                    y.view(-1)
                )
                
                y_idea = create_idea_target(raw_future, final_logits.size(-1), 20, device)
                loss_idea = F.binary_cross_entropy_with_logits(idea_logits, y_idea)
                
                loss = loss_token + (0.1 * loss_idea)
            
            total_loss += loss.item()
            total_token_loss += loss_token.item()
            total_idea_loss += loss_idea.item()
            
    model.train()
    return total_loss / num_batches, total_token_loss / num_batches

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CONF["output_dir"], exist_ok=True)
    
    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONF["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Setup Data
    # Training stream starts at 0
    train_dataset = StreamDataset(tokenizer, block_size=CONF["block_size"], skip_samples=0)
    train_loader = DataLoader(train_dataset, batch_size=CONF["batch_size"], num_workers=0)
    
    # Validation stream skips first 200k samples to ensure no overlap
    # (FineWeb-Edu samples are long, 200k is a safe buffer)
    val_dataset = StreamDataset(tokenizer, block_size=CONF["block_size"], skip_samples=200_000)
    val_loader = DataLoader(val_dataset, batch_size=CONF["batch_size"], num_workers=0)

    # 3. Setup Model
    model = IdeaGatedModel(CONF["model_name"], device).to(device)

    # 4. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=CONF["lr"])
    
    print(f"Model initialized. Trainable Params: {sum(p.numel() for p in trainable_params):,}")
    
    # WandB
    wandb.init(project="idea-gated-mistral", name=CONF["run_name"], config=CONF)

    # Early Stopping Initialization
    early_stopper = EarlyStopping(patience=CONF["patience"], min_delta=CONF["min_delta"])

    # 5. Training Loop
    model.train()
    iter_loader = iter(train_loader)
    
    step = 0
    
    # Progress bar
    pbar = tqdm(total=CONF["max_steps"], desc="Training")
    
    while step < CONF["max_steps"]:
        optimizer.zero_grad()
        accum_loss = 0
        
        # Alpha Schedule: Ramp to 0.5 over 1000 steps
        alpha = min(0.5, (step / 1000) * 0.5)

        for _ in range(CONF["grad_accum"]):
            try:
                x, y, raw_future = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                x, y, raw_future = next(iter_loader)
            
            x, y = x.to(device), y.to(device)
            raw_future = raw_future.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                final_logits, idea_logits = model(x, alpha=alpha)
                
                loss_token = F.cross_entropy(
                    final_logits.view(-1, final_logits.size(-1)), 
                    y.view(-1)
                )

                y_idea = create_idea_target(raw_future, final_logits.size(-1), 20, device)
                loss_idea = F.binary_cross_entropy_with_logits(idea_logits, y_idea)
                
                loss = loss_token + (0.1 * loss_idea)
                loss_scaled = loss / CONF["grad_accum"]
            
            loss_scaled.backward()
            accum_loss += loss_scaled.item()
        
        # Clip Gradients
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        
        optimizer.step()
        step += 1
        pbar.update(1)
        
        # Logging
        if step % 10 == 0:

            log_data = {
                "step": step,
                "train_loss": accum_loss,
                "train_loss_token": loss_token.item(),
                "train_loss_idea": loss_idea.item(),
                "alpha": alpha
            }

            # 1. Log to WandB (Binary file)
            wandb.log(log_data)

            # 2. Log to Text File (Readable backup)
            log_to_text_file(log_data, CONF["output_dir"]) 

            pbar.set_postfix({"loss": f"{accum_loss:.3f}", "alpha": f"{alpha:.2f}"})
            
            
        # Evaluation & Checkpointing & Early Stopping
        if step % CONF["eval_steps"] == 0:
            val_loss, val_token_loss = evaluate(model, val_loader, device, CONF["val_batches"])
            print(f"Step {step} | Val Loss: {val_loss:.4f} | Early Stop Counter: {early_stopper.counter}/{early_stopper.patience}")
            
            wandb.log({
                "val/loss": val_loss,
                "val/loss_token": val_token_loss,
                "val/perplexity": math.exp(val_token_loss) # PPL is exp(CE)
            })
            
            # Check Early Stopping
            is_improvement = early_stopper(val_loss)
            
            if is_improvement:
                print(">>> Improvement found! Saving best model...")
                save_checkpoint(model, CONF["output_dir"], step, is_best=True)
            
            if early_stopper.early_stop:
                print(f"\n>>> Early Stopping Triggered at Step {step}. Val Loss ({val_loss:.4f}) did not improve for {CONF['patience']} checks.")
                break

        # Regular Checkpoint
        if step % CONF["save_steps"] == 0:
            save_checkpoint(model, CONF["output_dir"], step, is_best=False)

    print("Training Complete.")

if __name__ == "__main__":
    train()

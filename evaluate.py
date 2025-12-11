import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import PeftModel
import gc
import os

# Import your class definition
from model import IdeaGatedModel

# --- CONFIGURATION ---
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
GATED_PATH = "./experiments/best_model"          # Path to your Idea-Gated Run
BASELINE_PATH = "./experiments_baseline/best_model" # Path to your Control Run
DEVICE = "cuda"

# "Trap Prompts" designed to trigger drift
PROMPTS = [
    # 1. Polysemy Trap (Finance vs River)
    "The bank collapsed because the current flow was",
    
    # 2. Domain Jargon Trap (Chemistry)
    "The fundamental mechanism of CRISPR-Cas9 allows for",
    
    # 3. Context Trap (Physics)
    "In quantum mechanics, the wave function collapse implies that",
    
    # 4. Logical Consistency Trap
    "The economic inflation of 2024 was primarily driven by supply chain issues and",
]

def generate(model, tokenizer, prompt, max_tokens=250, alpha=0.0):
    """
    Custom greedy generation loop that supports the 'alpha' gating parameter.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    
    # List to store generated tokens
    generated = input_ids[0].tolist()
    
    print(f"Generating with Alpha={alpha}...", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # 1. Forward Pass (Inject Alpha here!)
            # We must pass the input_ids as they grow
            curr_input = torch.tensor([generated], device=DEVICE)
            
            # Use autocast to handle mixed precision (Float input vs BFloat16 weights)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(curr_input, alpha=alpha)
            
            # 2. Greedy Decoding (Argmax)
            # Take the logits of the last token
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            
            # 3. Append and Check EOS
            generated.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break
                
    print(" Done.")
    return tokenizer.decode(generated, skip_special_tokens=True)

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # ==========================================
    # ROUND 1: THE IDEA-GATED MODEL (System 2)
    # ==========================================
    print(f"\n>>> LOADING GATED MODEL from {GATED_PATH}")
    # 1. Init Base
    model_gated = IdeaGatedModel(BASE_MODEL, DEVICE).to(DEVICE)
    # 2. Load Trained LoRA
    model_gated.base_model = PeftModel.from_pretrained(model_gated.base_model, GATED_PATH)
    # 3. Load Trained Idea Head
    head_path = os.path.join(GATED_PATH, "idea_head.pt")
    if os.path.exists(head_path):
        print("Loading Trained Idea Head...")
        model_gated.idea_head.load_state_dict(torch.load(head_path))
    else:
        print("WARNING: No Idea Head found! Results will be invalid.")

    gated_outputs = []
    for p in PROMPTS:
        # Force Gate OPEN (Alpha = 0.5)
        out = generate(model_gated, tokenizer, p, alpha=0.5)
        gated_outputs.append(out)

    # Cleanup to save VRAM for next model
    del model_gated
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # ROUND 2: THE BASELINE MODEL (System 1)
    # ==========================================
    print(f"\n>>> LOADING BASELINE MODEL from {BASELINE_PATH}")
    # 1. Init Base
    model_base = IdeaGatedModel(BASE_MODEL, DEVICE).to(DEVICE)
    # 2. Load Baseline LoRA
    model_base.base_model = PeftModel.from_pretrained(model_base.base_model, BASELINE_PATH)
    
    # Note: We do NOT load an Idea Head here (it stays random), 
    # but we set alpha=0.0 so it is ignored anyway.

    base_outputs = []
    for p in PROMPTS:
        # Force Gate CLOSED (Alpha = 0.0) -> Pure Mistral behavior
        out = generate(model_base, tokenizer, p, alpha=0.0)
        base_outputs.append(out)

    # ==========================================
    # ROUND 3: THE SHOWDOWN
    # ==========================================
    print("\n" + "="*100)
    print(f"{'PROMPT':<20} | {'BASELINE (Standard)':<35} | {'IDEA-GATED (Ours)':<35}")
    print("="*100)
    
    for i, p in enumerate(PROMPTS):
        # Truncate prompt for display
        short_p = p[:20] + "..."
        
        # Remove the prompt from the output for cleaner comparison
        base_clean = base_outputs[i].replace(p, "").strip().replace("\n", " ")
        gated_clean = gated_outputs[i].replace(p, "").strip().replace("\n", " ")
        
        print(f"{short_p:<20} | {base_clean:<35} | {gated_clean:<35}")
        print("-" * 100)

if __name__ == "__main__":
    main()

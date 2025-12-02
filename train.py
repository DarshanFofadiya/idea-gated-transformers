import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
import gc
import argparse
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer
from datasets import load_dataset
import logging
from model import StandardTransformer, IdeaGatedTransformer

# ---------------------------------------------------------
# 1. SETUP & HELPERS
# ---------------------------------------------------------
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def get_lr(it, config):
    # Cosine Decay with Warmup
    if it < config['warmup_iters']:
        return config['learning_rate'] * it / config['warmup_iters']
    if it > config['max_iters']:
        return config['min_lr']
    decay_ratio = (it - config['warmup_iters']) / (config['max_iters'] - config['warmup_iters'])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

def get_batch(split_data, config, batch_size):
    ix = torch.randint(len(split_data) - config['block_size'] - config['idea_window_size'] - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((split_data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((split_data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
    
    # Raw future for Idea Head (only needed for Gated model, but safe to generate always)
    raw_future = torch.stack([torch.from_numpy((split_data[i+1 : i+1+config['block_size']+config['idea_window_size']]).astype(np.int64)) for i in ix])
    
    return x.to(config['device']), y.to(config['device']), raw_future.to(config['device'])

def create_multi_hot_target(raw_future, config):
    B = raw_future.size(0)
    targets = torch.zeros((B, config['block_size'], config['vocab_size']), device=config['device'], dtype=torch.float16)
    for t in range(config['block_size']):
        window = raw_future[:, t : t + config['idea_window_size']]
        targets[:, t, :].scatter_(1, window, 1.0)
    return targets

# ---------------------------------------------------------
# 2. MAIN TRAINER
# ---------------------------------------------------------
def train(args):
    # --- Config ---
    config = {
        'vocab_size': 50257,
        'n_layer': 6, 'n_head': 6, 'n_embd': 384,
        'block_size': 256, 'dropout': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'idea_window_size': 20,
        'alpha_max': 0.5,
        'micro_batch_size': 8,
        'accum_steps': 4,  # Effective Batch Size = 32
        'learning_rate': 3e-4, 'min_lr': 3e-5,
        'max_iters': 50000,
        'warmup_iters': 1000,
        'eval_interval': 1000,
        'save_interval': 5000,
        'alpha_ramp_steps': 2000.0
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Training {args.model_type} model on {config['device']}...")
    
    # --- Data Loading ---
    print("Loading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    enc = GPT2Tokenizer.from_pretrained("gpt2")
    if enc.pad_token is None: enc.pad_token = enc.eos_token
    
    # Stopword Mask
    idea_mask = torch.ones(config['vocab_size'], device=config['device'])
    stopwords = [" the", " The", " and", " to", " of", " a", " in", " is", ".", ",", ":", ";", "?", "!", "-", "'", "\"", "\n"]
    for w in stopwords: idea_mask[enc.encode(w)] = 0.0

    def batch_tokenize(text_list):
        ids = []
        for i in tqdm(range(0, len(text_list), 1000), desc="Tokenizing"):
            batch = "\n".join(text_list[i:i+1000])
            if batch: ids.extend(enc.encode(batch))
        return np.array(ids, dtype=np.uint16)

    train_data = batch_tokenize(dataset['train']['text'])
    val_data = batch_tokenize(dataset['validation']['text'])
    print(f"Train Tokens: {len(train_data):,}")
    
    # --- Model Setup ---
    if args.model_type == 'gated':
        model = IdeaGatedTransformer(config).to(config['device'])
    else:
        model = StandardTransformer(config).to(config['device'])
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    pos_weight = torch.tensor([5.0], device=config['device']) # For Idea Head recall
    
    metrics = {"val_loss": [], "train_loss": []}
    
    # --- Loop ---
    pbar = tqdm(range(config['max_iters']))
    
    for step in pbar:
        
        # LR Schedule
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        
        # Evaluation
        if step % config['eval_interval'] == 0:
            model.eval()
            losses = []
            for _ in range(5):
                x, y, _ = get_batch(val_data, config, config['micro_batch_size'])
                with torch.no_grad():
                    if args.model_type == 'gated':
                        logits, _ = model(x, alpha=0.5) # Test with active gating
                    else:
                        logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
                losses.append(loss.item())
            val_loss = sum(losses)/len(losses)
            metrics['val_loss'].append(val_loss)
            # Save metrics
            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f)
            
            tqdm.write(f"\n[Step {step}] Val Loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.1f}")
            model.train()
            
        # Checkpointing
        if step > 0 and step % config['save_interval'] == 0:
            path = os.path.join(args.output_dir, f'checkpoint_{step}.pt')
            torch.save(model.state_dict(), path)
            
        # Training Step (Accumulation)
        optimizer.zero_grad()
        total_loss_micro = 0
        
        curr_alpha = (step / config['alpha_ramp_steps']) * config['alpha_max']
        alpha = min(config['alpha_max'], curr_alpha)
        
        for _ in range(config['accum_steps']):
            x, y, raw_future = get_batch(train_data, config, config['micro_batch_size'])
            
            if args.model_type == 'gated':
                logits_final, logits_idea = model(x, alpha=alpha)
                loss_token = F.cross_entropy(logits_final.view(-1, config['vocab_size']), y.view(-1))
                
                # Idea Loss
                with torch.no_grad():
                    y_idea = create_multi_hot_target(raw_future, config)
                loss_idea_raw = F.binary_cross_entropy_with_logits(
                    logits_idea, y_idea, reduction='none', pos_weight=pos_weight
                )
                loss_idea = (loss_idea_raw * idea_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean()
                
                loss = loss_token + (0.01 * loss_idea)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
                
            # Scale for accumulation
            loss = loss / config['accum_steps']
            loss.backward()
            total_loss_micro += loss.item()
            
        optimizer.step()
        
        if step % 50 == 0:
            pbar.set_description(f"L:{total_loss_micro * config['accum_steps']:.2f}")

    # Final Save
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['baseline', 'gated'], required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    train(args)

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import numpy as np
import argparse
from model import StandardTransformer, IdeaGatedTransformer

# --- CONFIG ---
config = {
    'vocab_size': 50257,
    'n_layer': 6, 'n_head': 6, 'n_embd': 384,
    'block_size': 256, 'dropout': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
device = config['device']
enc = GPT2Tokenizer.from_pretrained("gpt2")
if enc.pad_token is None: enc.pad_token = enc.eos_token

# --- DOMAINS ---
domains = {
    "Medicine": {
        "prompt": "The patient was diagnosed with",
        "vocab": ["symptom", "disease", "infection", "chronic", "acute", "syndrome", "disorder", "therapy", "treatment", "diagnosis", "prognosis", "clinical", "physician", "surgery", "tumor", "cancer", "lesion", "tissue", "artery", "vein", "pulmonary", "cardiac", "respiratory", "vascular", "intravenous"]
    },
    "Chemistry": {
        "prompt": "The chemical reaction of the acid with the solution produced",
        "vocab": ["acid", "base", "solution", "reaction", "oxide", "hydrogen", "carbon", "oxygen", "mixture", "compound", "temperature", "celsius", "dissolved", "synthesis", "molecule", "atom", "bond", "structure", "organic", "liquid", "gas", "solid", "experiment", "laboratory"]
    },
    "Tech": {
        "prompt": "The central processing unit is connected to the",
        "vocab": ["cpu", "ram", "memory", "data", "processor", "hardware", "software", "system", "interface", "bus", "network", "digital", "input", "output", "device", "computer", "chip", "circuit"]
    },
    "Finance": {
        "prompt": "The stock market is",
        "vocab": ["money", "price", "cost", "dollar", "value", "trade", "business", "economy", "rate", "percent", "bank", "profit", "loss", "share", "exchange", "market", "revenue", "sales", "investment"]
    }
}

def calculate_stickiness_ratio(text, vocab):
    stopwords = set(["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "was"])
    words = text.lower().replace(".", "").replace(",", "").split()
    content_words = [w for w in words if w not in stopwords]
    if not content_words: return 0.0
    domain_hits = [w for w in content_words if w in vocab]
    return len(domain_hits) / len(content_words)

def evaluate_model(model_path, model_type, num_samples=20, alpha=0.5):
    print(f"Evaluating {model_type} from {model_path}...")
    
    # Load Model
    if model_type == 'baseline':
        model = StandardTransformer(config).to(device)
    else:
        model = IdeaGatedTransformer(config).to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    results = {}
    
    for domain, data in domains.items():
        prompt = data['prompt']
        vocab = set(data['vocab'])
        scores = []
        
        idx_start = torch.tensor(enc.encode(prompt), device=device).unsqueeze(0)
        
        for _ in range(num_samples):
            idx = idx_start.clone()
            for _ in range(100):
                idx_cond = idx[:, -config['block_size']:]
                with torch.no_grad():
                    if model_type == 'gated':
                        logits, _ = model(idx_cond, alpha=alpha)
                    else:
                        logits = model(idx_cond)
                
                logits = logits[:, -1, :]
                # Repetition Penalty
                for i in range(idx_cond.size(1)):
                    id_prev = idx_cond[0, i].item()
                    if logits[0, id_prev] < 0: logits[0, id_prev] *= 1.2
                    else: logits[0, id_prev] /= 1.2
                
                probs = F.softmax(logits / 0.6, dim=-1)
                idx_next = torch.multinomial(probs, 1)
                idx = torch.cat((idx, idx_next), dim=1)
            
            text = enc.decode(idx[0].tolist())
            scores.append(calculate_stickiness_ratio(text, vocab))
            
        avg_score = np.mean(scores)
        results[domain] = avg_score
        print(f"Domain: {domain:<15} | Stickiness: {avg_score*100:.1f}%")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_path', type=str, required=True)
    parser.add_argument('--gated_path', type=str, required=True)
    args = parser.parse_args()
    
    print("\n--- BASELINE RESULTS ---")
    evaluate_model(args.baseline_path, 'baseline')
    
    print("\n--- GATED RESULTS ---")
    evaluate_model(args.gated_path, 'gated')

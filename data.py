import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset

class StreamDataset(IterableDataset):
    def __init__(self, tokenizer, block_size=512, idea_window=20, max_tokens=None, skip_samples=0):
        """
        Args:
            skip_samples: Number of samples to skip (for creating a validation split)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.idea_window = idea_window
        self.max_tokens = max_tokens
        
        print(f"Loading FineWeb-Edu stream (Skip={skip_samples})...")
        # We use the streaming API. We skip the first 'skip_samples' for validation sets.
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        if skip_samples > 0:
            self.dataset = self.dataset.skip(skip_samples)

    def __iter__(self):
        buffer = []
        token_count = 0
        needed_size = self.block_size + 1 + self.idea_window

        for sample in self.dataset:
            # text = sample['text'] # Depending on dataset version, sometimes it's 'content'
            text = sample.get('text', sample.get('content', '')) 
            
            if not text: continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= needed_size:
                chunk = buffer[:needed_size]
                chunk_t = torch.tensor(chunk, dtype=torch.long)
                
                # Input (x), Target (y), Future (raw_future)
                x = chunk_t[0 : self.block_size]
                y = chunk_t[1 : 1 + self.block_size]
                raw_future = chunk_t[1 : 1 + self.block_size + self.idea_window]

                yield x, y, raw_future

                # Stride by block_size
                buffer = buffer[self.block_size:] 
                
                token_count += self.block_size
                if self.max_tokens and token_count >= self.max_tokens:
                    return

def create_idea_target(raw_future, vocab_size, idea_window, device):
    """
    Creates the Multi-Hot target for the Idea Head.
    """
    B, total_len = raw_future.size()
    block_size = total_len - idea_window
    
    # Initialize with 0s in BFloat16
    targets = torch.zeros((B, block_size, vocab_size), dtype=torch.bfloat16, device=device)
    
    for t in range(block_size):
        window_indices = raw_future[:, t : t + idea_window]
        targets[:, t, :].scatter_(1, window_indices, 1.0)
        
    return targets

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

class IdeaGatedModel(nn.Module):
    def __init__(self, model_name, device, alpha_max=0.5):
        super().__init__()
        self.device = device
        self.alpha_max = alpha_max
        self.beta_clamp = -2.0 # From paper Eq (5)

        print(f"Loading Base Model: {model_name}...")
        
        # 1. Quantization Config (Crucial for g5.xlarge VRAM)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bf16
            bnb_4bit_use_double_quant=True,
        )

        # 2. Load Base Model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0}, # Force to primary GPU
            trust_remote_code=True
        )
        
        # Prepare for LoRA (Freezes weights, casts norms to fp32 for stability)
        self.base_model = prepare_model_for_kbit_training(self.base_model)

        # 3. Add LoRA Adapters
        print("Injecting LoRA Adapters...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,            # Rank
            lora_alpha=32,  # Alpha
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"] # Target Attention layers
        )
        self.base_model = get_peft_model(self.base_model, peft_config)
        self.base_model.print_trainable_parameters()

        # 4. Initialize The Idea Head (System 2)
        # Mistral-7B hidden size is 4096
        hidden_size = self.base_model.config.hidden_size
        vocab_size = self.base_model.config.vocab_size
        
        print(f"Initializing Idea Head ({hidden_size} -> {vocab_size})...")
        
        self.idea_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        ).to(self.device)
        
        # FORCE Idea Head to BFloat16 to match LoRA dtype
        self.idea_head = self.idea_head.to(dtype=torch.bfloat16)

    def forward(self, input_ids, alpha=0.0):
        # 1. Forward pass through Base Model (Mistral + LoRA)
        # We need hidden_states to feed the Idea Head
        outputs = self.base_model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Token Logits (Syntactic Stream) - Shape: [B, T, V]
        # These come out in the compute_dtype (bfloat16) usually
        token_logits = outputs.logits
        
        # Hidden States for Idea Head
        # Mistral/Llama hidden_states[-1] is the state BEFORE the final LM Head
        last_hidden_state = outputs.hidden_states[-1]
        
        # 2. Forward pass through Idea Head (Semantic Stream)
        # last_hidden_state is bfloat16, idea_head is bfloat16 -> OK
        idea_logits = self.idea_head(last_hidden_state)
        
        # 3. Gating Logic (The Paper's Mechanism)
        if alpha > 0:
            # Sigmoid to get probability of future concepts
            p_idea = torch.sigmoid(idea_logits)
            
            # Log-Space Gating (Eq 4)
            epsilon = 1e-8
            gate = alpha * torch.log(p_idea + epsilon)
            
            # Clamp (Eq 5)
            # Ensure tensor is on correct device and dtype
            clamp_val = torch.tensor(self.beta_clamp, device=self.device, dtype=gate.dtype)
            gate_clamped = torch.max(gate, clamp_val)
            
            # Apply to Token Logits (Eq 6)
            final_logits = token_logits + gate_clamped
        else:
            final_logits = token_logits
            
        return final_logits, idea_logits

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer, TransformerConfig

class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Optionally tie weights between token embeddings and LM head
        self.lm_head.weight = self.transformer.token_embeddings.weight

    def forward(self, input_ids, labels=None):
        # Create causal mask to prevent attending to future tokens
        seq_length = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Get transformer outputs
        hidden_states = self.transformer(input_ids, mask=~causal_mask)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        # If we are training
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1
            )
            return logits, loss
        
        return logits

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text using the language model.
        Args:
            input_ids: Starting token ids
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = no change, < 1.0 = more focused, > 1.0 = more random)
            top_k: If set, only sample from the top k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop input_ids to last block_size tokens if needed
            input_ids_cond = input_ids if input_ids.size(1) <= self.config.max_seq_length else input_ids[:, -self.config.max_seq_length:]
            
            # Get predictions
            logits = self(input_ids_cond)
            logits = logits[:, -1, :] / temperature  # Take last time step and apply temperature
            
            # Optionally crop logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)
        
        return input_ids

# Example configurations for different model sizes
def get_gpt2_config(size='small'):
    if size == 'small':  # GPT-2 Small (117M params)
        return TransformerConfig(
            vocab_size=50257,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            max_seq_length=1024
        )
    elif size == 'medium':  # GPT-2 Medium (345M params)
        return TransformerConfig(
            vocab_size=50257,
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
            max_seq_length=1024
        )
    elif size == 'large':  # GPT-2 Large (774M params)
        return TransformerConfig(
            vocab_size=50257,
            hidden_size=1280,
            num_layers=36,
            num_heads=20,
            max_seq_length=1024
        )
    else:
        raise ValueError(f"Unknown size: {size}")

def get_llama_config(size='7B'):
    if size == '7B':
        return TransformerConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            max_seq_length=2048
        )
    else:
        raise ValueError(f"Unknown size: {size}")

def get_mistral_config(size='7B'):
    if size == '7B':
        return TransformerConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            max_seq_length=8192
        )
    else:
        raise ValueError(f"Unknown size: {size}")

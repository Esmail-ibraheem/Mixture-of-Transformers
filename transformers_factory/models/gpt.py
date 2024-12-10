import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

from transformer import Transformer, TransformerConfig

class GPTConfig(TransformerConfig):
    """Configuration class for GPT model."""
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_length: int = 1024,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        tie_word_embeddings: bool = True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_length=max_seq_length,
            dropout=dropout,
            layer_norm_epsilon=layer_norm_epsilon,
        )
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

class GPT(nn.Module):
    """GPT language model implementation."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.token_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Indices of input sequence tokens (batch_size, sequence_length)
            attention_mask: Mask to avoid attention on padding tokens (batch_size, sequence_length)
            labels: Labels for language modeling (batch_size, sequence_length)
            use_cache: Whether to use past key/values for faster inference
            past_key_values: Past key/values for faster inference
            
        Returns:
            tuple: (logits, loss) if labels is provided, else logits
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Create causal mask if attention_mask is not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create causal mask for decoder attention
        seq_length = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length), diagonal=1
        ).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Combine attention mask with causal mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask * (~causal_mask)
        
        # Get transformer outputs
        hidden_states = self.transformer(
            input_ids,
            mask=attention_mask,
        )
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return (logits, loss) if loss is not None else logits

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = None,
        min_length: int = None,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
    ) -> torch.LongTensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Starting token ids
            max_length: Maximum length of generated sequence
            min_length: Minimum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of sequences to generate
            
        Returns:
            torch.LongTensor: Generated token ids
        """
        max_length = max_length if max_length is not None else self.config.max_seq_length
        min_length = min_length if min_length is not None else 0
        
        batch_size = input_ids.shape[0]
        
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        cur_len = input_ids.shape[1]
        
        while cur_len < max_length:
            # Get model predictions
            outputs = self(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        next_token_logits[i, previous_token] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            cur_len += 1
            
            # Check if we've generated min_length tokens
            if cur_len < min_length:
                continue
                
            # Check if any sequence has hit the EOS token
            eos_token_id = self.config.eos_token_id
            if eos_token_id is not None:
                eos_in_sents = next_tokens == eos_token_id
                if eos_in_sents.any():
                    break
        
        return input_ids

    @classmethod
    def from_pretrained(cls, model_path: str) -> 'GPT':
        """
        Load a pretrained GPT model from a directory.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            GPT: Loaded model
        """
        # Load config
        config_path = f"{model_path}/config.json"
        config = GPTConfig.from_json(config_path)
        
        # Create model
        model = cls(config)
        
        # Load state dict
        state_dict = torch.load(f"{model_path}/pytorch_model.bin")
        model.load_state_dict(state_dict)
        
        return model

    def save_pretrained(self, save_directory: str):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model to
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        self.config.to_json(f"{save_directory}/config.json")
        
        # Save model weights
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math

class BaseConfig:
    """Base configuration class for transformer models."""
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_length: int = 1024,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'BaseConfig':
        """Create a configuration from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.__dict__

class BaseAttention(nn.Module):
    """Base attention class that can be extended for different attention mechanisms."""
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        assert self.head_size * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()

        # Linear projections and reshape
        query = self.query(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        return self.proj(context)

class BaseMLP(nn.Module):
    """Base MLP class that can be extended for different feed-forward implementations."""
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.dense_4h_to_h = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class BaseBlock(nn.Module):
    """Base transformer block that can be extended for different architectures."""
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.attention = BaseAttention(config)
        self.mlp = BaseMLP(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask, **kwargs)
        hidden_states = residual + self.dropout(attention_output)

        # MLP
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(mlp_output)

        return hidden_states

class BaseTransformer(nn.Module):
    """Base transformer model that can be extended for different architectures."""
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.max_seq_length, config.hidden_size))
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([BaseBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        assert seq_length <= self.config.max_seq_length, f"Input sequence length {seq_length} exceeds maximum length {self.config.max_seq_length}"

        # Get embeddings
        hidden_states = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings[:, :seq_length, :]
        hidden_states = self.dropout(hidden_states + position_embeddings)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)

        # Convert attention mask to attention bias
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Forward through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, **kwargs)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        return hidden_states

    @classmethod
    def from_pretrained(cls, model_path: str) -> 'BaseTransformer':
        """Load a pretrained model from a directory."""
        config_dict = torch.load(f"{model_path}/config.json")
        config = BaseConfig.from_dict(config_dict)
        
        model = cls(config)
        state_dict = torch.load(f"{model_path}/pytorch_model.bin")
        model.load_state_dict(state_dict)
        
        return model

    def save_pretrained(self, save_directory: str):
        """Save the model to a directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_dict = self.config.to_dict()
        torch.save(config_dict, f"{save_directory}/config.json")
        
        # Save model weights
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

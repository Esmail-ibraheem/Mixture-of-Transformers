import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from base_transformer import BaseTransformer, BaseConfig, BaseAttention, BaseMLP
from transformer import Transformer, TransformerConfig

class LLaMAConfig(BaseConfig):
    """Configuration class for LLaMA model."""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_layers: int = 32,
        num_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        max_seq_length: int = 2048,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[float] = None,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = False,
        use_flash_attention: bool = False,
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
        self.intermediate_size = intermediate_size
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.use_flash_attention = use_flash_attention

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim: int, max_seq_length: int = 2048, theta: float = 10000.0, scaling_factor: Optional[float] = None):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta
        self.scaling_factor = scaling_factor
        
        # Create position embeddings
        position = torch.arange(max_seq_length).unsqueeze(1)
        if scaling_factor is not None:
            position = position * scaling_factor
        
        # Calculate frequencies
        freqs = torch.exp(
            -torch.arange(0, dim, 2).float() * math.log(theta) / dim
        )
        angles = position * freqs
        
        self.register_buffer("cos_cached", torch.cos(angles))
        self.register_buffer("sin_cached", torch.sin(angles))

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LLaMAAttention(nn.Module):
    """Multi-head attention with rotary position embeddings."""
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_seq_length = config.max_seq_length
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_length=config.max_seq_length,
            theta=config.rope_theta,
            scaling_factor=config.rope_scaling
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        # Repeat key and value states for multi-query attention
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(query_states, seq_length)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Reshape for attention computation
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += ((key_states, value_states),)
            
        return outputs

class LLaMAMLP(nn.Module):
    """LLaMA MLP with SwiGLU activation."""
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LLaMA(BaseTransformer):
    """LLaMA model implementation."""
    def __init__(self, config: LLaMAConfig):
        super().__init__(config)
        
        # Replace standard attention with LLaMA attention in all blocks
        for block in self.blocks:
            block.attention = LLaMAAttention(config)
            block.mlp = LLaMAMLP(config)
        
        # Use RMSNorm instead of LayerNorm
        for block in self.blocks:
            block.ln1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            block.ln2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)

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
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache
        )
        
        return outputs

    def _prepare_causal_mask(
        self,
        seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.full((seq_length, seq_length), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        return mask.to(dtype=dtype, device=device)

    @classmethod
    def from_pretrained(cls, model_path: str) -> 'LLaMA':
        """Load a pretrained LLaMA model from a directory."""
        config_path = f"{model_path}/config.json"
        config = LLaMAConfig.from_json(config_path)
        
        model = cls(config)
        state_dict = torch.load(f"{model_path}/pytorch_model.bin")
        model.load_state_dict(state_dict)
        
        return model

    def save_pretrained(self, save_directory: str):
        """Save the model to a directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        self.config.to_json(f"{save_directory}/config.json")
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

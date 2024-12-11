import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math
from dataclasses import dataclass

from base_transformer import BaseTransformer, BaseConfig, BaseAttention, BaseMLP

@dataclass
class MegatronConfig(BaseConfig):
    """Configuration class for Megatron model."""
    def __init__(
        self,
        vocab_size: int = 50432,
        hidden_size: int = 3072,
        num_layers: int = 24,
        num_heads: int = 24,
        max_seq_length: int = 2048,
        intermediate_size: int = 12288,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_bias: bool = True,
        init_method_std: float = 0.02,
        use_scaled_init: bool = True,
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = True,
        layer_norm_epsilon: float = 1e-5,
        use_cpu_initialization: bool = False,
        parallel_attention_mlp: bool = False,
        num_attention_heads_kv: Optional[int] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_length=max_seq_length,
            dropout=hidden_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
        )
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.init_method_std = init_method_std
        self.use_scaled_init = use_scaled_init
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.use_cpu_initialization = use_cpu_initialization
        self.parallel_attention_mlp = parallel_attention_mlp
        self.num_attention_heads_kv = num_attention_heads_kv or num_heads

def scaled_init_method(sigma: float, num_layers: int) -> float:
    """Initialize with a scaled standard deviation."""
    return sigma / math.sqrt(2.0 * num_layers)

class MegatronAttention(nn.Module):
    """Multi-head attention with Megatron optimizations."""
    def __init__(self, config: MegatronConfig, layer_number: int = 1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_attention_heads_kv = config.num_attention_heads_kv
        self.scaling = self.head_dim ** -0.5
        
        init_method = (scaled_init_method(config.init_method_std, config.num_layers)
                      if config.use_scaled_init else config.init_method_std)
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, config.num_attention_heads_kv * self.head_dim, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, config.num_attention_heads_kv * self.head_dim, bias=config.use_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        
        if config.apply_query_key_layer_scaling:
            self.layer_scaling = math.sqrt(layer_number)
        else:
            self.layer_scaling = 1.0
            
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.softmax_in_fp32 = config.attention_softmax_in_fp32

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_length, _ = hidden_states.shape
        
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        query_layer = query_layer.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_layer = key_layer.view(batch_size, seq_length, self.num_attention_heads_kv, self.head_dim).transpose(1, 2)
        value_layer = value_layer.view(batch_size, seq_length, self.num_attention_heads_kv, self.head_dim).transpose(1, 2)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=2)
            
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None
            
        # Repeat k/v heads if num_attention_heads_kv < num_heads
        if self.num_attention_heads_kv < self.num_heads:
            key_layer = key_layer.repeat_interleave(self.num_heads // self.num_attention_heads_kv, dim=1)
            value_layer = value_layer.repeat_interleave(self.num_heads // self.num_attention_heads_kv, dim=1)
            
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.layer_scaling
        attention_scores = attention_scores * self.scaling
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        if self.softmax_in_fp32:
            attention_probs = F.softmax(attention_scores.float(), dim=-1).type_as(attention_scores)
        else:
            attention_probs = F.softmax(attention_scores, dim=-1)
            
        attention_probs = self.attention_dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.hidden_size)
        
        output = self.dense(context_layer)
        
        outputs = (output, present)
        if output_attentions:
            outputs += (attention_probs,)
            
        return outputs

class MegatronMLP(nn.Module):
    """MLP with Megatron optimizations."""
    def __init__(self, config: MegatronConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)
        self.act = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states

class MegatronBlock(nn.Module):
    """Megatron transformer block."""
    def __init__(self, config: MegatronConfig, layer_number: int = 1):
        super().__init__()
        self.parallel_attention_mlp = config.parallel_attention_mlp
        
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attention = MegatronAttention(config, layer_number)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MegatronMLP(config)
        
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        layernorm_output = self.input_layernorm(hidden_states)
        
        # Run attention and mlp in parallel if configured
        if self.parallel_attention_mlp:
            attention_output = self.attention(
                layernorm_output,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            mlp_output = self.mlp(self.post_attention_layernorm(layernorm_output))
            
            attention_output, presents = attention_output[:2]
            
            output = hidden_states + self.hidden_dropout(attention_output) + self.hidden_dropout(mlp_output)
            
            outputs = (output,)
            if use_cache:
                outputs += (presents,)
            if output_attentions:
                outputs += (attention_output[2],)
        else:
            attention_output = self.attention(
                layernorm_output,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attention_output, presents = attention_output[:2]
            
            attention_output = hidden_states + self.hidden_dropout(attention_output)
            
            # MLP
            layernorm_output = self.post_attention_layernorm(attention_output)
            mlp_output = self.mlp(layernorm_output)
            output = attention_output + self.hidden_dropout(mlp_output)
            
            outputs = (output,)
            if use_cache:
                outputs += (presents,)
            if output_attentions:
                outputs += (attention_output[2],)
                
        return outputs

class MegatronModel(BaseTransformer):
    """Megatron-LM model implementation."""
    def __init__(self, config: MegatronConfig):
        super().__init__(config)
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [MegatronBlock(config, layer_number=i+1) for i in range(config.num_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        if not config.use_cpu_initialization:
            self.to(torch.cuda.current_device())
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], dict]:
        batch_size, seq_length = input_ids.shape
        
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
            
        hidden_states = self.embed_tokens(input_ids)
        
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, (block, layer_past) in enumerate(zip(self.blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            hidden_states = outputs[0]
            if use_cache:
                presents += (outputs[1],)
            if output_attentions:
                all_attentions += (outputs[2],)
                
        hidden_states = self.final_layernorm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": presents,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }
            
        return tuple(v for v in [
            hidden_states,
            presents,
            all_hidden_states,
            all_attentions,
        ] if v is not None)

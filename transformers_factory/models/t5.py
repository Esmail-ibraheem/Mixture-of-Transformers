import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math

from base_transformer import BaseTransformer, BaseConfig, BaseAttention, BaseMLP

class T5Config(BaseConfig):
    """Configuration class for T5 model."""
    def __init__(
        self,
        vocab_size: int = 32128,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_decoder_layers: Optional[int] = None,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
        feed_forward_proj: str = "relu",
        is_encoder_decoder: bool = True,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
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
        self.num_decoder_layers = num_decoder_layers or num_layers
        self.d_ff = d_ff
        self.feed_forward_proj = feed_forward_proj
        self.is_encoder_decoder = is_encoder_decoder
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

class T5LayerNorm(nn.Module):
    """T5-style layer normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class T5Attention(BaseAttention):
    """T5 attention mechanism with relative position bias."""
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__(config)
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, config.num_heads
            )

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
            
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.size()
        
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        query = query.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(seq_length, seq_length, hidden_states.device)
            
        if position_bias is not None:
            scores += position_bias
            
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        context = self.proj(context)
        
        outputs = (context,)
        if output_attentions:
            outputs += (attn_weights,)
        if self.has_relative_attention_bias:
            outputs += (position_bias,)
            
        return outputs

class T5Block(nn.Module):
    """T5 transformer block."""
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = config.is_encoder_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5Attention(
                config,
                has_relative_attention_bias=has_relative_attention_bias
            )
        )
        if self.is_decoder:
            self.layer.append(T5Attention(config))
            
        self.layer.append(BaseMLP(config))
        
        self.layer_norm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        
        self_attention_outputs = self.layer[0](
            self.layer_norm(hidden_states),
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(self_attention_outputs[0])
        position_bias = self_attention_outputs[-1]
        
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.layer[1](
                self.layer_norm(hidden_states),
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                output_attentions=output_attentions,
            )
            hidden_states = hidden_states + self.dropout(cross_attention_outputs[0])
            
        feed_forward_outputs = self.layer[-1](self.layer_norm(hidden_states))
        hidden_states = hidden_states + self.dropout(feed_forward_outputs)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attention_outputs[1],)
            if self.is_decoder:
                outputs += (cross_attention_outputs[1],)
        outputs += (position_bias,)
        
        return outputs

class T5Stack(BaseTransformer):
    """T5 stack of transformer blocks."""
    def __init__(self, config: T5Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.is_decoder = config.is_encoder_decoder
        self.embed_tokens = embed_tokens
        self.blocks = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], dict]:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None
        
        for i, layer_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            position_bias = layer_outputs[-1]
            
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "position_bias": position_bias,
        }

class T5(nn.Module):
    """T5 model with encoder and decoder."""
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size)
        encoder_config = config
        decoder_config = config
        decoder_config.is_decoder = True
        
        self.encoder = T5Stack(encoder_config, self.shared)
        self.decoder = T5Stack(decoder_config, self.shared)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                
    def forward(
        self,
        input_ids: torch.LongTensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], dict]:
        # Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        if return_dict:
            return {
                "encoder_last_hidden_state": encoder_outputs["last_hidden_state"],
                "encoder_hidden_states": encoder_outputs.get("hidden_states"),
                "encoder_attentions": encoder_outputs.get("attentions"),
                "decoder_last_hidden_state": decoder_outputs["last_hidden_state"],
                "decoder_hidden_states": decoder_outputs.get("hidden_states"),
                "decoder_attentions": decoder_outputs.get("attentions"),
                "cross_attentions": decoder_outputs.get("cross_attentions"),
            }
        
        return (
            encoder_outputs["last_hidden_state"],
            decoder_outputs["last_hidden_state"],
        )

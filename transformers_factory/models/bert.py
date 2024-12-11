import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math

from base_transformer import BaseTransformer, BaseConfig, BaseAttention, BaseMLP

class BertConfig(BaseConfig):
    """Configuration class for BERT model."""
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-12,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
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
        self.attention_dropout = attention_dropout
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.classifier_dropout = classifier_dropout or dropout

class BertAttention(BaseAttention):
    """BERT attention mechanism."""
    def __init__(self, config):
        super().__init__(config)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Reshape for multi-head attention
        batch_size = hidden_states.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)

        outputs = (self.proj(context),)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs[0] if len(outputs) == 1 else outputs

class BertIntermediate(nn.Module):
    """BERT intermediate layer."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    """BERT output layer."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    """BERT transformer layer."""
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0] if isinstance(attention_outputs, tuple) else attention_outputs

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,)
        if output_attentions:
            outputs += (attention_outputs[1],)

        return outputs

class BertPooler(nn.Module):
    """BERT pooler for sentence-level tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Bert(BaseTransformer):
    """BERT model implementation."""
    def __init__(self, config: BertConfig):
        super().__init__(config)
        
        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # Replace transformer blocks with BERT layers
        self.blocks = nn.ModuleList([BertLayer(config) for _ in range(config.num_layers)])
        
        # Add pooler
        self.pooler = BertPooler(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], dict]:
        batch_size, seq_length = input_ids.size()
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=input_ids.device)
            
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        # Get embeddings
        embeddings = self.embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Add token type embeddings
        hidden_states = embeddings + token_type_embeddings
        
        # Add position embeddings
        if self.config.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings[:, :seq_length, :]
            hidden_states = hidden_states + position_embeddings
            
        hidden_states = self.dropout(hidden_states)
        
        # Convert attention mask to attention bias
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(hidden_states.dtype).min
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Forward through transformer blocks
        for layer in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions += (layer_outputs[1],)
                
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        # Get pooled output
        pooled_output = self.pooler(hidden_states)
        
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "pooler_output": pooled_output,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }
        
        return tuple(v for v in [
            hidden_states,
            pooled_output,
            all_hidden_states,
            all_attentions
        ] if v is not None)

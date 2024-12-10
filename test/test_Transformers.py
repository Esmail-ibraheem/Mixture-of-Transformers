from ..transformers_factory.models import GPT, GPTConfig
from ..transformers_factory.models import LLaMA, LLaMAConfig
from ..transformers_factory.base_transformer import BaseTransformer, BaseConfig

import torch
import torch.nn as nn

def test_GPT():
    config = GPTConfig(
        vocab_size=1000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=1024,
        dropout=0.1
    )
    model = GPT(config)
    assert model.config == config

def test_LLaMA():
    config = LLaMAConfig(
        vocab_size=1000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=1024,
        dropout=0.1
    )
    model = LLaMA(config)
    assert model.config == config


def test_BaseTransformer():
    config = BaseConfig(
        vocab_size=1000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=1024,
        dropout=0.1
    )
    model = BaseTransformer(config)
    assert model.config == config

class MyAttention(BaseTransformer):
    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.attention = BaseAttention(config)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        return self.attention(hidden_states, attention_mask, **kwargs)

class MyModel(BaseTransformer):
    def __init__(self, config):
        super().__init__(config)
        # Add custom layers
        self.custom_layer = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, input_ids, attention_mask=None):
        # Custom forward implementation
        hidden_states = super().forward(input_ids, attention_mask)
        return self.custom_layer(hidden_states)

if __name__ == '__main__':
    test_GPT()
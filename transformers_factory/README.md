# Transformers Factory

A flexible and modular library for building transformer-based language models. This library provides base components and implementations for various transformer architectures like GPT, LLaMA, and more.

## Installation

```bash
pip install transformers-factory
```

## Quick Start

```python
from transformers_factory.models import GPT
from transformers_factory.config import GPTConfig

# Create a GPT model with custom configuration
config = GPTConfig(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    max_seq_length=1024
)
model = GPT(config)

# Generate text
input_ids = torch.tensor([[1, 2, 3]])  # Your tokenized input
output = model.generate(input_ids, max_new_tokens=50)
```

## Features

- Modular transformer architecture
- Easy-to-customize configurations
- Support for various model architectures (GPT, LLaMA, etc.)
- Text generation with different sampling strategies
- Training utilities

## Creating Custom Models

You can create custom models by:
1. Extending the base transformer classes
2. Modifying the configuration
3. Implementing custom attention mechanisms

Example:
```python
from transformers_factory.models import BaseTransformer
from transformers_factory.config import TransformerConfig

class MyCustomModel(BaseTransformer):
    def __init__(self, config):
        super().__init__(config)
        # Add custom layers here
```

## License

MIT License

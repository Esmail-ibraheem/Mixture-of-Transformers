import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers_factory.base_transformer import BaseTransformer, BaseConfig, BaseAttention

class CustomAttention(BaseAttention):
    """Custom attention mechanism with additional features."""
    def __init__(self, config):
        super().__init__(config)
        # Add custom attention features
        self.attention_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Custom attention implementation
        attention_output = super().forward(hidden_states, attention_mask)
        return attention_output * self.attention_scale

class CustomConfig(BaseConfig):
    """Custom configuration with additional parameters."""
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=1024,
        dropout=0.1,
        # Custom parameters
        use_custom_attention=True,
        custom_feature_size=64
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        self.use_custom_attention = use_custom_attention
        self.custom_feature_size = custom_feature_size

class CustomModel(BaseTransformer):
    """Custom transformer model with additional features."""
    def __init__(self, config):
        super().__init__(config)
        
        # Replace standard attention with custom attention if specified
        if config.use_custom_attention:
            for block in self.blocks:
                block.attention = CustomAttention(config)
        
        # Add custom head
        self.custom_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.custom_feature_size),
            nn.Tanh(),
            nn.Linear(config.custom_feature_size, config.vocab_size)
        )
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get base transformer outputs
        hidden_states = super().forward(input_ids, attention_mask)
        
        # Apply custom head
        logits = self.custom_head(hidden_states)
        
        return logits

def main():
    # Create custom configuration
    config = CustomConfig(
        vocab_size=30000,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        max_seq_length=512,
        custom_feature_size=128
    )
    
    # Initialize custom model
    model = CustomModel(config)
    
    # Example usage
    batch_size, seq_length = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    outputs = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Save and load model
    model.save_pretrained("custom_model")
    loaded_model = CustomModel.from_pretrained("custom_model")

if __name__ == "__main__":
    main()

import torch
from transformers_factory.models import LLaMA, LLaMAConfig

def main():
    # Create a small LLaMA model for demonstration
    config = LLaMAConfig(
        vocab_size=32000,
        hidden_size=512,      # Smaller for demonstration
        intermediate_size=1024,
        num_layers=8,         # Fewer layers for demonstration
        num_heads=8,
        num_key_value_heads=4,  # Using grouped-query attention
        max_seq_length=1024,
        rope_theta=10000.0,
        rope_scaling=1.0
    )
    
    # Initialize the model
    model = LLaMA(config)
    
    # Example input (batch_size=2, sequence_length=10)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    
    # Forward pass
    outputs = model(input_ids)
    hidden_states = outputs[0]
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output hidden states shape: {hidden_states.shape}")
    
    # Example of loading and saving the model
    # model.save_pretrained("path/to/save")
    # loaded_model = LLaMA.from_pretrained("path/to/save")

if __name__ == "__main__":
    main()

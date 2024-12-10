import torch
from transformers_factory.models import GPT, GPTConfig

def main():
    # Create a small GPT model for demonstration
    config = GPTConfig(
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=1024,
        dropout=0.1
    )
    
    # Initialize the model
    model = GPT(config)
    
    # Example input (batch_size=2, sequence_length=10)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    
    # Forward pass
    outputs = model(input_ids)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Example of text generation
    generated = model.generate(
        input_ids,
        max_length=20,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    print(f"Generated sequence shape: {generated.shape}")

if __name__ == "__main__":
    main()

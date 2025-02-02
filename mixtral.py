import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardExpert(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        A two-layer MLP acting as a single expert.
        """
        super(FeedForwardExpert, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # (alternatively use GELU)

    def forward(self, x):
        """
        x: Tensor of shape (num_tokens, d_model)
        Returns: Tensor of shape (num_tokens, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SwitchFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, dropout=0.1, capacity_factor=1.25):
        """
        Implements a top-1 gated MoE layer with capacity constraints and an 
        auxiliary load-balancing loss as described in the Switch Transformer paper.
        
        Args:
            d_model: hidden dimension.
            d_ff: hidden dimension of the MLP experts.
            num_experts: number of expert FFNs.
            dropout: dropout rate.
            capacity_factor: multiplier to determine the maximum tokens an expert 
                             can receive (relative to average tokens per expert).
        """
        super(SwitchFeedForward, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # Gating network: projects token representations to logits over experts.
        self.gate = nn.Linear(d_model, num_experts)

        # Create a ModuleList of experts.
        self.experts = nn.ModuleList(
            [FeedForwardExpert(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq, d_model)
        Returns:
            output: Tensor of shape (batch, seq, d_model) after MoE routing.
            aux_loss: Scalar auxiliary loss (for load balancing).
        """
        B, S, D = x.shape
        total_tokens = B * S

        # Flatten tokens to shape (B*S, D)
        x_flat = x.view(total_tokens, D)

        # Compute gating scores and probabilities for each token.
        gate_logits = self.gate(x_flat)         # (total_tokens, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (total_tokens, num_experts)

        # Top-1 gating: select the expert with the highest probability.
        top_expert_indices = gate_probs.argmax(dim=-1)  # (total_tokens,)
        top_gate = gate_probs[torch.arange(total_tokens), top_expert_indices]  # (total_tokens,)

        # Determine expert capacity.
        capacity = int(math.ceil(total_tokens / self.num_experts * self.capacity_factor))

        # Prepare output (flattened) and count tokens per expert.
        output_flat = torch.zeros_like(x_flat)
        # Will use this to compute the load (fraction of tokens assigned) per expert.
        token_counts = torch.zeros(self.num_experts, device=x.device)

        # Process each expert separately.
        for expert_id, expert in enumerate(self.experts):
            # Find indices of tokens routed to this expert.
            indices = (top_expert_indices == expert_id).nonzero(as_tuple=False).squeeze(-1)
            n_tokens = indices.numel()
            token_counts[expert_id] = n_tokens

            if n_tokens == 0:
                continue  # No tokens routed to this expert.

            # Enforce capacity: only process up to "capacity" tokens.
            if n_tokens > capacity:
                indices = indices[:capacity]

            # Select tokens and route them through the expert.
            tokens_expert = x_flat[indices]  # (n_effective, D)
            expert_output = expert(tokens_expert)  # (n_effective, D)

            # Scale expert output by the corresponding gate probability.
            scaling = top_gate[indices].unsqueeze(-1)  # (n_effective, 1)
            expert_output = expert_output * scaling

            # Write expert output back to the proper positions.
            output_flat[indices] = expert_output

        # Reshape output back to (B, S, D)
        output = output_flat.view(B, S, D)

        # --- Compute auxiliary load-balancing loss ---
        # importance: average gate probability for each expert over all tokens.
        importance = gate_probs.mean(dim=0)  # (num_experts,)
        # load: fraction of tokens routed (before capacity truncation) to each expert.
        load = token_counts / total_tokens  # (num_experts,)
        # Auxiliary loss encourages a balanced load across experts.
        aux_loss = (importance * load).sum() * self.num_experts

        return output, aux_loss

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_experts, dropout=0.1, capacity_factor=1.25):
        """
        A single transformer encoder layer with self-attention and a Switch MoE FFN.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model

        # Self-attention sublayer.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Layer normalization modules.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout modules.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Replace the standard FFN with our Switch MoE layer.
        self.switch_ff = SwitchFeedForward(d_model, d_ff, num_experts,
                                             dropout=dropout, capacity_factor=capacity_factor)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: Tensor of shape (batch, seq, d_model)
        Returns:
            output: Tensor of shape (batch, seq, d_model)
            aux_loss: Sum of the auxiliary loss from the MoE layer.
        """
        # MultiheadAttention in PyTorch expects (seq, batch, d_model)
        x_transposed = x.transpose(0, 1)  # (seq, batch, d_model)
        attn_output, _ = self.self_attn(x_transposed, x_transposed, x_transposed,
                                        attn_mask=attn_mask,
                                        key_padding_mask=key_padding_mask)
        attn_output = attn_output.transpose(0, 1)  # Back to (batch, seq, d_model)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Switch Feed-Forward (MoE) sublayer.
        ff_output, aux_loss = self.switch_ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x, aux_loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Implements sinusoidal positional encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the denominator term.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dimension.
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq, d_model)
        Returns:
            Tensor of shape (batch, seq, d_model) with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SwitchTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, d_ff=2048,
                 num_layers=6, num_experts=4, max_seq_length=512,
                 dropout=0.1, capacity_factor=1.25):
        """
        A transformer encoder model with Switch MoE layers.
        For example purposes, this model is set up for tasks such as language modeling.
        
        Args:
            vocab_size: Size of the input vocabulary.
            d_model: Embedding and hidden dimension.
            nhead: Number of attention heads.
            d_ff: Hidden dimension of the FFN (each expert).
            num_layers: Number of transformer encoder layers.
            num_experts: Number of experts in each Switch MoE layer.
            max_seq_length: Maximum input sequence length.
            dropout: Dropout probability.
            capacity_factor: Controls per-expert capacity (typically around 1.25).
        """
        super(SwitchTransformer, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)

        # Create a stack of transformer encoder layers.
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_ff, num_experts,
                                      dropout=dropout, capacity_factor=capacity_factor)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        # Final linear layer to project hidden states to vocabulary logits.
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        """
        Args:
            src: Tensor of shape (batch, seq) containing token indices.
        Returns:
            logits: Tensor of shape (batch, seq, vocab_size).
            total_aux_loss: Sum of auxiliary losses from all MoE layers.
        """
        # Embed tokens and scale (as in “Attention Is All You Need”).
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        logits = self.fc_out(x)
        return logits, total_aux_loss


if __name__ == "__main__":
    # Example configuration.
    vocab_size = 10000      # Size of the vocabulary.
    batch_size = 2
    seq_length = 16         # Input sequence length.

    # Instantiate the model (using smaller dimensions for demonstration).
    model = SwitchTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        d_ff=1024,
        num_layers=2,
        num_experts=4,
        max_seq_length=seq_length,
        dropout=0.1,
        capacity_factor=1.25
    )

    # Create a random batch of token indices.
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass: obtain logits and the auxiliary load-balancing loss.
    logits, aux_loss = model(input_ids)
    print("Logits shape:", logits.shape)         # Expected: (batch_size, seq_length, vocab_size)
    print("Auxiliary loss:", aux_loss.item())

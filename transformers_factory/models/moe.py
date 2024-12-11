import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math

from base_transformer import BaseTransformer, BaseConfig, BaseAttention, BaseMLP

class MoEConfig(BaseConfig):
    """Configuration class for Mixture of Experts model."""
    def __init__(
        self,
        vocab_size: int = 32128,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        expert_capacity_factor: float = 1.0,
        expert_hidden_size: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        router_z_loss_coef: float = 0.001,
        router_aux_loss_coef: float = 0.001,
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
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_capacity_factor = expert_capacity_factor
        self.expert_hidden_size = expert_hidden_size
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef

class ExpertLayer(nn.Module):
    """Single expert layer implementing a feed-forward network."""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.expert_hidden_size)
        self.fc2 = nn.Linear(config.expert_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class Router(nn.Module):
    """Router module that assigns tokens to experts."""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_capacity = int(config.expert_capacity_factor * 
                                 (config.max_seq_length * config.num_experts_per_tok) / 
                                 config.num_experts)
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Calculate routing scores
        router_logits = self.router(hidden_states)  # [batch, seq_len, num_experts]
        
        # Calculate routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        
        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Create dispatch tensors
        dispatch_tensor = torch.zeros(
            batch_size,
            sequence_length,
            self.num_experts,
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # Create combine tensor for weighted combination of expert outputs
        combine_tensor = dispatch_tensor.clone()
        
        # Assign tokens to experts
        position_in_expert = torch.zeros(
            (batch_size, self.num_experts),
            device=hidden_states.device,
            dtype=torch.int32
        )
        
        # Compute auxiliary load balancing loss
        aux_loss = torch.mean(
            torch.sum(router_probs * router_probs, dim=-1)
        ) * self.num_experts
        
        # Route tokens to experts
        for i in range(self.num_experts_per_tok):
            expert_index = expert_indices[:, :, i]
            expert_weight = expert_weights[:, :, i]
            
            # Update position counts
            for batch_idx in range(batch_size):
                for seq_idx in range(sequence_length):
                    expert_idx = expert_index[batch_idx, seq_idx]
                    if position_in_expert[batch_idx, expert_idx] < self.expert_capacity:
                        position = position_in_expert[batch_idx, expert_idx]
                        dispatch_tensor[batch_idx, seq_idx, expert_idx] = 1.0
                        combine_tensor[batch_idx, seq_idx, expert_idx] = expert_weight[batch_idx, seq_idx]
                        position_in_expert[batch_idx, expert_idx] += 1
        
        # Compute router z-loss
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
        
        return dispatch_tensor, combine_tensor, router_probs, {
            "aux_loss": aux_loss,
            "z_loss": z_loss
        }

class MoELayer(nn.Module):
    """Mixture of Experts layer."""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.router = Router(config)
        self.experts = nn.ModuleList([ExpertLayer(config) for _ in range(config.num_experts)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        # Get routing information
        dispatch_tensor, combine_tensor, router_probs, router_losses = self.router(hidden_states)
        
        expert_outputs = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_input = hidden_states * dispatch_tensor[:, :, i].unsqueeze(-1)
            # Process tokens
            processed = expert(expert_input)
            # Combine outputs
            expert_outputs += processed * combine_tensor[:, :, i].unsqueeze(-1)
        
        output = self.dropout(expert_outputs) + residual
        return output, router_losses

class MoEBlock(nn.Module):
    """Transformer block with Mixture of Experts."""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.attention = BaseAttention(config)
        self.moe = MoELayer(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(attention_output) + residual
        
        # MoE FFN
        moe_output, router_losses = self.moe(hidden_states)
        
        return moe_output, router_losses

class MoETransformer(BaseTransformer):
    """Transformer model with Mixture of Experts layers."""
    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.config = config
        self.blocks = nn.ModuleList([MoEBlock(config) for _ in range(config.num_layers)])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_router_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        hidden_states = self.embeddings(input_ids)
        
        total_router_losses = {
            "aux_loss": 0.0,
            "z_loss": 0.0
        }
        
        for block in self.blocks:
            hidden_states, router_losses = block(hidden_states, attention_mask)
            total_router_losses["aux_loss"] += router_losses["aux_loss"]
            total_router_losses["z_loss"] += router_losses["z_loss"]
        
        # Scale losses by their coefficients
        total_router_losses["aux_loss"] *= self.config.router_aux_loss_coef
        total_router_losses["z_loss"] *= self.config.router_z_loss_coef
        
        if output_router_logits:
            return hidden_states, total_router_losses
        return hidden_states

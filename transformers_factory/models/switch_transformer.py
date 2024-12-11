import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math

from base_transformer import BaseTransformer, BaseConfig, BaseAttention, BaseMLP
from .moe import ExpertLayer, Router

class SwitchConfig(BaseConfig):
    """Configuration class for Switch Transformer model."""
    def __init__(
        self,
        vocab_size: int = 32128,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_experts: int = 8,
        expert_capacity_factor: float = 1.0,
        expert_hidden_size: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        router_z_loss_coef: float = 0.001,
        jitter_noise: float = 0.1,
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
        self.expert_capacity_factor = expert_capacity_factor
        self.expert_hidden_size = expert_hidden_size
        self.router_z_loss_coef = router_z_loss_coef
        self.jitter_noise = jitter_noise

class SwitchRouter(nn.Module):
    """Switch Transformer router that routes each token to a single expert."""
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = int(config.expert_capacity_factor * config.max_seq_length / config.num_experts)
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.jitter_noise = config.jitter_noise
        
    def _add_jitter(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add noise to encourage load balancing."""
        if self.training and self.jitter_noise > 0:
            noise = torch.rand_like(hidden_states) * self.jitter_noise
            hidden_states = hidden_states + noise
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Add jitter noise during training
        hidden_states = self._add_jitter(hidden_states)
        
        # Calculate routing scores
        router_logits = self.router(hidden_states)
        
        # Calculate routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top expert
        expert_weights, expert_indices = torch.max(router_probs, dim=-1)
        
        # Create dispatch tensors
        dispatch_tensor = torch.zeros(
            batch_size,
            sequence_length,
            self.num_experts,
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # Create combine tensor
        combine_tensor = dispatch_tensor.clone()
        
        # Track number of tokens per expert
        position_in_expert = torch.zeros(
            (batch_size, self.num_experts),
            device=hidden_states.device,
            dtype=torch.int32
        )
        
        # Compute load balancing loss
        aux_loss = torch.mean(
            torch.sum(router_probs * router_probs, dim=-1)
        ) * self.num_experts
        
        # Route tokens to experts
        for batch_idx in range(batch_size):
            for seq_idx in range(sequence_length):
                expert_idx = expert_indices[batch_idx, seq_idx]
                if position_in_expert[batch_idx, expert_idx] < self.expert_capacity:
                    dispatch_tensor[batch_idx, seq_idx, expert_idx] = 1.0
                    combine_tensor[batch_idx, seq_idx, expert_idx] = expert_weights[batch_idx, seq_idx]
                    position_in_expert[batch_idx, expert_idx] += 1
        
        # Compute router z-loss
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
        
        return dispatch_tensor, combine_tensor, router_probs, {
            "aux_loss": aux_loss,
            "z_loss": z_loss
        }

class SwitchLayer(nn.Module):
    """Switch Transformer layer with single expert routing."""
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.router = SwitchRouter(config)
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

class SwitchBlock(nn.Module):
    """Transformer block with Switch routing."""
    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.attention = BaseAttention(config)
        self.switch = SwitchLayer(config)
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
        
        # Switch FFN
        switch_output, router_losses = self.switch(hidden_states)
        
        return switch_output, router_losses

class SwitchTransformer(BaseTransformer):
    """Switch Transformer model."""
    def __init__(self, config: SwitchConfig):
        super().__init__(config)
        self.config = config
        self.blocks = nn.ModuleList([SwitchBlock(config) for _ in range(config.num_layers)])
        
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
        
        # Scale losses
        total_router_losses["z_loss"] *= self.config.router_z_loss_coef
        
        if output_router_logits:
            return hidden_states, total_router_losses
        return hidden_states

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .search_space import SearchSpace, ArchitectureConfig, ModelType, AttentionType, FFNType, NormType

class ArchitectureController(nn.Module):
    """LSTM controller for generating transformer architectures."""
    def __init__(
        self,
        search_space: SearchSpace,
        hidden_size: int = 128,
        num_layers: int = 2,
        temperature: float = 1.0
    ):
        super().__init__()
        self.search_space = search_space
        self.hidden_size = hidden_size
        self.temperature = temperature
        
        # Embedding for discrete choices
        self.choice_embeddings = nn.ModuleDict()
        for name, choices in search_space.get_discrete_choices().items():
            num_choices = len(choices)
            self.choice_embeddings[name] = nn.Embedding(num_choices, hidden_size)
        
        # LSTM controller
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Output heads for different decisions
        self.discrete_heads = nn.ModuleDict()
        for name, choices in search_space.get_discrete_choices().items():
            self.discrete_heads[name] = nn.Linear(hidden_size, len(choices))
            
        self.continuous_heads = nn.ModuleDict()
        for name, (min_val, max_val) in search_space.get_continuous_ranges().items():
            # For each continuous parameter, predict mean and log_std
            self.continuous_heads[name] = nn.Linear(hidden_size, 2)
    
    def forward(self, batch_size: int = 1) -> Tuple[List[ArchitectureConfig], torch.Tensor]:
        """Generate architecture configurations and their log probabilities."""
        device = next(self.parameters()).device
        
        # Initialize LSTM hidden state
        h = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_size, device=device)
        hidden = (h, c)
        
        # Start token embedding
        x = torch.zeros(batch_size, 1, self.hidden_size, device=device)
        
        configs = []
        log_probs = []
        
        # Generate discrete choices first
        discrete_choices = {}
        for name in self.search_space.get_discrete_choices().keys():
            output, hidden = self.lstm(x, hidden)
            logits = self.discrete_heads[name](output[:, -1]) / self.temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample from categorical distribution
            distribution = torch.distributions.Categorical(probs)
            choice = distribution.sample()
            log_prob = distribution.log_prob(choice)
            
            discrete_choices[name] = choice
            log_probs.append(log_prob)
            
            # Update input for next step
            x = self.choice_embeddings[name](choice).unsqueeze(1)
        
        # Generate continuous parameters
        continuous_params = {}
        for name, (min_val, max_val) in self.search_space.get_continuous_ranges().items():
            output, hidden = self.lstm(x, hidden)
            stats = self.continuous_heads[name](output[:, -1])
            mean, log_std = stats[:, 0], stats[:, 1]
            
            # Sample from truncated normal distribution
            std = torch.exp(log_std)
            distribution = torch.distributions.Normal(mean, std)
            param = distribution.sample()
            log_prob = distribution.log_prob(param)
            
            # Scale to range and clip
            param = torch.sigmoid(param) * (max_val - min_val) + min_val
            param = torch.clamp(param, min_val, max_val)
            
            if name in ['num_layers', 'num_heads', 'hidden_size', 'num_experts']:
                param = param.round()
                
            continuous_params[name] = param
            log_probs.append(log_prob)
        
        # Create configurations
        configs = []
        for i in range(batch_size):
            config = ArchitectureConfig(
                model_type=ModelType(discrete_choices['model_type'][i].item()),
                attention_type=AttentionType(discrete_choices['attention_type'][i].item()),
                ffn_type=FFNType(discrete_choices['ffn_type'][i].item()),
                norm_type=NormType(discrete_choices['norm_type'][i].item()),
                num_layers=int(continuous_params['num_layers'][i].item()),
                num_heads=int(continuous_params['num_heads'][i].item()),
                hidden_size=int(continuous_params['hidden_size'][i].item()),
                intermediate_size=int(continuous_params['hidden_size'][i].item() * 
                                   continuous_params['intermediate_ratio'][i].item()),
                dropout=continuous_params['dropout'][i].item(),
                learning_rate=float(discrete_choices['learning_rate'][i].item()),
            )
            
            # Add optional parameters based on choices
            if config.ffn_type == FFNType.MOE:
                config.num_experts = int(continuous_params['num_experts'][i].item())
                config.experts_per_token = int(continuous_params['experts_per_token'][i].item())
                
            if config.attention_type == AttentionType.SLIDING_WINDOW:
                config.window_size = int(continuous_params['window_size'][i].item())
                
            configs.append(config)
        
        return configs, torch.stack(log_probs).sum(dim=0)

class ReinforcementLearner:
    """Reinforcement learning for the architecture controller."""
    def __init__(
        self,
        controller: ArchitectureController,
        learning_rate: float = 0.001,
        entropy_weight: float = 0.0001,
        baseline_decay: float = 0.999
    ):
        self.controller = controller
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
        self.entropy_weight = entropy_weight
        self.baseline_decay = baseline_decay
        self.baseline = None
    
    def update_baseline(self, reward: float):
        """Update the moving average baseline."""
        if self.baseline is None:
            self.baseline = reward
        else:
            self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
    
    def step(self, rewards: torch.Tensor):
        """Update controller parameters using REINFORCE."""
        self.optimizer.zero_grad()
        
        # Generate new architectures
        configs, log_probs = self.controller(len(rewards))
        
        # Compute advantages
        rewards = torch.tensor(rewards, device=log_probs.device)
        advantages = rewards - self.baseline if self.baseline is not None else rewards
        
        # Policy gradient loss
        policy_loss = -(advantages * log_probs).mean()
        
        # Add entropy regularization if specified
        if self.entropy_weight > 0:
            entropy_loss = -self.entropy_weight * log_probs.mean()
            loss = policy_loss + entropy_loss
        else:
            loss = policy_loss
        
        # Update parameters
        loss.backward()
        self.optimizer.step()
        
        # Update baseline
        self.update_baseline(rewards.mean().item())
        
        return loss.item()

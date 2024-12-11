import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass

from .search_space import ArchitectureConfig
from ..models import (
    GPT, GPTConfig,
    LLaMA, LLaMAConfig,
    Bert, BertConfig,
    T5, T5Config,
    MoETransformer, MoEConfig,
    SwitchTransformer, SwitchConfig,
    MistralModel, MistralConfig,
    MegatronModel, MegatronConfig
)

@dataclass
class EvaluationMetrics:
    """Metrics for evaluating transformer architectures."""
    perplexity: float
    latency_ms: float
    memory_mb: float
    params_m: float  # parameters in millions
    
    def compute_reward(
        self,
        perplexity_weight: float = 1.0,
        latency_weight: float = 0.3,
        memory_weight: float = 0.3,
        params_weight: float = 0.2
    ) -> float:
        """Compute weighted reward from metrics."""
        # Normalize metrics (lower is better)
        norm_perplexity = 1.0 / max(self.perplexity, 1e-6)
        norm_latency = 1.0 / max(self.latency_ms, 1e-6)
        norm_memory = 1.0 / max(self.memory_mb, 1e-6)
        norm_params = 1.0 / max(self.params_m, 1e-6)
        
        # Compute weighted sum
        reward = (
            perplexity_weight * norm_perplexity +
            latency_weight * norm_latency +
            memory_weight * norm_memory +
            params_weight * norm_params
        )
        
        return reward

class ModelEvaluator:
    """Evaluates transformer architectures using proxy tasks."""
    def __init__(
        self,
        eval_batch_size: int = 32,
        eval_seq_length: int = 512,
        num_eval_steps: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.eval_batch_size = eval_batch_size
        self.eval_seq_length = eval_seq_length
        self.num_eval_steps = num_eval_steps
        self.device = device
        
        # Create dummy data for quick evaluation
        self.dummy_input = torch.randint(
            0, 1000,
            (eval_batch_size, eval_seq_length),
            device=device
        )
        self.dummy_labels = torch.randint(
            0, 1000,
            (eval_batch_size, eval_seq_length),
            device=device
        )
    
    def create_model(self, config: ArchitectureConfig) -> nn.Module:
        """Create a model from architecture configuration."""
        if config.model_type == "decoder_only":
            if config.ffn_type == "moe":
                model_config = MoEConfig(
                    vocab_size=32000,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.experts_per_token
                )
                return MoETransformer(model_config)
            else:
                model_config = GPTConfig(
                    vocab_size=32000,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads
                )
                return GPT(model_config)
        elif config.model_type == "encoder_decoder":
            model_config = T5Config(
                vocab_size=32000,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads
            )
            return T5(model_config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    def measure_latency(self, model: nn.Module) -> float:
        """Measure average inference latency in milliseconds."""
        model.eval()
        total_time = 0
        
        with torch.no_grad():
            for _ in range(10):  # Warm up
                _ = model(self.dummy_input)
            
            for _ in range(self.num_eval_steps):
                start_time = time.time()
                _ = model(self.dummy_input)
                total_time += (time.time() - start_time) * 1000  # Convert to ms
        
        return total_time / self.num_eval_steps
    
    def measure_memory(self, model: nn.Module) -> float:
        """Measure peak memory usage in MB."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Run forward pass to measure memory
            model(self.dummy_input)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
            return peak_memory
        else:
            return 0.0  # Cannot measure memory on CPU
    
    def count_parameters(self, model: nn.Module) -> float:
        """Count number of trainable parameters in millions."""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params / 1e6
    
    def evaluate_architecture(
        self,
        config: ArchitectureConfig,
        quick_eval: bool = True
    ) -> EvaluationMetrics:
        """Evaluate an architecture configuration."""
        try:
            # Create and move model to device
            model = self.create_model(config).to(self.device)
            
            # Measure latency
            latency = self.measure_latency(model)
            
            # Measure memory usage
            memory = self.measure_memory(model)
            
            # Count parameters
            params = self.count_parameters(model)
            
            # Quick evaluation of perplexity
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            
            total_loss = 0
            num_steps = 10 if quick_eval else self.num_eval_steps
            
            for _ in range(num_steps):
                optimizer.zero_grad()
                outputs = model(self.dummy_input)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    self.dummy_labels.view(-1)
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            perplexity = torch.exp(torch.tensor(total_loss / num_steps)).item()
            
            return EvaluationMetrics(
                perplexity=perplexity,
                latency_ms=latency,
                memory_mb=memory,
                params_m=params
            )
            
        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            # Return poor metrics for failed architectures
            return EvaluationMetrics(
                perplexity=float('inf'),
                latency_ms=float('inf'),
                memory_mb=float('inf'),
                params_m=float('inf')
            )

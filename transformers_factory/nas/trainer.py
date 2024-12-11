import torch
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import json
from pathlib import Path

from .controller import ArchitectureController, ReinforcementLearner
from .evaluator import ModelEvaluator, EvaluationMetrics
from .search_space import SearchSpace, ArchitectureConfig, create_default_search_space

class NASTrainer:
    """Neural Architecture Search trainer for transformer models."""
    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        controller_hidden_size: int = 128,
        controller_num_layers: int = 2,
        controller_temperature: float = 1.0,
        rl_learning_rate: float = 0.001,
        entropy_weight: float = 0.0001,
        baseline_decay: float = 0.999,
        eval_batch_size: int = 32,
        eval_seq_length: int = 512,
        num_eval_steps: int = 100,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.search_space = search_space or create_default_search_space()
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.controller = ArchitectureController(
            self.search_space,
            hidden_size=controller_hidden_size,
            num_layers=controller_num_layers,
            temperature=controller_temperature
        ).to(device)
        
        self.rl_learner = ReinforcementLearner(
            self.controller,
            learning_rate=rl_learning_rate,
            entropy_weight=entropy_weight,
            baseline_decay=baseline_decay
        )
        
        self.evaluator = ModelEvaluator(
            eval_batch_size=eval_batch_size,
            eval_seq_length=eval_seq_length,
            num_eval_steps=num_eval_steps,
            device=device
        )
        
        # History tracking
        self.history = {
            "rewards": [],
            "best_reward": float('-inf'),
            "best_config": None,
            "best_metrics": None
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "controller_state": self.controller.state_dict(),
            "history": self.history
        }
        
        checkpoint_path = self.checkpoint_dir / f"nas_checkpoint_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best configuration
        if self.history["best_config"] is not None:
            best_config_path = self.checkpoint_dir / "best_config.json"
            with open(best_config_path, 'w') as f:
                json.dump({
                    "config": vars(self.history["best_config"]),
                    "metrics": vars(self.history["best_metrics"])
                }, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.controller.load_state_dict(checkpoint["controller_state"])
        self.history = checkpoint["history"]
        return checkpoint["epoch"]
    
    def train(
        self,
        num_epochs: int = 100,
        architectures_per_epoch: int = 10,
        checkpoint_frequency: int = 10,
        resume_from: Optional[str] = None
    ):
        """Train the architecture search controller."""
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            self.logger.info(f"Resumed training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, num_epochs):
            self.logger.info(f"Starting epoch {epoch}")
            
            # Generate and evaluate architectures
            configs, log_probs = self.controller(architectures_per_epoch)
            rewards = []
            
            for config in tqdm(configs, desc="Evaluating architectures"):
                # Skip invalid configurations
                if not config.validate(self.search_space):
                    rewards.append(0.0)
                    continue
                
                # Evaluate architecture
                metrics = self.evaluator.evaluate_architecture(config, quick_eval=True)
                reward = metrics.compute_reward()
                rewards.append(reward)
                
                # Update best architecture
                if reward > self.history["best_reward"]:
                    self.history["best_reward"] = reward
                    self.history["best_config"] = config
                    self.history["best_metrics"] = metrics
                    self.logger.info(f"New best architecture found! Reward: {reward:.4f}")
                    self.logger.info(f"Config: {config}")
                    self.logger.info(f"Metrics: {metrics}")
            
            # Update controller
            loss = self.rl_learner.step(torch.tensor(rewards, device=self.device))
            
            # Log progress
            avg_reward = sum(rewards) / len(rewards)
            self.history["rewards"].append(avg_reward)
            self.logger.info(
                f"Epoch {epoch} - Avg Reward: {avg_reward:.4f}, "
                f"Loss: {loss:.4f}, Best Reward: {self.history['best_reward']:.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_frequency == 0:
                self.save_checkpoint(epoch)
        
        # Final save
        self.save_checkpoint(num_epochs - 1)
        self.logger.info("Training completed!")
        
        return self.history["best_config"], self.history["best_metrics"]

def train_nas(config_path: Optional[str] = None, **kwargs):
    """Convenience function to train NAS."""
    # Load custom search space if provided
    search_space = None
    if config_path:
        with open(config_path) as f:
            config = json.load(f)
            search_space = SearchSpace(**config)
    
    # Create and train NAS
    trainer = NASTrainer(search_space=search_space, **kwargs)
    best_config, best_metrics = trainer.train()
    
    return best_config, best_metrics

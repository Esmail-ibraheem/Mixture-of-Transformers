from dataclasses import dataclass
from typing import List, Optional, Union, Dict
from enum import Enum

class ModelType(Enum):
    DECODER_ONLY = "decoder_only"
    ENCODER_DECODER = "encoder_decoder"
    RETRIEVAL_AUGMENTED = "retrieval_augmented"

class AttentionType(Enum):
    STANDARD = "standard"
    MULTI_QUERY = "multi_query"
    SLIDING_WINDOW = "sliding_window"
    SPARSE = "sparse"
    LINEAR = "linear"
    MEMORY_BASED = "memory_based"

class FFNType(Enum):
    STANDARD = "standard"
    GEGLU = "geglu"
    MOE = "moe"
    SWIGLU = "swiglu"

class NormType(Enum):
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"

@dataclass
class SearchSpace:
    """Defines the search space for transformer architectures."""
    # Core architecture choices
    model_types: List[ModelType]
    attention_types: List[AttentionType]
    ffn_types: List[FFNType]
    norm_types: List[NormType]
    
    # Structural parameters
    min_layers: int = 2
    max_layers: int = 48
    min_heads: int = 4
    max_heads: int = 64
    min_hidden_size: int = 256
    max_hidden_size: int = 8192
    min_intermediate_ratio: float = 2.0
    max_intermediate_ratio: float = 8.0
    
    # MoE specific parameters
    min_experts: int = 4
    max_experts: int = 32
    min_experts_per_token: int = 1
    max_experts_per_token: int = 4
    
    # Attention specific parameters
    min_window_size: int = 128
    max_window_size: int = 8192
    min_kv_heads: int = 1
    max_kv_heads: Optional[int] = None  # If None, equals num_heads
    
    # Training parameters
    min_dropout: float = 0.0
    max_dropout: float = 0.3
    learning_rates: List[float] = (1e-4, 3e-4, 1e-3)
    
    def get_discrete_choices(self) -> Dict[str, List]:
        """Returns all discrete choices in the search space."""
        return {
            "model_type": [t.value for t in self.model_types],
            "attention_type": [t.value for t in self.attention_types],
            "ffn_type": [t.value for t in self.ffn_types],
            "norm_type": [t.value for t in self.norm_types],
            "learning_rate": self.learning_rates
        }
    
    def get_continuous_ranges(self) -> Dict[str, tuple]:
        """Returns all continuous parameter ranges."""
        return {
            "num_layers": (self.min_layers, self.max_layers),
            "num_heads": (self.min_heads, self.max_heads),
            "hidden_size": (self.min_hidden_size, self.max_hidden_size),
            "intermediate_ratio": (self.min_intermediate_ratio, self.max_intermediate_ratio),
            "num_experts": (self.min_experts, self.max_experts),
            "experts_per_token": (self.min_experts_per_token, self.max_experts_per_token),
            "window_size": (self.min_window_size, self.max_window_size),
            "dropout": (self.min_dropout, self.max_dropout)
        }

@dataclass
class ArchitectureConfig:
    """Configuration for a specific architecture in the search space."""
    model_type: ModelType
    attention_type: AttentionType
    ffn_type: FFNType
    norm_type: NormType
    num_layers: int
    num_heads: int
    hidden_size: int
    intermediate_size: int
    dropout: float
    learning_rate: float
    
    # Optional parameters based on choices
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None
    window_size: Optional[int] = None
    num_kv_heads: Optional[int] = None
    
    def validate(self, search_space: SearchSpace) -> bool:
        """Validates that the configuration is within the search space."""
        if self.model_type not in search_space.model_types:
            return False
        if self.attention_type not in search_space.attention_types:
            return False
        if self.ffn_type not in search_space.ffn_types:
            return False
        if self.norm_type not in search_space.norm_types:
            return False
            
        ranges = search_space.get_continuous_ranges()
        if not (ranges["num_layers"][0] <= self.num_layers <= ranges["num_layers"][1]):
            return False
        if not (ranges["num_heads"][0] <= self.num_heads <= ranges["num_heads"][1]):
            return False
        if not (ranges["hidden_size"][0] <= self.hidden_size <= ranges["hidden_size"][1]):
            return False
        if not (ranges["dropout"][0] <= self.dropout <= ranges["dropout"][1]):
            return False
            
        # Validate optional parameters
        if self.ffn_type == FFNType.MOE:
            if self.num_experts is None or self.experts_per_token is None:
                return False
            if not (ranges["num_experts"][0] <= self.num_experts <= ranges["num_experts"][1]):
                return False
            if not (ranges["experts_per_token"][0] <= self.experts_per_token <= ranges["experts_per_token"][1]):
                return False
                
        if self.attention_type == AttentionType.SLIDING_WINDOW:
            if self.window_size is None:
                return False
            if not (ranges["window_size"][0] <= self.window_size <= ranges["window_size"][1]):
                return False
                
        return True

def create_default_search_space() -> SearchSpace:
    """Creates a default search space with common architectural choices."""
    return SearchSpace(
        model_types=[ModelType.DECODER_ONLY, ModelType.ENCODER_DECODER],
        attention_types=[
            AttentionType.STANDARD,
            AttentionType.MULTI_QUERY,
            AttentionType.SLIDING_WINDOW
        ],
        ffn_types=[FFNType.STANDARD, FFNType.GEGLU, FFNType.SWIGLU],
        norm_types=[NormType.LAYER_NORM, NormType.RMS_NORM],
        min_layers=4,
        max_layers=32,
        min_heads=8,
        max_heads=32,
        min_hidden_size=512,
        max_hidden_size=4096
    )

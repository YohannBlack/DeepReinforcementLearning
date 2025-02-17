from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional


class Environment(ABC):
    def __init__(self):
        self.current_state = None
        self.episode_steps = 0
        self.total_reward = 0.0

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment to initial state.
        
        Returns:
            Initial state observation
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and advance environment one timestep.
        
        Args:
            action: Agent's action
            
        Returns:
            Tuple containing:
            - Next state observation
            - Reward
            - Done flag
            - Info dictionary with additional data
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> Dict:
        """Define action space properties."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Dict:
        """Define observation space properties."""
        pass

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render current environment state.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            None for 'human' mode, numpy array for 'rgb_array' mode
        """
        raise NotImplementedError

    def close(self):
        """Clean up environment resources."""
        pass

    def seed(self, seed: int = None):
        """Set random seed for reproducibility."""
        pass

import torch
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def sample(self, *args, **kwargs):
        """Abstract method for sampling from the model."""
        pass

    @abstractmethod
    def compute_log_p_zs_given_zt(self, *args, **kwargs):
        """Abstract method for computing log probability of zs given zt."""
        pass

    @abstractmethod
    def sample_p_zs_given_zt(self, *args, **kwargs):
        """Abstract method for sampling zs given zt."""
        pass


from abc import ABC, abstractmethod

from verl_diffusion.protocol import DataProto

__all__ = ["BaseRollout"]


class BaseRollout(ABC):
    def __init__(self):
        """

        Args:
            dataloader: an Iterable of TensorDict that consistently generates prompts. Note that the dataloader
            should handle when the training stops.
        """
        super().__init__()

    @abstractmethod
    def generate_samples(self, prompts: DataProto) -> dict:
        """Generate sequences"""
        pass
from abc import ABC, abstractmethod


__all__ = ["BaseRollout"]


class BaseReward(ABC):
    def __init__(self):
        """

        Args:
            dataloader: an Iterable of TensorDict that consistently generates prompts. Note that the dataloader
            should handle when the training stops.
        """
        super().__init__()

    @abstractmethod
    def calculate_rewards(self, data: dict) -> dict:
        """calculate_rewards"""
        pass
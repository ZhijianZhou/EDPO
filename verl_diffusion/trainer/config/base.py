import os
import yaml
from typing import Dict, Any


class BaseConfig:
    def __init__(self):
        """Initialize base configuration"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseConfig':
        """Load configuration from YAML file
        
        Args:
            yaml_path (str): Path to YAML configuration file
            
        Returns:
            BaseConfig: Configuration object
            
        Raises:
            FileNotFoundError: If YAML file does not exist
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
            
        return config

    def save_yaml(self, save_path: str) -> None:
        """Save configuration to YAML file
        
        Args:
            save_path (str): Path to save YAML configuration file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

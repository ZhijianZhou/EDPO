# RLPF: Reinforcement Learning with Physical Feedback

A flexible framework for implementing reinforcement learning with physical feedback using diffusion models, built on top of the verl_diffusion framework.

## Overview

The RLPF (Reinforcement Learning with Physical Feedback) framework provides a structured approach to implementing diffusion models with reinforcement learning capabilities and physical feedback mechanisms. It's designed to be modular and extensible, allowing you to integrate your own models, reward functions, and training procedures while leveraging the robust verl_diffusion foundation.

## Framework Structure

The RLPF framework is organized into several key components, built on the verl_diffusion architecture:

- **Model**: Defines the diffusion model architecture
- **Trainer**: Handles the training process
- **Worker**: Contains components for rollouts, rewards, and filtering
- **Utils**: Helper functions and utilities
- **Dataloader**: Data loading and preprocessing

## 📦 Installation

### 1. Setup Python Environment

```bash
conda create -n RLPF python=3.10.14
conda activate RLPF
pip install -r requirements.txt
```

### 2. Install xtb for Force Calculations (Used in Reward)

```bash
cd RLPF
git clone https://github.com/grimme-lab/xtb-python.git
cd xtb-python
conda install mkl mkl-devel
conda install -c conda-forge "gfortran<12"
conda install -c conda-forge mkl mkl-devel blas lapack
pip install .
cd ..
```

## 🚀 How to use RLPF to finetune EDM with xtb reward 
### 1. Set up EDM environment
```bash
cd ./Model/EDM
pip install .
```
### 2. For QM9 molecule generation
```bash
bash example/edm_ddpo_xtb/run.sh
```

### 3. Eval on QM9 molecule generation
```bash
bash example/edm_ddpo_xtb/run_eval.sh
```

## Getting Started with RLPF

The RLPF framework allows you to fine-tune diffusion models like EDM using physical feedback rewards such as XTB (Extended Tight Binding) calculations. Here's how to get started:

### 1. Define Your Custom Model

To use your own model with the RLPF framework, you need to create a class that inherits from `BaseModel` in the verl_diffusion package:

```python
from verl_diffusion.model.base import BaseModel

class YourCustomModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        # Initialize your model components
        
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context=None, fix_noise=False, timestep=1000):
        """
        Draw samples from your generative model.
        
        Args:
            n_samples: Number of samples to generate
            n_nodes: Number of nodes per sample
            node_mask: Mask for nodes
            edge_mask: Mask for edges
            context: Optional context information
            fix_noise: Whether to use fixed noise
            timestep: Number of diffusion timesteps
            
        Returns:
            Generated samples and associated information
        """
        # Implement your sampling logic
        pass
        
    def compute_log_p_zs_given_zt(self, x, mu, sigma, node_mask=None):
        """
        Compute log probability of zs given zt.
        
        Args:
            x: Input tensor
            mu: Mean tensor
            sigma: Standard deviation tensor
            node_mask: Optional node mask
            
        Returns:
            Log probability values
        """
        # Implement your probability computation
        pass
        
    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context=None, fix_noise=False):
        """
        Sample from p(zs | zt).
        
        Args:
            s: Source timestep
            t: Target timestep
            zt: Latent at timestep t
            node_mask: Node mask
            edge_mask: Edge mask
            context: Optional context
            fix_noise: Whether to use fixed noise
            
        Returns:
            Sampled values and associated information
        """
        # Implement your conditional sampling
        pass
```

### 2. Define Your Physical Feedback Reward Function

Create a reward function that inherits from the base reward class:

```python
from verl_diffusion.worker.reward.base import BaseReward

class YourCustomReward(BaseReward):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your reward components
        
    def calculate_rewards(self, samples):
        """
        Calculate rewards for generated samples.
        
        Args:
            samples: Generated samples
            
        Returns:
            Reward values
        """
        # Implement your reward calculation
        pass
```

### 3. Configure Your RLPF Training Process

Create a configuration file that specifies your RLPF training parameters and physical feedback settings:

```python
config = {
    "model": {
        "diffusion_steps": 1000,
        "diffusion_noise_schedule": "cosine",
        "diffusion_noise_precision": 1e-5,
        "diffusion_loss_type": "l2",
        "normalize_factors": [1.0, 1.0, 1.0],
        "include_charges": True
    },
    "train": {
        "batch_size": 32,
        "micro_batch_size": 8,
        "learning_rate": 1e-4,
        "clip_advantage_value": 5.0,
        "save_path": "./exp/your_model"
    },
    "dataloader": {
        "epoches": 100
    },
    "wandb": {
        "enabled": True,
        "project": "your-project",
        "name": "your-run-name"
    }
}
```

### 4. Set Up the RLPF Training Pipeline

```python
from verl_diffusion.trainer.ddpo_trainer import DDPOTrainer
from verl_diffusion.dataloader.dataloader import EDMDataLoader

# Initialize your components
model = YourCustomModel(config)
dataloader = EDMDataLoader(config)
rollout = YourCustomRollout(config, model)
rewarder = YourCustomReward(config)
actor = YourCustomActor(config)

# Create the trainer
trainer = DDPOTrainer(
    config=config,
    model=model,
    dataset_info=dataset_info,
    device=device,
    dataloader=dataloader,
    rollout=rollout,
    rewarder=rewarder,
    actor=actor
)

# Start training
trainer.fit()
```

## Key Components of RLPF

The RLPF framework leverages the following key components from verl_diffusion while adding physical feedback capabilities:

### DataProto

The framework uses `DataProto` for data exchange between components. It provides a standardized way to handle tensor and non-tensor data:

```python
from verl_diffusion.protocol import DataProto

# Create a DataProto
data = DataProto.from_dict(
    tensors={"key1": tensor1, "key2": tensor2},
    non_tensors={"key3": array1},
    meta_info={"info": "metadata"}
)

# Access data
tensor_data = data.batch["key1"]
non_tensor_data = data.non_tensor_batch["key3"]
meta_data = data.meta_info["info"]
```

### Rollout

The rollout component handles the generation process:

```python
from verl_diffusion.worker.rollout.base import BaseRollout

class YourCustomRollout(BaseRollout):
    def generate_minibatch(self, batch):
        """
        Generate samples for a mini-batch.
        
        Args:
            batch: Input batch data
            
        Returns:
            Generated samples
        """
        # Implement your generation logic
        pass
```

### Actor

The actor component defines the policy for action selection:

```python
from verl_diffusion.worker.actor.base import BaseActor

class YourCustomActor(BaseActor):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your actor components
        
    def act(self, state):
        """
        Select actions based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected actions
        """
        # Implement your action selection logic
        pass
```
## Advanced Usage

### Custom Filters

You can implement custom filters to process generated samples:

```python
from verl_diffusion.worker.filter.base import BaseFilter

class YourCustomFilter(BaseFilter):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your filter components
        
    def filter(self, samples):
        """
        Filter generated samples.
        
        Args:
            samples: Generated samples
            
        Returns:
            Filtered samples
        """
        # Implement your filtering logic
        pass
```

### Parallel Processing

The framework supports parallel processing using Ray:

```python
import ray

# Initialize Ray
ray.init()

# Use Ray for parallel processing in your components
@ray.remote
def parallel_process(data):
    # Process data in parallel
    pass
```

## Best Practices for RLPF

1. **Model Design**:
   - Ensure your model implements all required methods from `BaseModel`
   - Use appropriate masking for variable-sized inputs
   - Implement proper normalization and denormalization

2. **Physical Feedback Reward Function**:
   - Design rewards that incorporate meaningful physical properties (e.g., XTB energy, stability)
   - Normalize rewards to prevent training instability
   - Consider using reward shaping for better learning with physical constraints

3. **RLPF Training Process**:
   - Use appropriate batch sizes for your hardware
   - Monitor training metrics with wandb
   - Implement proper checkpointing and model saving
   - Balance physical feedback frequency with computational cost

4. **Data Handling**:
   - Use `DataProto` for consistent data exchange
   - Implement proper batching and chunking
   - Handle variable-sized inputs with appropriate masking

## Troubleshooting

Common issues and solutions:

1. **Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Implement proper memory management

2. **Training Instability**:
   - Check reward normalization
   - Adjust learning rate
   - Monitor advantage clipping

3. **Slow Training**:
   - Use appropriate batch sizes
   - Implement parallel processing
   - Optimize data loading

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 

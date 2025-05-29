import torch
import queue
import time
import threading
import ray
from tqdm import tqdm
import wandb
import os
import yaml
from verl_diffusion.trainer.base import BaseTrainer
from verl_diffusion.protocol import DataProto


class DDPOTrainer(BaseTrainer):
    def __init__(self, config, model, dataset_info, device, dataloader, rollout, rewarder, actor, filters=None):
        """
        Initialize the DDPO Trainer
        
        Args:
            config: Configuration dictionary
            model: The EDM model to use for generation
            dataset_info: Dataset information
            device: The device to run the model on
            dataloader: DataLoader for generating prompts
            rollout: The EDMRollout instance
            rewarder: The ForceReward instance
        """
        # Call parent constructor with required arguments
        super().__init__(config=config, rollout=rollout, reward=rewarder)
        
        # Store additional parameters
        self.model = model
        self.dataset_info = dataset_info
        self.device = device
        self.dataloader = dataloader
        self.config = config
        self.save_path = os.path.join("./exp",config["wandb"]["wandb_name"])
        os.makedirs(self.save_path, exist_ok=True)
        self.epoches = self.config["dataloader"]["epoches"]
        # Use provided rollout and rewarder instances
        self.rollout = rollout
        self.rewarder = rewarder
        
        # Create queue between rollout and reward calculation
        self.samples_queue = queue.Queue(maxsize=config.get("queue_size", 256))
        
        # List to store reward calculation results
        self.reward_results = []
        self.filters = filters
        self.actor = actor
        self.best_reward = float('-inf')
        
        # Initialize Ray for parallel processing
        if not ray.is_initialized():
            ray.init()
            
        # Initialize wandb if enabled in config
        if config.get("wandb", False):
            wandb.init(
                project=config["wandb"].get("wandb_project", "edm-ddp"),
                name=config["wandb"].get("wandb_name", "edm-ddp-run"),
                config=config
            )
            self.wandb_enabled = True
        else:
            self.wandb_enabled = False
    
    def _reward_worker(self):
        """Reward calculation worker thread function"""
        while True:
            try:
                # Get sample from queue
                sample = self.samples_queue.get(timeout=1.0)
                
                # Check for termination signal
                if sample is None:
                    self.samples_queue.task_done()
                    break
                
                # Calculate rewards

                result = self.rewarder.calculate_rewards(sample)
            
                
                # Store results
                self.reward_results.append(result)
                
                
                # Mark task as done
                self.samples_queue.task_done()
                
            except queue.Empty:
                # Queue is empty, continue waiting
                continue
            except Exception as e:
                print(f"Error in reward calculation: {e}")
                self.samples_queue.task_done()
    
    def process_batch(self, batch_idx, prompts):
        """
        Process a single batch from the dataloader
        
        Args:
            batch_idx: Batch index
            prompts: Batch data from dataloader
            
        Returns:
            List of reward results for this batch
        """
        
        # Start reward calculation thread
        self.reward_results = []  # Clear previous results
        reward_thread = threading.Thread(target=self._reward_worker, daemon=True)
        reward_thread.start()
        
        try:
            batch_size = prompts.batch.batch_size[0]
            num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
            
            # Split into mini-batches
            batch_prompts = prompts.chunk(chunks=num_chunks)
            
            # Initialize a list to store all sample results as DataProto objects
            sample_results = []
            # Process each mini-batch
            for chunk_idx, batch in enumerate(batch_prompts):
            
                
                # Generate sample using rollout
                sample = self.rollout.generate_minibatch(batch)
                
                # Add batch identifier
                sample_results.append(sample) 
                
                # Queue sample for reward calculation
                # This will block if queue is full, until reward calculation catches up
                self.samples_queue.put(sample)
                
            
            # Wait for all samples in this batch to complete reward calculation
            self.samples_queue.join()
            
            # Processing complete, send termination signal to reward worker
            self.samples_queue.put(None)
            
            # Wait for reward thread to complete
            reward_thread.join(timeout=10)
            
            # Output statistics for this batch
            sample_results = DataProto.concat(sample_results)
            reward_results = DataProto.concat(self.reward_results)
            reward_results = reward_results.to(self.device)
            return sample_results.union(reward_results)
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Send termination signal (if not already sent)
            try:
                self.samples_queue.put_nowait(None)
            except queue.Full:
                pass
            
            # Wait for reward thread to complete
            if reward_thread.is_alive():
                reward_thread.join(timeout=5)
            
            return []
    def compute_advantage(self, samples):
        group_index = samples.batch["group_index"]
        rewards = samples.batch["rewards"]
        
        # Get unique group indices
        unique_groups = torch.unique(group_index)
        normalized_rewards = torch.zeros_like(rewards)
        # Normalize rewards within each group
        
        for group_id in unique_groups:
            group_mask = (group_index == group_id)
            group_rewards = rewards[group_mask]
            
            # Calculate mean and std for this group
            group_mean = group_rewards.mean()
            group_std = group_rewards.std()

            # Normalize rewards for this group
            normalized_rewards[group_mask] = (group_rewards - group_mean) / (group_std+1e-8)
        
        samples.batch["advantages"] = normalized_rewards
        # Clip advantages to prevent extreme values
        clip_value = self.config["train"].get("clip_advantage_value", 5.0)
        samples.batch["advantages"] = torch.clamp(samples.batch["advantages"], -clip_value, clip_value)
        
        return samples

    def save_checkpoint(self, epoch, metrics=None):
        """Save model checkpoint and config"""
        # Save config
        config_path = os.path.join(self.save_path, 'config.yaml')
        if isinstance(self.config, dict):
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
        else:
            self.config.to_yaml(config_path)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'actor_state_dict': self.actor.model.state_dict(),
            'optimizer_state_dict': self.actor.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_path, 'checkpoint_latest.pth'))
        
        # Save epoch checkpoint
        torch.save(checkpoint, os.path.join(self.save_path, f'checkpoint_epoch_{epoch}.pth'))
        
        # Save best checkpoint if metrics are provided
        if metrics is not None:
            if metrics['reward'] > self.best_reward:
                self.best_reward = metrics['reward']
                
                torch.save(self.model.state_dict(),os.path.join(self.save_path, 'generative_model_ema.npy'))
                torch.save(checkpoint, os.path.join(self.save_path, 'checkpoint_best.pth'))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor.model.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'metrics' in checkpoint:
            self.best_reward = checkpoint['metrics'].get('reward', float('-inf'))
        
        return checkpoint.get('epoch', 0)

    def fit(self):
        """
        Training loop for DDPO - processes each dataloader batch and updates model parameters
        
        Overrides BaseTrainer.fit() method
        """
        try:
            all_results = []
            start_epoch = 0
            global_batch_idx = 0
            # Load checkpoint if resuming
            if self.config.get('resume', False) and self.config.get('checkpoint_path'):
                start_epoch = self.load_checkpoint(self.config['checkpoint_path'])
                print(f"Resuming training from epoch {start_epoch}")
            for epoch in range(start_epoch, self.epoches):
                # Process each batch from the dataloader and update model after each batch
                for batch_idx, prompts in enumerate(self.dataloader):
                    # Process the batch
                    samples = self.process_batch(batch_idx, prompts)
                    samples, filter_ratio, novelty_penalty_ratio = self.filters.filter(samples)
                    samples = self.compute_advantage(samples)
                    metrics = self.actor.update_policy(samples)
                    metrics["reward"] = samples.batch["rewards"].mean().item()
                    metrics["filter_ratio"] = filter_ratio
                    metrics["novelty_penalty_ratio"] = novelty_penalty_ratio
                    metrics["molecule_stability"] = samples.batch['stability'].mean().item()
                    
                    # Save checkpoint periodically
                    if (batch_idx + 1) % self.config.get('save_interval', 10) == 0:
                        self.save_checkpoint(batch_idx, metrics)
                    global_batch_idx += 1
                    # Log metrics to wandb if enabled
                    if self.wandb_enabled:
                        # Combine all metrics into a single log call
                        log_dict = {
                            "train/reward": metrics["reward"],
                            "train/filter_ratio": metrics["filter_ratio"],
                            "train/novelty_penalty_ratio": metrics["novelty_penalty_ratio"],
                            "train/molecule_stability": metrics["molecule_stability"],
                            "train/step": global_batch_idx
                        }
                        # Add any additional metrics from actor update
                        for k, v in metrics.items():
                            if k not in ["reward", "filter_ratio", "novelty_penalty_ratio", "molecule_stability"]:
                                log_dict[f"train/{k}"] = v
                        wandb.log(log_dict)
                    
                    print(metrics)
                
                # Save checkpoint at the end of each epoch
                self.save_checkpoint(epoch, metrics)
                print(f"Saved checkpoint at epoch {epoch}")
                
                # Save final checkpoint
                self.save_checkpoint(len(self.dataloader), metrics)
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            # Save checkpoint on interruption
            if 'metrics' in locals():
                self.save_checkpoint(batch_idx, metrics)
            
        return self.model
    
    def clean_up(self):
        """Clean up resources"""
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            
        # Finish wandb run if enabled
        if self.wandb_enabled:
            wandb.finish() 
from typing import Dict
from tqdm import tqdm as tq
from collections import defaultdict
from verl_diffusion.protocol import DataProto
from verl_diffusion.utils.math import kl_divergence_normal
from .base import BaseActor
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np

class EDMActor(BaseActor):
    def __init__(self, model, config):
        """
        Initialize the EDM rollout worker.
        
        Args:
            model: The EDM model to use for rollout
            config: Configuration parameters for the rollout
        """
        super().__init__()  
        self.model = model
        self.config = config
        optimizer_cls = torch.optim.AdamW
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.config["train"]["learning_rate"],
            betas=(self.config["train"]["adam_beta1"], self.config["train"]["adam_beta2"]),
            weight_decay=self.config["train"]["adam_weight_decay"],
            eps=self.config["train"]["adam_epsilon"],
        )
        self.train_micro_batch_size = self.config["train"]["train_micro_batch_size"]
        self.max_grad_norm = self.config["train"]["max_grad_norm"]
        self.clip_range = self.config["train"]["clip_range"]
        self.num_timesteps = self.config["model"]["time_step"]
    def train_batched_samples(self, batched_samples):
        """
        Train on a batch of samples. Main training segment

        Args:
            global_step (int): The current global step
            batched_samples (list[dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """
        info = defaultdict(list)
        self.T = self.num_timesteps + 1
        for _i, sample in tq(enumerate(batched_samples),desc= "Training", unit="Batch",leave=False):

            for j in tq(range(self.T ),desc= "Training Batch", unit="timesteps",leave=False):
                if self.condition :
                    context = sample["context"]
                else:
                    context = None
                loss, clipfrac = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["logps"][:, j],
                        sample["advantages"],
                        sample["nodesxsample"],
                        context = context
                    )
                info["clipfrac"].append(clipfrac.item())
                info["loss"].append(loss.item())
                loss.backward()
                clip_grad_norm_(self.model.parameters(),max_norm=1)
        self.optimizer.step()
        self.optimizer.zero_grad()
            
        metric = {}
        metric["ClipFrac"] = np.mean(np.array(info["clipfrac"]))
        metric["Loss"] = np.mean(np.array(info["loss"]))

        return metric
    
    def calculate_loss(self, latents, timesteps, next_latents, log_prob_old, advantages, nodesxsample, context = None):
        """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            log_prob (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            context (torch.Tensor):
                The embedding of context.
        Returns:
            loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor)
            (all of these are of shape (1,))
        """
        s_array = timesteps
        t_array = s_array + 1
        s_array = s_array / self.num_timesteps
        t_array = t_array / self.num_timesteps
        ## need add
        node_mask, edge_mask = self.model.get_mask(nodesxsample, latents.shape[0], self.max_n_nodes)
        node_mask = node_mask.to(latents.device)
        edge_mask = edge_mask.to(latents.device)

        _, log_prob_current, _, _ = self.model.sample_p_zs_given_zt(s_array, t_array, latents, node_mask, edge_mask, prev_sample=next_latents,context=context)
    
        ## log_prob_current is old latents in new policy
        
        # compute the log prob of next_latents given latents under the current model
        dif_logp = (log_prob_current - log_prob_old)
        ratio = torch.exp(dif_logp)
        loss = self.loss(advantages, self.clip_range, ratio)
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float())
    
        return loss, clipfrac
    
    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss =    -1.0 * advantages * ratio
        clipped_loss =   -1.0  * advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))
    
    def update_policy(self, data: DataProto) -> Dict:
        samples = {}
        self.max_n_nodes = data.meta_info["max_n_nodes"]
        self.batch_size = data.batch.batch_size[0]
        self.condition = data.meta_info["condition"]
        samples["advantages"] = data.batch["advantages"]
        
        samples["latents"] = data.batch["latents"][:,:self.num_timesteps+1].clone()
        samples["next_latents"] = data.batch["latents"][:,1:].clone()
        samples["timesteps"] = data.batch["timesteps"]
        samples["logps"] = data.batch["logps"]
        samples["nodesxsample"] = data.batch["nodesxsample"]
        if self.condition:
           samples["context"] = data.batch["context"]

        original_keys = samples.keys()
        original_values = samples.values()
        # rebatch them as user defined train_batch_size is different from sample_batch_size
        samples_batched = []
        
        if self.train_micro_batch_size >= self.batch_size:
            # If train_micro_batch_size is larger than or equal to batch_size,
            # just use the entire batch as one micro batch
            samples_batched.append(samples)
        else:
            # Calculate number of complete batches and remaining samples
            n_complete_batches = self.batch_size // self.train_micro_batch_size
            remaining_samples = self.batch_size % self.train_micro_batch_size
            
            # Process complete batches
            if n_complete_batches > 0:
                reshaped_values = [v.reshape(-1, self.train_micro_batch_size, *v.shape[1:]) for v in original_values]
                transposed_values = zip(*reshaped_values)
                samples_batched.extend([dict(zip(original_keys, row_values)) for row_values in transposed_values])
            
            # Process remaining samples if any
            if remaining_samples > 0:
                remaining_values = [v[-remaining_samples:].unsqueeze(0) for v in original_values]
                samples_batched.append(dict(zip(original_keys, remaining_values)))
        
        # Train each batch
        metric = self.train_batched_samples(samples_batched)
        
        return metric
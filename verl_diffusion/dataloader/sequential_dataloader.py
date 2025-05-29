import torch
from verl_diffusion.dataloader.dataloader import EDMDataLoader
from verl_diffusion.protocol import DataProto, TensorDict

class SequentialDataLoader(EDMDataLoader):
    def __init__(self, config, dataset_info, nodes_dist, prop_dist, device, condition=False, num_batches=None):
        """
        Initialize the Sequential dataloader that iterates through nodes 8-29
        
        Args:
            config: Configuration object containing sampling parameters
            dataset_info: Dictionary containing dataset information
            nodes_dist: Distribution for sampling nodes (not used in this implementation)
            prop_dist: Distribution for sampling properties
            device: Device to put tensors on
            condition: Whether to generate conditional samples
            num_batches: Number of batches per epoch (if None, will run indefinitely)
        """
        # Calculate number of batches needed to iterate through nodes 14-29
        total_nodes = 29 - 18 + 1  # Total number of nodes to iterate through
        group_sample = config["dataloader"]["sample_group_size"]
        num_batches = (total_nodes + group_sample - 1) // group_sample  # Ceiling division
        
        super().__init__(config, dataset_info, nodes_dist, prop_dist, device, condition, num_batches)
        self.current_node = 14  # Start from 8 nodes
        self.max_node = 21     # End at 29 nodes
        
    def generate_batch(self):
        """
        Generate a batch of data with sequential node numbers
        
        Returns:
            DataProto: DataProto containing the generated batch data
        """
        tensors = {}
        max_n_nodes = self.dataset_info['max_n_nodes']
        
        # Create sequential nodesxsample for each group
        remaining_nodes = self.max_node - self.current_node + 1
        current_group_sample = min(self.group_sample, remaining_nodes)
        nodesxsample = torch.arange(self.current_node, self.current_node + current_group_sample, device=self.device)
        # Update current_node for next batch
        self.current_node += current_group_sample
        if self.current_node > self.max_node:
            self.current_node = 18  # Reset to 8 when reaching max
            
        group_index = torch.tensor(range(current_group_sample))
        group_index = group_index.repeat_interleave(self.each_prompt_sample, dim=0)
        group_index = group_index.to(self.device)
        self.batch_size = self.each_prompt_sample * current_group_sample
        
        if self.condition:
            context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
            context = context.repeat_interleave(self.each_prompt_sample, dim=0)
            nodesxsample = nodesxsample.repeat_interleave(self.each_prompt_sample, dim=0)
            # Store tensors
            tensors.update({
                'nodesxsample': nodesxsample,
                'context': context,
                'group_index': group_index,
            })
        else:
            nodesxsample = nodesxsample.repeat_interleave(self.each_prompt_sample, dim=0)
            
            # Store tensors
            tensors.update({
                'nodesxsample': nodesxsample.to(self.device),
                'group_index': group_index
            })
        
        # Create meta info
        meta_info = {
            'max_n_nodes': max_n_nodes,
            'condition': self.condition
        }
            
        # Create DataProto
        return DataProto(batch=TensorDict(tensors, batch_size=[self.batch_size]), meta_info=meta_info)
        
    def reset(self):
        """Reset the iterator and node counter"""
        super().reset()
        self.current_node = 8 
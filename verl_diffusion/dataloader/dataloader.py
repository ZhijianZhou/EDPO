import torch
from verl_diffusion.protocol import DataProto, TensorDict

class EDMDataLoader:
    def __init__(self, config, dataset_info, nodes_dist, prop_dist, device, condition=False, num_batches=None):
        """
        Initialize the EDM dataloader
        
        Args:
            config: Configuration object containing sampling parameters
            dataset_info: Dictionary containing dataset information
            nodes_dist: Distribution for sampling nodes
            prop_dist: Distribution for sampling properties
            device: Device to put tensors on
            condition: Whether to generate conditional samples
            num_batches: Number of batches per epoch (if None, will run indefinitely)
        """
        self.config = config
        self.group_sample = self.config["dataloader"]["sample_group_size"]
        self.each_prompt_sample = self.config["dataloader"]["each_prompt_sample"]
        self.dataset_info = dataset_info
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.device = device
        self.condition = condition
        self.num_batches = num_batches
        self.current_batch = 0
        
    def __iter__(self):
        """Make the class iterable"""
        self.current_batch = 0
        return self
        
    def __len__(self):
        """Return the number of batches in an epoch"""
        return self.num_batches if self.num_batches is not None else float('inf')
    
    def __next__(self):
        """Get the next batch of data"""
        if self.num_batches is not None and self.current_batch >= self.num_batches:
            raise StopIteration
            
        self.current_batch += 1
        return self.generate_batch()
        
        
    def generate_batch(self):
        """
        Generate a batch of data
        
        Returns:
            DataProto: DataProto containing the generated batch data
        """
        tensors = {}
        max_n_nodes = self.dataset_info['max_n_nodes']
        nodesxsample = self.nodes_dist.sample(self.group_sample)
        group_index = torch.tensor(range(self.group_sample))
        group_index = group_index.repeat_interleave(self.each_prompt_sample, dim=0)
        group_index = group_index.to(self.device)
        self.batch_size = self.each_prompt_sample * self.group_sample
        
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
                'group_index' : group_index
            })
        
        # Create meta info
        meta_info = {
            'max_n_nodes': max_n_nodes,
            'condition': self.condition
        }
            
        # Create DataProto
        return DataProto(batch=TensorDict(tensors, batch_size=[self.batch_size]), meta_info=meta_info)
        
    def reset(self):
        """Reset the iterator"""
        self.current_batch = 0
   
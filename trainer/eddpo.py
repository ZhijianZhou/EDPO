import os
from trainer.eddpo_config import EDDPOConfig
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.models import get_model
import torch
from trl.trainer import BaseTrainer
from os.path import join
from tqdm import tqdm as tq
class EDDPOTrainer(BaseTrainer):
    """
    The EDDPOTrainer uses e3nn diffusion policy optimization to optimise diffusion models.
    Attributes:
        **config**
        **reward_fuction**
        **sd_pipline**
    """
    def __init__(
        self,
        config_path,
    ):
        self.config_path = config_path
        self.config = EDDPOConfig(config_path)
        # ## load edm model
        # self.generate_model = _create_edm_pipline()
        
    def _create_edm_pipline(self):
        '''
        Load EDM model parameters
        
        Returns:
            flow (torch.nn.moudle): The model
        '''
        edm_config = self.config.edm_config
        model_path = self.config_path
        dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)

        dataloaders, charge_scale = dataset.retrieve_dataloaders(edm_config)

        flow, nodes_dist, prop_dist = get_model(edm_config, edm_config.device, dataset_info, dataloaders['train'])
        flow.to(edm_config.device)
        
        fn = 'generative_model_ema.npy' if edm_config.ema_decay > 0 else 'generative_model.npy'
        flow_state_dict = torch.load(join(model_path, fn),
                                    map_location=edm_config.device )
        flow.load_state_dict(flow_state_dict)
        
        self.nodes_dist = nodes_dist
        self.dataset_info = dataset_info
        ## qm9 node distribution ⬆
        
        return flow
    def step(self, epoch: int, global_step: int):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step.

        """
        # 1. generate samples
        samples = self._generate_samples(
            iterations=self.config.sample_num_batches_per_epoch,
            batch_size=self.config.sample_batch_size,
        )
        ## concat samples
        samples_warped = {}
        for key in samples[0].keys():
            if key != "h":
                samples_warped[key] = torch.cat([s[key] for s in samples])
            else:
                samples_warped[key] = {}
                for k in samples[0][key].keys():
                    samples_warped[key][k]  = torch.cat([s[key][k] for s in samples])
        
        # 2. compute rewards
        rewards = self.compute_rewards()
                
    def _batch_samples(self, batch_size, timestep=1000, context=None, fix_noise=False):
        """
        Generate a batch of samples from the model's input distribution and process the resulting data.

        Args:
            batch_size (int): The number of samples to generate in the batch.
            timestep (int, optional): The current timestep for sampling. Defaults to 1000.
            context (optional): Any additional context for the model, if applicable (default is None).
            fix_noise (bool, optional): Whether to fix the noise for reproducibility (default is False).

        Returns:
            dict: A dictionary containing the generated sample results, with keys:
                - "x" (Tensor): The generated node positions, shape [batch_size, max_n_nodes, 3].
                - "h" (dict): A dictionary with keys:
                    - "integer" (Tensor): Integer features for each node, shape [batch_size, max_n_nodes, 1].
                    - "categorical" (Tensor): Categorical features for each node, shape [batch_size, max_n_nodes, 5].
                - "latents" (Tensor): The latent representations of the samples, shape [batch_size, timestep+1, max_n_nodes, 9].
                - "logps" (Tensor): The log-probabilities of the generated samples, shape [batch_size, timestep, max_n_nodes, 9].
                - "nodesxsample" (Tensor): The number of each sample's node.

        Side Effects:
            - Creates node and edge masks based on the node distribution and batch size.
            - Samples from the model's EDM (Energy-based Deep Markov Model) to generate node positions, features, and latent variables.

        Raises:
            AssertionError: If the maximum number of nodes sampled exceeds the dataset's maximum node limit.
        """
        device = self.config.edm_config.device
        # 1. prepare edm input 
        nodesxsample = self.nodes_dist.sample(batch_size)
        max_n_nodes = self.dataset_info['max_n_nodes']  # <- this is the maximum node_size in QM9

        assert int(torch.max(nodesxsample)) <= max_n_nodes

        node_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            node_mask[i, 0:nodesxsample[i]] = 1

        ## Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
        node_mask = node_mask.unsqueeze(2).to(device)
        
        # 2. generate sample from edm
        x, h, latents, logps = self.generate_model.sample_eddpo(
            batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise, timestep=timestep
        )
        
        # 3. warp result
        res = {
            "x": x,
            "h": h,
            "latents": torch.stack(latents, dim=1),
            "logps": torch.stack(logps, dim=1),
            "nodesxsample": nodesxsample
        }
        
        return res
    def _generate_samples(self, iterations, batch_size):
        """
        Generate samples from the model

        Args:
            iterations (int): Number of iterations to generate samples for
            batch_size (int): Batch size to use for sampling

        Returns:
            samples (list[dict[str, torch.Tensor]]), prompt_image_pairs (list[list[Any]])
        """
        self.generate_model = self._create_edm_pipline()
        samples = []
        for _ in tq(range(iterations)):
            sample = self._batch_samples(batch_size)
            samples.append(sample)
        return samples
    
    def compute_rewards(self,):
        return 0
if __name__ == "__main__":
    trainer = EDDPOTrainer("../outputs/edm_qm9")
    trainer._create_edm_pipline()
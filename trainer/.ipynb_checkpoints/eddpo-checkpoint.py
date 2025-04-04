import os
os.environ["OMP_NUM_THREADS"] = "18"
from collections import defaultdict
from trainer.eddpo_config import EDDPOConfig
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.models import get_model
import torch
from trl.trainer import BaseTrainer
from os.path import join
from tqdm import tqdm as tq
from xtb.ase.calculator import XTB
from ase import Atoms
from ase.optimize import BFGS
import numpy as np
from torch.nn.utils import clip_grad_norm_
from qm9.analyze import check_stability
import torch.nn as nn
import torch.nn.init as init
from trainer.reward import qm_reward_model

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
def kl_divergence_normal(mu_P, sigma_P, mu_Q):
    """
    Compute the Kullback-Leibler (KL) divergence between two normal distributions using PyTorch.
    
    Args:
    - mu_P (float or tensor): Mean of the first normal distribution (P)
    - sigma_P (float or tensor): Standard deviation of the first normal distribution (P)
    - mu_Q (float or tensor): Mean of the second normal distribution (Q)
    - sigma_Q (float or tensor): Standard deviation of the second normal distribution (Q)
    
    Returns:
    - float or tensor: KL divergence from P to Q
    """
    # First term: log(sigma_Q / sigma_P)
    
    # Second term: (sigma_P^2 + (mu_P - mu_Q)^2) / (2 * sigma_Q^2)
    term2 = (sigma_P**2 + (mu_P - mu_Q)**2) / (2 * sigma_P**2+1e-8)
    
    # Third term: -1/2
    term3 = -0.5
    
    # Calculate the KL divergence
    kl_div = term2 + term3
    return kl_div

def rmsd(A, B=None):
    """
    计算两个二维矩阵 A 和 B 之间的 RMSD (Root Mean Square Deviation)。
    如果未提供 B，则默认将 B 设为全零矩阵。

    参数:
    A: numpy.ndarray, 形状为 (m, n)
    B: numpy.ndarray, 形状为 (m, n)，默认为 None，若为 None，B 将被设置为全零矩阵
    
    返回:
    float: RMSD 值
    """
    # 如果 B 为 None，则将 B 设为与 A 相同形状的全零矩阵
    if B is None:
        B = np.zeros_like(A)

    # 确保输入矩阵 A 和 B 具有相同的形状
    if A.shape != B.shape:
        raise ValueError("输入的矩阵 A 和 B 必须具有相同的形状")
    
    # 计算矩阵 A 和 B 之间的差异的平方
    diff = A - B
    squared_diff = np.square(diff)
    
    # 计算均方根偏差 (RMSD)
    rmsd_value = np.sqrt(np.mean(squared_diff))
    
    return rmsd_value
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
        self.generate_model = self._create_edm_pipline()
        self.optimizer = self._setup_optimizer(self.generate_model.parameters())
        # if self.config.resume:
        #     self.optimizer.load_state_dict(torch.load(self.config.check_point_optimizer_path,
        #                                 map_location=self.config.edm_config.device ))
        
    def save_model(self,path,name):
        torch.save(self.generate_model.state_dict(), join(path,f"{name}_generative_model.npy"))
        torch.save(self.optimizer.state_dict(), join(path,f"{name}_optimize.npy"))
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
        if self.config.resume :
            flow_state_dict = torch.load(self.config.check_point_model_path,
                                        map_location=edm_config.device )
        else:
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
            ref= False
        )
        
        # 2. compute rewards
        rewards, stables = self.compute_rewards(samples)
        rewards = torch.tensor(rewards,dtype=float).to(samples['x'].device)
        stables = torch.tensor(stables,dtype=float).to(samples['x'].device)
        # print(rewards.mean().item(),rewards.std().item())
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # advantages = (rewards - 0.6504) / (0.7600 + 1e-8)
        # advantages = rewards
        total_batch_size, num_timesteps, max_atoms_num, _ = samples["latents"].shape
        num_timesteps -= 2
        
        # 3. reshape matrix for batch train
        samples["advantages"] = advantages
        samples["next_latents"] = samples["latents"][:,1:num_timesteps+2]
        samples["latents"] = samples["latents"][:,:num_timesteps+1]
        samples["timesteps"] = samples["timesteps"].unsqueeze(0).repeat([total_batch_size,1])
        results = []
        del samples["x"]
        del samples["h"]
        for inner_epoch in range(self.config.train_num_inner_epochs):
            # shuffle batch
            perm = torch.randperm(total_batch_size, device=self.config.device)
            for key in ["timesteps", "latents", "next_latents", "logps",'nodesxsample','advantages','mu','sigma']:
                samples[key] = samples[key][perm]
            # shuffle timesteps
            perms = torch.stack(
                [torch.randperm(num_timesteps+1, device=self.config.device) for _ in range(total_batch_size)]
            )
            for key in ["timesteps", "latents", "next_latents","logps",'mu','sigma']:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=self.config.device)[:, None],
                    perms,
                ]
            original_keys = samples.keys()
            original_values = samples.values()
            # rebatch them as user defined train_batch_size is different from sample_batch_size
            reshaped_values = [v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for v in original_values]
            # Transpose the list of original values
            transposed_values = zip(*reshaped_values)
            # Create new dictionaries for each row of transposed values
            samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]
            # training~
            result = self._train_batched_samples(inner_epoch, epoch, global_step, samples_batched)
            result["Stable"] = stables.mean().item()
            result["Reward"] = rewards.mean().item()
            results.append(result)
            
        return results
    def get_mask(self,nodesxsample,batch_size,device,max_n_nodes):
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
        
        return node_mask, edge_mask
    def _batch_samples(self, batch_size, timestep=1000, context=None, fix_noise=False, ref= False):
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
        node_mask,edge_mask = self.get_mask(nodesxsample,batch_size,device,max_n_nodes)

        # 2. generate sample from edm
        x, h, latents, logps, timestep, mu, sigma = self.generate_model.sample_eddpo(
            batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise, timestep=timestep
        )
       
        # 3. warp result
        res = {
            "x": x,
            "h": h,
            "latents": torch.stack(latents, dim=1),
            "logps": torch.stack(logps, dim=1),
            "nodesxsample": nodesxsample.to(self.config.device),
            "timesteps":torch.tensor(timestep).to(self.config.device),
            "mu": torch.stack(mu, dim=1),
            "sigma": torch.stack(sigma, dim=1)
        }
        
        return res
    def _generate_samples(self, iterations, batch_size, ref = False):
        """
        Generate samples from the model

        Args:
            iterations (int): Number of iterations to generate samples for
            batch_size (int): Batch size to use for sampling

        Returns:
            samples (list[dict[str, torch.Tensor]]), prompt_image_pairs (list[list[Any]])
        """
        
        samples = []
        for _ in tq(range(iterations),desc = "Generate samples", unit = "sample",leave=False):
            sample = self._batch_samples(batch_size, timestep=self.config.num_train_timesteps,ref=ref)
            samples.append(sample)
            
        ## concat samples
        samples_warped = {}
        for key in samples[0].keys():
            if key != "h":
                samples_warped[key] = torch.cat([s[key] for s in samples])
            else:
                samples_warped[key] = {}
                for k in samples[0][key].keys():
                    samples_warped[key][k]  = torch.cat([s[key][k] for s in samples])
        return samples_warped 
    
    # def compute_rewards(self,samples):
    #     '''
    #     use stable as reward fuction
    #     '''
    #     ## encoder ['H','C','O','N','F']
    #     atom_encoder = self.dataset_info['atom_decoder']
    #     one_hot = samples["h"]["categorical"]
    #     x = samples['x']
    #     nodesxsample = samples["nodesxsample"]
        
    #     node_mask = torch.zeros(x.shape[0], self.dataset_info['max_n_nodes'])
    #     for i in range(x.shape[0]):
    #         node_mask[i, 0:nodesxsample[i]] = 1
    #     force = qm_reward_model(one_hot,x,atom_encoder,node_mask,"10.245.164.153",x.shape[0])
    #     n_samples = len(x)
    #     processed_list = []
    #     rewards = []
    #     real_force = []
    #     molecule_stable = []
    #     for i in range(n_samples):
    #         atom_type = one_hot[i].argmax(1).cpu().detach()
    #         pos = x[i].cpu().detach()
    #         atom_type = atom_type[0:int(nodesxsample[i])]
    #         pos = pos[0:int(nodesxsample[i])]
    #         processed_list.append((pos, atom_type))
    #     calc = XTB(method="GFN2-xTB")
    #     for mol in tq(processed_list, desc="Calculate Forces",leave=False):
    #         pos = mol[0].tolist()
    #         atom_type = mol[1].tolist()
    #         validity_results = check_stability(np.array(pos), atom_type, self.dataset_info)
    #         atom_type = [atom_encoder[atom] for atom in atom_type]
    #         molecule_stable.append(int(validity_results[0]))
    #         # if validity_results[0]:
    #         #     rewards.append(1.0)
    #         # else:
    #         #     rewards.append(-1.0)
    #         # # rewards.append(int(validity_results[0]))
    #         atoms = Atoms(symbols=atom_type, positions=pos)
    #         atoms.calc = calc
    #         forces = atoms.get_forces()
    #         mean_abs_forces = rmsd(forces)
    #         # opt1 = BFGS(atoms,
    #         #             trajectory=f'mode_pre_opt.traj',
    #         #             logfile=f"mode_pre_opt.log")
    #         # opt1.run(fmax=0.01)
    #         real_force.append(mean_abs_forces)
    #         rewards.append(-1 * mean_abs_forces)
    #         # if mean_abs_forces < 0.30 and validity_results[0]:
    #         #     rewards.append(1.0)
    #         # else:
    #         #     rewards.append(-1.0)
    #     # print("\n","Rewards:",np.mean(rewards),"Force:",np.mean(real_force),"Stable:", np.mean(molecule_stable))
        
    #     print("\n","Rewards:",np.mean(rewards),"Stable:", np.mean(molecule_stable),"QM_force:",np.mean(force))
    #     return force, molecule_stable
    
    def compute_rewards(self,samples):
        '''
        use stable as reward fuction
        '''
        ## encoder ['H','C','O','N','F']
        atom_encoder = self.dataset_info['atom_decoder']
        one_hot = samples["h"]["categorical"]
        x = samples['x']
        nodesxsample = samples["nodesxsample"]
        
        node_mask = torch.zeros(x.shape[0], self.dataset_info['max_n_nodes'])
        for i in range(x.shape[0]):
            node_mask[i, 0:nodesxsample[i]] = 1
        
        n_samples = len(x)
        processed_list = []
        rewards = []
        real_force = []
        molecule_stable = []
        for i in range(n_samples):
            atom_type = one_hot[i].argmax(1).cpu().detach()
            pos = x[i].cpu().detach()
            atom_type = atom_type[0:int(nodesxsample[i])]
            pos = pos[0:int(nodesxsample[i])]
            processed_list.append((pos, atom_type))
        calc = XTB(method="GFN2-xTB")
        for mol in tq(processed_list, desc="Calculate Forces",leave=False):
            pos = mol[0].tolist()
            atom_type = mol[1].tolist()
            validity_results = check_stability(np.array(pos), atom_type, self.dataset_info)
            atom_type = [atom_encoder[atom] for atom in atom_type]
            molecule_stable.append(validity_results[1]/validity_results[2])
            # if validity_results[0]:
            #     rewards.append(1.0)
            # else:
            #     rewards.append(-1.0)
            # # rewards.append(int(validity_results[0]))
            atoms = Atoms(symbols=atom_type, positions=pos)
            atoms.calc = calc
            try:
                forces = atoms.get_forces()
                mean_abs_forces = rmsd(forces)
            except:
                mean_abs_forces = 5.0
            # opt1 = BFGS(atoms,
            #             trajectory=f'mode_pre_opt.traj',
            #             logfile=f"mode_pre_opt.log")
            # opt1.run(fmax=0.01)
            real_force.append(mean_abs_forces)
            rewards.append(-1 * mean_abs_forces)
            # if mean_abs_forces < 0.30 and validity_results[0]:
            #     rewards.append(1.0)
            # else:
            #     rewards.append(-1.0)
        # print("\n","Rewards:",np.mean(rewards),"Force:",np.mean(real_force),"Stable:", np.mean(molecule_stable))
        print("\n","Rewards:",np.mean(rewards),"Stable:", np.mean(molecule_stable))
        return rewards, molecule_stable
        

    def _train_batched_samples(self, inner_epoch, epoch, global_step, batched_samples):
        """
        Train on a batch of samples. Main training segment

        Args:
            inner_epoch (int): The current inner epoch
            epoch (int): The current epoch
            global_step (int): The current global step
            batched_samples (list[dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """
        info = defaultdict(list)

        self.T = self.config.num_train_timesteps+1
        for _i, sample in tq(enumerate(batched_samples),desc= "Training", unit="Batch",leave=False):
            for j in tq(range(self.T ),desc= "Training Batch", unit="timesteps",leave=False):
                loss, approx_kl, clipfrac = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["logps"][:, j],
                        sample["advantages"],
                        sample["nodesxsample"],
                        sample["mu"][:, j],
                        sample["sigma"][:, j]
                    )
                info["approx_kl"].append(approx_kl.item())
                info["clipfrac"].append(clipfrac.item())
                info["loss"].append(loss.item())
                loss = loss / self.config.num_train_timesteps 
                loss.backward()
            clip_grad_norm_(self.generate_model.parameters(),max_norm=1)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        result = {}
        result["KL"] = np.mean(np.array(info["approx_kl"]))
        result["ClipFrac"] = np.mean(np.array(info["clipfrac"]))
        result["Loss"] = np.mean(np.array(info["loss"]))
        result["GlobalStep"] = global_step + 1
        # print("\n","Kl:",np.mean(np.array(info["approx_kl"])),"ClipFrac:",np.mean(np.array(info["clipfrac"])),"lamda:",np.mean(np.array(info["lamda"])),np.std(np.array(info["lamda"])))
            # # Checks if the accelerator has performed an optimization step behind the scenes
            # if self.accelerator.sync_gradients:
            #     # log training-related stuff
            #     info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
            #     info = self.accelerator.reduce(info, reduction="mean")
            #     info.update({"epoch": epoch, "inner_epoch": inner_epoch})
            #     self.accelerator.log(info, step=global_step)
            #     global_step += 1
            #     info = defaultdict(list)
        return result
    
    def _setup_optimizer(self, trainable_layers_parameters):
        optimizer_cls = torch.optim.AdamW
        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )
        
    def calculate_loss(self, latents, timesteps, next_latents, log_prob_old, advantages, nodesxsample, mu_old, sigma_old, context = None):
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
        s_array = s_array / self.config.num_train_timesteps
        t_array = t_array / self.config.num_train_timesteps
        node_mask, edge_mask = self.get_mask(nodesxsample,self.config.train_batch_size,self.config.device,self.dataset_info['max_n_nodes'])
        ## need credit
        latents, log_prob_current, mu_current, sigma_current = self.generate_model.sample_p_zs_given_zt_eddpo(s_array, t_array, latents, node_mask, edge_mask,prev_sample=next_latents)
        

        ## log_prob is old latents in new policy
        # compute the log prob of next_latents given latents under the current model

        advantages = torch.clamp(
            advantages,
            -self.config.train_adv_clip_max,
            self.config.train_adv_clip_max,
        )
        # import pdb;pdb.set_trace()
        dif_logp = (log_prob_current - log_prob_old)
        
        ratio = torch.exp(dif_logp)
        
        # import pdb;pdb.set_trace()

        kl = torch.mean(kl_divergence_normal(mu_current*node_mask,sigma_current,mu_old*node_mask))
        # approx_kl = 0.5 * torch.mean((log_prob_current - log_prob_old) ** 2)
        
        
        # sigma_old.squeeze(1,2) 
        # lamda = 1.0 / lamda
        # loss =   -1.0 * advantages * ratio + approx_kl 
        # loss = torch.mean(loss)
        # loss = self.loss(advantages, self.config.train_clip_range, ratio,lamda) + (1-torch.mean(lamda))*100.0*kl
        loss = self.loss(advantages, self.config.train_clip_range, ratio)
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())
    
        return loss, kl, clipfrac
    
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
    def eval(self):
        print("Valing...")
        stable_all = []
        for i in range(self.config.val_size//self.config.sample_batch_size):
            samples = self._generate_samples(
                    iterations=self.config.sample_num_batches_per_epoch,
                    batch_size=self.config.sample_batch_size,
                    ref= False
                )
            
                # 2. compute rewards
            _,stable= self.compute_rewards(samples)
            stable = torch.tensor(stable,dtype=float).to(samples['x'].device)
            stable_all.append(stable.mean().item())
        print("eval:",np.mean(stable_all))

if __name__ == "__main__":
    trainer = EDDPOTrainer("../outputs/edm_qm9")
    trainer._create_edm_pipline()
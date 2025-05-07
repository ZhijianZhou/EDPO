import torch
from tqdm import tqdm as tq
from .base import BaseModel
from Model.EDM.qm9.models import get_model
from Model.EDM.equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from torch.nn import functional as F

class EDMModel(BaseModel, EnVariationalDiffusion):
    def __init__(self, model, config):
        # Initialize EnVariationalDiffusion first since it's a nn.Module
        # Get configuration from the existing model
        model_config = {
            'dynamics': model.dynamics,
            'in_node_nf': model.in_node_nf,
            'n_dims': 3,
            'timesteps': config.diffusion_steps,
            'noise_schedule': config.diffusion_noise_schedule,  # Default value from the example
            'noise_precision': config.diffusion_noise_precision,     # Default value
            'loss_type': config.diffusion_loss_type,          # Default value
            'norm_values': config.normalize_factors,           # Default value
            'include_charges': config.include_charges      # Default value
        }
        
        # Initialize EnVariationalDiffusion with the configuration
        EnVariationalDiffusion.__init__(
            self,
            dynamics=model_config['dynamics'],
            in_node_nf=model_config['in_node_nf'],
            n_dims=model_config['n_dims'],
            timesteps=model_config['timesteps'],
            noise_schedule=model_config['noise_schedule'],
            noise_precision=model_config['noise_precision'],
            loss_type=model_config['loss_type'],
            norm_values=model_config['norm_values'],
            include_charges=model_config['include_charges']
        )

        # Initialize BaseModel
        BaseModel.__init__(self)
        # Store the model after parent classes are initialized
        self.model = model
        self.config = config
        
    def get_mask(self, nodesxsample, batch_size, max_n_nodes):
        """
        Generate node and edge masks based on the number of nodes
        
        Args:
            nodesxsample: Number of nodes per sample
            batch_size: Batch size
            max_n_nodes: Maximum number of nodes
            
        Returns:
            tuple: (node_mask, edge_mask)
        """
        # Create node mask
        node_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            node_mask[i, 0:nodesxsample[i]] = 1
        
        # Create edge mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1)
        node_mask = node_mask.unsqueeze(2)
        
        return node_mask, edge_mask
    
    def _assert_mean_zero_with_mask(self, x, node_mask, eps=1e-10):
        self._assert_correctly_masked(x, node_mask)
        largest_value = x.abs().max().item()
        error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        rel_error = error / (largest_value + eps)
        assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'
        
    def _assert_correctly_masked(self, variable, node_mask):
        assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
            'Variables not masked properly.'
            
    def _remove_mean_with_mask(self, x, node_mask):
        masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
        assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
        N = node_mask.sum(1, keepdims=True)

        mean = torch.sum(x, dim=1, keepdim=True) / N
        x = x - mean * node_mask
        return x
    
    def load(self, model_path):
        flow_state_dict = torch.load(model_path, map_location= self.config.device )
        self.model.load_state_dict(flow_state_dict)
        
    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context=None, fix_noise=False, timestep=1000):
        """
        Draw samples from the generative model.
        """
        self.T = timestep
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        self._assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
        latents = []
        logps = []
        timesteps = []
        mus = []
        sigmas = []
        latents.append(z)
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in tq(reversed(range(0, self.T)), desc="sampling", leave=False, unit="step"):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            z, logp, mu, sigma = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)
            latents.append(z)
            logps.append(logp)
            timesteps.append(s)
            mus.append(mu)
            sigmas.append(sigma)
        
        # Finally sample p(x, h | z_0).
        x, h, mu, sigma, logp, s, z = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        latents.append(z)
        logps.append(logp)
        timesteps.append(0)
        mus.append(mu)
        sigmas.append(sigma.unsqueeze(-1))
        
        self._assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = self._remove_mean_with_mask(x, node_mask)

        return x, h, latents, logps, timesteps, mus, sigmas
    
    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context = None, fix_noise=False, prev_sample=None):
        """Samples x ~ p(x|z0)."""
        
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)

        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)

        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        if prev_sample != None :
            # import pdb; pdb.set_trace()
            log_p = self.compute_log_p_zs_given_zt(prev_sample, mu_x, sigma_x, node_mask = node_mask)
        else:
            log_p = self.compute_log_p_zs_given_zt(xh, mu_x, sigma_x, node_mask = node_mask)
        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}

        return x, h, mu_x, sigma_x.squeeze(-1), log_p, zeros, xh

    def compute_log_p_zs_given_zt(self, x, mu, sigma, node_mask = None):
        '''
        Compute log p(zs | zt) for a Gaussian distribution.

        Args:
            x (Tensor): The input tensor (e.g., zs values).
            mu (Tensor): The mean tensor (e.g., zt values).
            sigma (Tensor): The standard deviation tensor (e.g., sigma values).

        Returns:
            Tensor: The log of the probability p(zs | zt).
        '''
        # Ensure sigma is positive to avoid numerical issues
        epsilon = 1e-6
        sigma = torch.max(sigma, torch.tensor(epsilon, device=sigma.device))

        delta = x.detach()
        
        log_p = -0.5 * ((delta - mu) ** 2) / (sigma ** 2) * node_mask
        # Sum over dimensions and normalize
        p_zs_zt = log_p.sum(dim=tuple(range(1, log_p.ndim))) / (node_mask.sum(dim=tuple(range(1, node_mask.ndim))) * 9)
        return p_zs_zt 

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context=None, fix_noise=False, prev_sample=None):
        """Samples from zs ~ p(zs | zt). """
    
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)
        # Neural net prediction 
        
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        self._assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        self._assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)
        
        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [self._remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
        
        # compute logp
        if prev_sample is not None:
            log_p = self.compute_log_p_zs_given_zt(prev_sample, mu, sigma, node_mask=node_mask)
        else:
            log_p = self.compute_log_p_zs_given_zt(zs, mu, sigma,  node_mask=node_mask)
            
        return zs, log_p, mu, sigma
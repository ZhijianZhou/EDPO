import argparse
import pickle
import os
import torch
import ray
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = "*****"
from verl_diffusion.trainer.config.base import BaseConfig
from verl_diffusion.trainer.ddpo_trainer import DDPOTrainer

from Model.EDM.configs.datasets_config import get_dataset_info
from Model.EDM.qm9 import dataset
from Model.EDM.qm9.models import get_model
from verl_diffusion.model.edm_model import EDMModel
from verl_diffusion.dataloader.dataloader import EDMDataLoader
from verl_diffusion.worker.rollout.edm_rollout import EDMRollout
from verl_diffusion.worker.reward.force import ForceReward
from verl_diffusion.worker.filter.filter import Filter
from verl_diffusion.worker.actor.edm_actor import EDMActor

def parse_args():
    parser = argparse.ArgumentParser(description="DDPO Training for EDM")
    parser.add_argument('--config_path', type=str, default="./verl_diffusion/trainer/config/ddpo_config.yaml", help="Path to the configuration file")
    parser.add_argument('--save_path', type=str, default="./saved_models/edm_ddpo", help="Path to save the model")
    parser.add_argument('--wandb', action='store_true', help="Enable wandb logging")
    parser.add_argument('--wandb_project', type=str, default="edm-ddp", help="Wandb project name")
    parser.add_argument('--wandb_name', type=str, default="edm-ddp-run", help="Wandb run name")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to checkpoint to resume from")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    import random
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    
    # Load configuration
    config = BaseConfig()
    config = config.from_yaml(args.config_path)
    config = config.to_dict()
    print("Config loaded")
    
    # Load EDM config
    edm_config_path = config["model"]["config"]
    with open(edm_config_path, 'rb') as f:
        edm_config = pickle.load(f)
    edm_config.datadir = "./Model/EDM/qm9/temp"
    print(edm_config)
    # Set default values if not present
    if not hasattr(edm_config, 'normalization_factor'):
        edm_config.normalization_factor = 1
    if not hasattr(edm_config, 'aggregation_method'):
        edm_config.aggregation_method = 'sum'

    # Set up device
    edm_config.cuda = not edm_config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if edm_config.cuda else "cpu")
    edm_config.device = device
    
    # Load dataset info and model
    dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(edm_config)
    flow, nodes_dist, prop_dist = get_model(edm_config, edm_config.device, dataset_info, dataloaders['train'])
    flow.to(device)
    
    # Initialize EDM model
    model = EDMModel(flow, edm_config)
    model.load(model_path=config["model"]["model_path"])
    
    # Initialize dataloader
    dataloader = EDMDataLoader(config=config,
                               dataset_info=dataset_info,
                               nodes_dist=nodes_dist,
                               prop_dist=prop_dist,
                               device=device,
                               condition=False,
                               num_batches=config["dataloader"]["epoches"])
    
    # Initialize rollout and rewarder in main function
    rollout = EDMRollout(model, config)
    rollout.model.to(device)
    rewarder = ForceReward(dataset_info)
    filters = Filter(dataset_info,config["dataloader"]["smiles_path"],False,False,False)
    actor = EDMActor(model, config)
    # Initialize Ray for parallel processing
    if not ray.is_initialized():
        ray.init()
    
    # Initialize DDPO Trainer with rollout and rewarder
    trainer = DDPOTrainer(
        config=config,
        model=model,
        dataset_info=dataset_info,
        device=device,
        dataloader=dataloader,
        rollout=rollout,
        rewarder=rewarder,
        filters=filters,
        actor = actor
    )
    
    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Clean up resources
        trainer.clean_up()
        print("Training completed")

if __name__ == "__main__":
    main() 
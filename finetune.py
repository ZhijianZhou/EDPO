from trainer.edpo import EDPOTrainer
from tqdm import tqdm as tq
import os
import wandb
import torch
import numpy
import random
import pprint
import argparse
def set_seed(seed):
    random.seed(seed)      
    torch.manual_seed(seed) 
    numpy.random.seed(seed) 
def parse_args():
    parser = argparse.ArgumentParser(description="EDPO Trainer Script")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the configuration file")
    parser.add_argument('--reward', type=str, required=True, help="Reward value to be used in training")
    parser.add_argument('--resume', type=bool, default=False, help="Whether to resume training from checkpoint")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # trainer init
    trainer = EDPOTrainer(config_path=args.config_path, reward=args.reward, resume=args.resume)
    # set seed
    set_seed(trainer.config.seed)
    pprint.pprint(trainer.config.to_dict())
    # wandb init
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_API_KEY"] = "*****"
    global_step = 0
    best_reward = -1000
    wandb.init(project="eddpo",name=trainer.config.name,config=trainer.config.to_dict())
    wandb.save('*.txt')
    root_path = os.path.join("exp",trainer.config.name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    for i in tq(range(0,trainer.config.train_epoch_num),desc = "Training epoch",unit = "epoch"):
        results = trainer.step(i,global_step)
        for result in results:
            if best_reward < result["Reward"]:
                best_reward = result["Reward"]
                trainer.save_model(root_path,"best")
            if i % trainer.config.save_freq == 0:
                trainer.save_model(root_path,str(i))
            for key,value in result.items():
                wandb.log({key: value}, commit=True)
            
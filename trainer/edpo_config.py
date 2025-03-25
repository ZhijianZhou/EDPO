import os
import argparse
import pickle
import torch
import pprint
class EDPOConfig:
    '''
        edm_model_path :  Edm model path
    '''
    def __init__(self,edm_model_path):
        # sample params ⬇
        self.sample_batch_size = 512
        self.sample_num_batches_per_epoch = 1
        self.train_num_inner_epochs = 1
        self.train_batch_size = 512
        self.save_freq = 1
        self.num_train_timesteps = 1000
        self.mini_batch = 128
        # optimizer paramas ⬇
        self.train_epoch_num = 100
        self.train_learning_rate = 1e-5
        self.train_adam_beta1 = 0.9
        self.train_adam_beta2 = 0.999
        self.train_adam_weight_decay = 1e-4
        self.train_adam_epsilon = 1e-8
        # loss paramas ⬇
        self.train_adv_clip_max = 1.0
        self.train_clip_range = 0.2
        # val parmas ⬇
        self.val_size = 5120
        # seed ⬇
        self.seed = 3407
        # name
        self.name = "stable-test"
        self.resume = False
        if self.resume:
            self.check_point_model_path = "./exp/RewardStableGeom/2_generative_model.npy"
            self.check_point_optimizer_path = "./exp/RewardForceQM/55_optimize.npy"
        with open(os.path.join(edm_model_path, 'args.pickle'), 'rb') as f:
            args = pickle.load(f)
        # be careful with this -->
        if not hasattr(args, 'normalization_factor'):
            args.normalization_factor = 1
        if not hasattr(args, 'aggregation_method'):
            args.aggregation_method = 'sum'
        if args.context_node_nf != 0:
            self.condition = args.conditioning[0]
        else:
            self.condition = None
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        args.device = device
        dtype = torch.float32
        
        # read edm config
        self.edm_config = args
        self.device = device

    def to_dict(self):
        # Convert the attributes of this class to a dictionary
        config_dict = {
            'sample_batch_size': self.sample_batch_size,
            'sample_num_batches_per_epoch': self.sample_num_batches_per_epoch,
            'train_num_inner_epochs': self.train_num_inner_epochs,
            'train_batch_size': self.train_batch_size,
            'save_freq': self.save_freq,
            'num_train_timesteps': self.num_train_timesteps,
            'train_learning_rate': self.train_learning_rate,
            'train_adam_beta1': self.train_adam_beta1,
            'train_adam_beta2': self.train_adam_beta2,
            'train_adam_weight_decay': self.train_adam_weight_decay,
            'train_adam_epsilon': self.train_adam_epsilon,
            'train_adv_clip_max': self.train_adv_clip_max,
            'train_clip_range': self.train_clip_range,
            'val_size': self.val_size,
            'seed': self.seed,
            'edm_config': vars(self.edm_config),  # Make sure edm_config is also a dictionary
            'device': str(self.device),  # Convert device to string for better readability
        }
        return config_dict

    

if __name__ == "__main__":
    config = EDPOConfig("../outputs/edm_qm9")
    print(config.edm_config)
    
    
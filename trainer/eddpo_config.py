import os
import argparse
import pickle
import torch
import pprint
class EDDPOConfig:
    '''
        edm_model_path :  Edm model path
    '''
    def __init__(self,edm_model_path):
        self.sample_batch_size = 10
        self.sample_num_batches_per_epoch = 2
        with open(os.path.join(edm_model_path, 'args.pickle'), 'rb') as f:
            args = pickle.load(f)
        # be careful with this -->
        if not hasattr(args, 'normalization_factor'):
            args.normalization_factor = 1
        if not hasattr(args, 'aggregation_method'):
            args.aggregation_method = 'sum'

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        args.device = device
        dtype = torch.float32
        
        # read edm config
        self.edm_config = args
        pprint.pprint(vars(args), indent=4)

    

if __name__ == "__main__":
    config = EDDPOConfig("../outputs/edm_qm9")
    print(config.edm_config)
    
    
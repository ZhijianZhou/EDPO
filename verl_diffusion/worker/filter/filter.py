from verl_diffusion.protocol import DataProto
from Model.EDM.qm9.rdkit_functions import build_molecule, mol2smiles
import torch
import pickle
import random
import numpy as np

class Filter:
    def __init__(self, dataset_info, file_name, condition, enable_filtering=True, enable_penalty=True):
        self.dataset_info = dataset_info
        with open(file_name, 'rb') as f:
            self.dataset_smiles_list = pickle.load(f)
        self.dataset_smiles = set(self.dataset_smiles_list)
        self.condition = condition
        self.enable_filtering = enable_filtering
        self.enable_penalty = enable_penalty
    def process_data(self, samples:DataProto) -> list:
        """
        Process the DataProto object to prepare it for force calculation.

        Args:
            samples (DataProto): A DataProto object containing the data to process.
            
        Returns:
            list: A list of processed molecule tuples (position, atom_type)
        """
        
        one_hot = samples.batch["categorical"]
        x = samples.batch['x']
        nodesxsample = samples.batch["nodesxsample"]
        node_mask = torch.zeros(x.shape[0], self.dataset_info['max_n_nodes'])
        
        for i in range(x.shape[0]):
            node_mask[i, 0:nodesxsample[i]] = 1
        n_samples = len(x)
        processed_list = []
        
        for i in range(n_samples):
            atom_type = one_hot[i].argmax(1).cpu().detach()
            pos = x[i].cpu().detach()
            atom_type = atom_type[0:int(nodesxsample[i])]
            pos = pos[0:int(nodesxsample[i])]
            if self.condition:
                processed_list.append((pos, atom_type, samples.batch["context"][i][0].cpu().detach()))
            else:
                processed_list.append((pos, atom_type))
                
        return processed_list
        
    def filter(self, data: DataProto) -> tuple[DataProto, float]:
        processed_list = self.process_data(data)
        all_smiles = []
        for graph in processed_list:
            mol = build_molecule(*graph, self.dataset_info)
            smiles = mol2smiles(mol)
            all_smiles.append(smiles)
         
        # Create a dictionary to store indices for each unique smiles
        smiles_indices = {}
        None_idx = []
        novelty_penalty = []
        
        for idx, smiles in enumerate(all_smiles):
            if smiles in self.dataset_smiles:
                novelty_penalty.append(-1)
            else:
                novelty_penalty.append(0)
            if smiles is not None:  # Only process non-None smiles
                if smiles not in smiles_indices:
                    smiles_indices[smiles] = []
                smiles_indices[smiles].append(idx)
            else:
                None_idx.append(idx)
        novelty_penalty_ratio = 1 + sum(novelty_penalty) / len(novelty_penalty)
        # Create a boolean mask, initially all False
        keep_mask = [False] * len(all_smiles)
        
        if self.enable_filtering:
            # For each unique smiles, randomly select one index to keep
            for indices in smiles_indices.values():
                if indices:  # If there are indices for this smiles
                    keep_idx = random.choice(indices)
                    keep_mask[keep_idx] = True
            for idx in None_idx:
                keep_mask[idx] = True
            indices_to_keep = np.where(keep_mask)[0]
        else:
            # If filtering is disabled, keep all indices
            indices_to_keep = np.arange(len(all_smiles))
            
        # Apply penalty if enabled
        if self.enable_penalty:
            novelty_penalty = torch.tensor(novelty_penalty).to(data.batch["rewards"].device)
            data.batch["rewards"] = data.batch["rewards"] + novelty_penalty * 0.1
            
        # filter 
        if self.enable_filtering:
            filtered_data_proto = DataProto.select_idxs(data, indices_to_keep)
        else:
            filtered_data_proto = data
        
        # Calculate filtering ratio
        total_samples = len(all_smiles)
        kept_samples = len(indices_to_keep)
        filtering_ratio = kept_samples / total_samples if total_samples > 0 else 0.0
        
        return filtered_data_proto, filtering_ratio, novelty_penalty_ratio
    
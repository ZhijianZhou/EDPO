from .base import BaseReward
import os
os.environ["OMP_NUM_THREADS"] = "18"
import numpy as np
import torch
from tqdm import tqdm as tq
import ray
import queue
import threading
import time

from xtb.ase.calculator import XTB
from ase import Atoms
from verl_diffusion.utils.math import kl_divergence_normal, rmsd
from Model.EDM.qm9.analyze import check_stability

from verl_diffusion.protocol import DataProto, TensorDict

@ray.remote(num_cpus=8)
def calcuate_xtb_force(mol, calc, dataset_info, atom_encoder):
    pos = mol[0].tolist()
    atom_type = mol[1].tolist()
    validity_results = check_stability(np.array(pos), atom_type, dataset_info)
    atom_type = [atom_encoder[atom] for atom in atom_type]
    atoms = Atoms(symbols=atom_type, positions=pos)
    atoms.calc = calc
    try:
        forces = atoms.get_forces()
        mean_abs_forces = rmsd(forces)
    except:
        mean_abs_forces = 5.0
    return -1 * mean_abs_forces, float(validity_results[0])



class ForceReward(BaseReward):
    def __init__(self, dataset_info:dict, condition:bool=False):
        super().__init__()
        self.calc = XTB(method="GFN2-xTB")
        self.dataset_info = dataset_info
        self.atom_encoder = self.dataset_info['atom_decoder']
        self.condition = condition
        self.input_queue = queue.Queue(maxsize=512)
        self.output_queue = queue.Queue()
        self.running = False
        self.thread = None
        
    def start_async(self):
        """Start asynchronous reward calculation thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(
            target=self._async_reward_loop,
            daemon=True
        )
        self.thread.start()
        
    def stop_async(self):
        """Stop the asynchronous reward calculation thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=10)
            self.thread = None
    
    def _async_reward_loop(self):
        """Continuously calculate rewards for samples in the input queue"""
        while self.running:
            try:
                # Try to get a sample from the input queue
                sample = self.input_queue.get(timeout=0.1)
                
                # Calculate rewards for this sample
                result = self.calculate_rewards(sample)
                
                # Put the results in the output queue
                self.output_queue.put((sample, result))
                
                # Mark task as done
                self.input_queue.task_done()
            except queue.Empty:
                # No samples available, sleep briefly
                time.sleep(0.01)
            except Exception as e:
                # Log any other exceptions
                print(f"Error in async reward calculation: {e}")
                
    def submit_batch(self, samples):
        """
        Submit a batch of samples for reward calculation
        
        Args:
            samples: List of samples to calculate rewards for
            
        Returns:
            List of reward results
        """
        results = []
        result = self.calculate_rewards(samples)
        results.append(result)
        return results
                
    def submit_sample(self, sample):
        """
        Submit a sample for asynchronous reward calculation
        
        Args:
            sample: The sample to calculate rewards for
        
        Returns:
            bool: True if the sample was submitted, False if the queue is full
        """
        try:
            self.input_queue.put_nowait(sample)
            return True
        except queue.Full:
            return False
            
    def get_next_result(self, timeout=None):
        """
        Get the next reward calculation result from the output queue
        
        Args:
            timeout: Maximum time to wait for a result
            
        Returns:
            Tuple of (sample, result) or None if timeout occurs
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
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
        
    def calculate_rewards(self, data: DataProto) -> DataProto:
        """
        Calculate the force reward for a given DataProto object.

        Args:
            data (DataProto): A DataProto object containing the data to calculate the reward for.
            
        Returns:
            DataProto: A DataProto object containing the original data and rewards
        """
        processed_list = self.process_data(data)
        rewards = []
        molecule_stable = []
        futures = [calcuate_xtb_force.remote(mol, self.calc, self.dataset_info, self.atom_encoder) for mol in processed_list]
        outputs_list = ray.get(futures)
        
        for i in outputs_list:
            rewards.append(i[0])
            molecule_stable.append(i[1])
            
        result = {}
        result['rewards'] = torch.tensor(rewards)
        result['stability'] = torch.tensor(molecule_stable)
        
        return DataProto(batch=TensorDict(result, batch_size=[len(rewards)]), meta_info=data.meta_info.copy())
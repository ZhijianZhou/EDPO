import os 
import torch
import time
from ase.io import read
import subprocess
import ray
from verl_diffusion.utils.math import rmsd
import numpy as np

@ray.remote(num_cpus=8)
def calcuate_dft_force(gjf_file):
    try:
        subprocess.run(["g16", gjf_file], check=True, text=True, capture_output=True)
        log_file = gjf_file.replace('.gjf', '.log')
        frames = read(log_file, index=':', format='gaussian-out')
        final_atoms = frames[-1]
        forces = final_atoms.get_forces()
        rmsd_forces = rmsd(np.array(forces))
    except:
        rmsd_forces = 5.0
    return rmsd_forces

def save_gjf_file_from_xyz(contents, path="./output"):
    os.makedirs(path, exist_ok=True)
    header = """%mem=8gb\n%nprocshared=8\n#P B3LYP/6-31G(2df,p) nosymm Force pop=Always \n\ntest\n\n0 1\n"""
    
    for batch_i in range(len(contents)):
        f = open(os.path.join(path, f"{batch_i}.gjf"),"w")
        f.write(header) 
        for i in contents[batch_i]:
            f.write(i)
            
        f.write("\n")
        f.close()
        
def delete_gjf_files(directory):
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith(".gjf") or filename.endswith(".log") and os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"wrong: {e}")
        
def save_gjf_file(one_hot, positions, atom_decoder, node_mask=None, path="./output"):
    os.makedirs(path, exist_ok=True)
    header = """%mem=8gb\n%nprocshared=8\n#P B3LYP/6-31G(2df,p) nosymm Force pop=Always \n\ntest\n\n0 1\n"""
    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        f = open(os.path.join(path, f"{batch_i}.gjf"),"w")
        f.write(header) 
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        n_atoms = int(atomsxmol[batch_i])
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = atom_decoder[atom]
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.write("\n")
        f.close()

def read_force(batch_size, path="./output"):
    all_force = []
    for i in range(0,batch_size):
        file_name = os.path.join(path, f"{i}.log")
        try:
            frames = read(file_name, index=':', format='gaussian-out')
            final_atoms = frames[-1]
            forces = final_atoms.get_forces()
            rmsd_forces = rmsd(np.array(forces))
            all_force.append(-1.0 * rmsd_forces)
        except:
            all_force.append(-5.0)
    return all_force

def qm_reward_model(one_hot, x, atom_decoder, node_mask, host_ip, batch_size, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    delete_gjf_files(output_dir)
    save_gjf_file(one_hot, x, atom_decoder, node_mask=node_mask, path=output_dir)
    time.sleep(10)
    for i in range(batch_size):
        gif_file = os.path.join(output_dir, f"{i}.gjf")
        subprocess.run(["g16", gif_file], check=True, text=True, capture_output=True)
    time.sleep(5)
    force = read_force(batch_size, path=output_dir)
    return force

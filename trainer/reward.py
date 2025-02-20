import requests
import os 
from qm9.visualizer import save_gjf_file
import os
from ase.io import read
import numpy as np
import time
import re
from tqdm import tqdm
from ase import Atoms
from ase.io import read
from xtb.ase.calculator import XTB
from qm9.analyze import check_stability_simple

import numpy as np

def rmsd(A, B=None):
    """
    Calculate the RMSD (Root Mean Square Deviation) between two 2D matrices A and B.
    If B is not provided, it defaults to a zero matrix.

    Parameters:
    A: numpy.ndarray, shape (m, n)
    B: numpy.ndarray, shape (m, n), default is None. If None, B will be set to a zero matrix.
    
    Returns:
    float: RMSD value
    """
    # If B is None, set B to a zero matrix with the same shape as A
    if B is None:
        B = np.zeros_like(A)

    # Ensure matrices A and B have the same shape
    if A.shape != B.shape:
        raise ValueError("The input matrices A and B must have the same shape")
    
    # Calculate the squared differences between matrices A and B
    diff = A - B
    squared_diff = np.square(diff)
    
    # Compute the root mean square deviation (RMSD)
    rmsd_value = np.sqrt(np.mean(squared_diff))
    
    return rmsd_value

def delete_gjf_files(directory):
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith(".gjf") or filename.endswith(".log") and os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"wrong: {e}")
        
def send_request(host_ip, batch_size):
    url =f"http://{host_ip}:8098/start_do"  
    data = {
        "batch_size": batch_size  
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            print("Request successful.")
            print("Response:", response.json())
        else:
            print(f"Request failed with status code: {response.status_code}")
            print("Response:", response.text)
    except Exception as e:
        print("An error occurred while sending the request:", e)


def process_eigenvalues(file_path):
    # Lists to store data
    occ_values = []
    virt_values = []
    
    # Open the file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Look for lines containing "occ. eigenvalues" and extract the values
            if "occ. eigenvalues" in line:
                occ_values.extend(map(float, re.findall(r"-?\d+\.\d+", line)))
            # Look for lines containing "Alpha virt. eigenvalues" and extract the values
            elif "Alpha virt. eigenvalues" in line:
                virt_values.extend(map(float, re.findall(r"-?\d+\.\d+", line)))
    
    # Find the largest occ. eigenvalue and the smallest Alpha virt. eigenvalue
    max_occ_value = max(occ_values) if occ_values else None
    min_virt_value = min(virt_values) if virt_values else None
    
    # Return the difference between the smallest virt. eigenvalue and the largest occ. eigenvalue, converted to eV
    return (min_virt_value - max_occ_value) * 27.211396127707

def read_gap(batch_size):
    all_gap = []
    for i in range(0,batch_size):
        file_name = os.path.join("./Temp", f"{i}.log")
        try:
            gap = process_eigenvalues(file_name)
            all_gap.append(gap)
        except:
            all_gap.append(0)
    return all_gap
        
def read_force(batch_size):
    all_force = []
    for i in range(0,batch_size):
        file_name = os.path.join("./Temp", f"{i}.log")
        try:
            frames = read(file_name, index=':', format='gaussian-out')
            final_atoms = frames[-1]
            forces = final_atoms.get_forces()
            rmsd_forces = rmsd(np.array(forces))
            all_force.append(-1.0 * rmsd_forces)
        except:
            all_force.append(-5.0)
    return all_force

def qm_reward_model(one_hot,x,atom_decoder,node_mask,host_ip,batch_size):
    delete_gjf_files("./Temp")
    save_gjf_file(one_hot, x, atom_decoder, node_mask = node_mask)
    time.sleep(10)
    send_request(host_ip,batch_size)
    time.sleep(5)
    force = read_force(batch_size)
    return force

def save_gjf_file_from_xyz(contents,path="./Temp"):
    header = """%mem=8gb\n%nprocshared=8\n#P B3LYP/6-31G(2df,p) nosymm Force pop=Always \n\ntest\n\n0 1\n"""

    for batch_i in range(len(contents)):
        f = open(os.path.join(path, f"{batch_i}.gjf"),"w")
        f.write(header) 
        for i in contents[batch_i]:
            f.write(i)
            
        f.write("\n")
        f.close()
    
def qm_reward_model_xyz(contents,host_ip,batch_size):
    delete_gjf_files("./Temp")
    save_gjf_file_from_xyz(contents)
    time.sleep(10)
    send_request(host_ip,batch_size)
    force = read_force(batch_size)
    return force
def force_reward_xtb_xyz(files):
    
    calc = XTB(method="GFN2-xTB")
    forces = []
    for file in tqdm(files):
        atoms = read(file)
        atoms.calc = calc
        try:
            force = atoms.get_forces()
            mean_abs_forces = rmsd(force)
        except:
            mean_abs_forces = -5
        forces.append(mean_abs_forces)
    return np.array(forces)
def stable_reward_xyz(files):
    molecule_stable= []
    for file in tqdm(files):
        atoms = read(file)
        atom_types = atoms.get_chemical_symbols()  
        atom_positions = atoms.get_positions()  
        validity_results = check_stability_simple(np.array(atom_positions), atom_types)
        molecule_stable.append(int(validity_results[0]))
    return sum(molecule_stable)/len(molecule_stable)

def get_qm_gap(one_hot,x,atom_decoder,node_mask,host_ip,batch_size):
    delete_gjf_files("./Temp")
    save_gjf_file(one_hot, x, atom_decoder, node_mask = node_mask)
    time.sleep(10)
    send_request(host_ip,batch_size)
    time.sleep(5)
    gap = read_gap(batch_size)
    return gap

def get_qm_gap_xyz(contents,host_ip,batch_size):
    delete_gjf_files("./Temp")
    save_gjf_file_from_xyz(contents)
    time.sleep(10)
    send_request(host_ip,batch_size)
    time.sleep(5)
    gap = read_gap(batch_size)
    return gap
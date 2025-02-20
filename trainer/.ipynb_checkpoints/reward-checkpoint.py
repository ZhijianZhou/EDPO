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

def rmsd(A, B=None):
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
def delete_gjf_files(directory):
    try:
        # 遍历目录中的所有文件
        for filename in os.listdir(directory):
            # 构造文件的完整路径
            file_path = os.path.join(directory, filename)
            # 检查是否是 .gjf 文件并且是文件（不是目录）
            if filename.endswith(".gjf") or filename.endswith(".log") and os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"发生错误: {e}")
        
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
    # 用于存储数据
    occ_values = []
    virt_values = []
    
    # 打开文件进行读取
    with open(file_path, 'r') as file:
        for line in file:
            # 查找含有 "occ. eigenvalues" 的行，并提取其数值
            if "occ. eigenvalues" in line:
                occ_values.extend(map(float, re.findall(r"-?\d+\.\d+", line)))
            # 查找含有 "Alpha virt. eigenvalues" 的行，并提取其数值
            elif "Alpha virt. eigenvalues" in line:
                virt_values.extend(map(float, re.findall(r"-?\d+\.\d+", line)))
    
    # 找到最大的 occ. eigenvalue 和最小的 Alpha virt. eigenvalue
    max_occ_value = max(occ_values) if occ_values else None
    min_virt_value = min(virt_values) if virt_values else None
    return (min_virt_value - max_occ_value)* 27.211396127707

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
def stable_reward_xtb_xyz(files):
    stables = []
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
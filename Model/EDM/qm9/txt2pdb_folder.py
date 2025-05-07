import os
import glob

# 输入文件夹和输出文件名
input_folder = "outputs/edm_qm9/epoch_1960_0/chain"  # 输入文件夹
output_filename = "output.pdb"  # 输出文件名

# 打开输出PDB文件准备写入数据
with open(output_filename, "w") as outfile:
    # 写入PDB头部信息
    outfile.write("HEADER    Sample molecule\n")
    
    atom_index = 1  # 原子编号
    molecule_index = 1  # 分子编号
    
    # 获取文件夹下所有的 .txt 文件
    txt_files =  sorted(glob.glob(os.path.join(input_folder, "*.txt")))
    print(txt_files)
    # 遍历所有 .txt 文件，每个文件代表一个分子
    for txt_file in txt_files:
        # 为每个分子添加 MODEL 和 ENDMDL 来区分
        outfile.write(f"MODEL     {molecule_index:4d}\n")
        
        # 打开并读取每个输入txt文件
        with open(txt_file, "r") as infile:
            # 读取第一行，获取原子数量
            num_atoms = int(infile.readline().strip())
            infile.readline().strip()  # 跳过第二行
            lines = infile.readlines()

            # 确保文件中的原子数据行数与第一行数字一致
            assert len(lines) == num_atoms, f"文件中的原子数量 ({len(lines)}) 与文件头的数字 ({num_atoms}) 不匹配！"
            
            # 处理每一行原子数据
            for line in lines:
                parts = line.split()
                
                # 确保每行包含正确数量的数据：元素、x, y, z 坐标
                if len(parts) == 4:
                    element = parts[0]
                    x, y, z = map(float, parts[1:])
                    
                    # 写入PDB格式的ATOM行
                    outfile.write(f"ATOM  {atom_index:5d}  {element:2s} MOL     1    {x:8.3f} {y:8.3f} {z:8.3f}  1.00  0.00      1\n")
                    atom_index += 1
        
        # 为每个分子添加 ENDMDL
        outfile.write(f"ENDMDL   {molecule_index:4d}\n")
        
        # 增加分子编号
        molecule_index += 1
    
    # 写入PDB尾部信息
    outfile.write("END\n")

print(f"PDB file has been generated as '{output_filename}'.")

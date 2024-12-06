# 读取输入文件并写入PDB格式文件
input_filename = "outputs/edm_qm9/eval/chain_80/chain_109.txt"  # 输入文件名
output_filename = "output.pdb"  # 输出文件名

# 打开输入文件读取数据
with open(input_filename, "r") as infile:
    # 读取第一行，获取原子数量
    num_atoms = int(infile.readline().strip())
    infile.readline().strip()
    # 读取原子数据
    lines = infile.readlines()

# 确保文件中的原子数据行数与第一行数字一致
assert len(lines) == num_atoms, f"文件中的原子数量 ({len(lines)}) 与文件头的数字 ({num_atoms}) 不匹配！"

# 打开输出PDB文件写入数据
with open(output_filename, "w") as outfile:
    # 写入PDB头部信息
    outfile.write("HEADER    Sample molecule\n")
    
    atom_index = 1  # 原子编号
    for line in lines:
        parts = line.split()
        
        # 确保每行包含正确数量的数据：元素、x, y, z 坐标
        if len(parts) == 4:
            element = parts[0]
            x, y, z = map(float, parts[1:])
            
            # 写入PDB格式的ATOM行
            outfile.write(f"ATOM  {atom_index:5d}  {element:2s} MOL     1    {x:8.3f} {y:8.3f} {z:8.3f}  1.00  0.00      1\n")
            atom_index += 1
    
    # 写入PDB尾部信息
    outfile.write("END\n")

print(f"PDB file has been generated as '{output_filename}'.")
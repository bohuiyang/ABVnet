import os

# 定义需要处理的文件路径
annotation_dir = "/home/yangbohui/MMA/annotation"
output_dir = "/home/yangbohui/MMA/annotation_new"  # 使用临时路径
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

# 定义文件列表（set1-set5 测试集和训练集）
files_to_process = [
                       f"MAFW_set_{i}_test_faces.txt" for i in range(1, 6)
                   ] + [
                       f"MAFW_set_{i}_train_faces.txt" for i in range(1, 6)
                   ]

for filename in files_to_process:
    input_file = os.path.join(annotation_dir, filename)
    output_file = os.path.join(output_dir, filename)

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        continue

    # 读取文件并修改内容
    modified_lines = []
    with open(input_file, "r") as f:
        for line in f:
            parts = line.strip().split()  # 拆分每一行
            if len(parts) >= 3:
                # 将第二项加 1
                parts[1] = str(int(parts[1]) + 1)
                modified_lines.append(" ".join(parts) + "\n")

    # 写入新的文件
    with open(output_file, "w") as f:
        f.writelines(modified_lines)

    print(f"Processed: {input_file} -> {output_file}")

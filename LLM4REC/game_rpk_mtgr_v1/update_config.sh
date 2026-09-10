#!/bin/bash
# 使用环境变量 input_dir（源文件）和 output_dir（目标文件）进行文件复制

# 检查环境变量是否已设置
if [ -z "$input_dir" ]; then
    echo "错误: 环境变量 input_dir 未设置" >&2
    exit 1
fi

if [ -z "$output_dir" ]; then
    echo "错误: 环境变量 output_dir 未设置" >&2
    exit 1
fi

# 检查源文件是否存在且为普通文件
if [ ! -f "$input_dir" ]; then
    echo "错误: 源文件 '$input_dir' 不存在或不是普通文件" >&2
    exit 1
fi

# 确保目标文件的父目录存在（如果目标路径包含目录）
target_dir=$(dirname "$output_dir")
if [ ! -d "$target_dir" ]; then
    mkdir -p "$target_dir" || { echo "无法创建目标目录 '$target_dir'" >&2; exit 1; }
fi

# 执行文件复制（覆盖目标文件，保留源文件属性）
cp -p "$input_dir" "$output_dir"

# 检查复制是否成功
if [ $? -eq 0 ]; then
    echo "文件复制完成: $input_dir -> $output_dir"
else
    echo "复制过程中出现错误" >&2
    exit 1
fi

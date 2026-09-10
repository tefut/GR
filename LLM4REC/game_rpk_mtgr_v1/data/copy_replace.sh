#!/bin/bash

if [ "$period" == "" ]; then
    # 获取最近数据日期
    date=`ls ${data_dir} | grep '^[0-9]\{8\}$' | awk 'BEGIN {max=0} {if($1>max) max=$1} END {print max}'`
    # 必选参数，用以区分测试日期
    period=$(printf "%s-000000" ${date})
else
    date=${period%-*}
fi
echo "period: ${period}"


# 检查源目录是否存在
if [ ! -d "$input_dir" ]; then
    echo "错误：源目录 $input_dir 不存在！"
    exit 1
fi

# 检查源文件是否存在
source_file="${input_dir}/${file_name}"
if [ ! -f "$source_file" ]; then
    echo "错误：源文件 $source_file 不存在！"
    exit 1
fi


output_dir=${output_dir}/${date}/${out_opt}
# 检查目标目录是否存在，不存在则创建
if [ ! -d "$output_dir" ]; then
    echo "目标目录 $output_dir 不存在，正在自动创建..."
    mkdir -p "$output_dir"
    if [ $? -ne 0 ]; then
        echo "错误：无法创建目标目录 $output_dir"
        exit 1
    fi
fi

# 目标文件完整路径
target_file="${output_dir}/${file_name}"

# 执行复制（存在则替换，不存在则新建）
echo "正在处理：$source_file -> $target_file"
cp -f "$source_file" "$target_file"

# 检查是否成功
if [ $? -eq 0 ]; then
    echo "操作成功！文件已复制/替换完成。"
else
    echo "操作失败！"
    exit 1
fi

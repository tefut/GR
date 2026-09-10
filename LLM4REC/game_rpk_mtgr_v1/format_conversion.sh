#!/bin/bash
# SID Format Conversion on MTP production environment

set -euo pipefail
echo -e "\n--> SID Format Conversion on MTP production environment\n"
echo -e "Period: ${period}"

cur_dir=$(cd "$(dirname "$0")" && pwd)
echo "cur_dir: ${cur_dir}"
parent_dir=$(cd "${cur_dir}/.." && pwd)
export PYTHONPATH="${parent_dir}:${PYTHONPATH}"

if [ -z "$period" ]; then
    # 获取最近数据日期
    date=$(ls "${data_dir}" | grep '^[0-9]\{8\}$' | awk 'BEGIN {max=0} {if($1>max) max=$1} END {print max}')
    # 必选参数，用以区分测试日期
    period=$(printf "%s-000000" "${date}")
else
    date=${period%-*}
fi
echo "period: ${period}"

: "${data_path_prefix:=}"
data_path="${data_path_prefix}${date}"
echo "data_path: ${data_dir}${data_path}"

: "${mergeFeatureMaxIndexName:=feature_map.json}"
feature_map_path="${data_dir}${data_path}/config/${mergeFeatureMaxIndexName}"
echo "The date_run is: ${date}."
echo "The feature_map_path is: ${feature_map_path}."
mkdir -p "${data_dir}${data_path}"

python "${cur_dir}/utils/format_conversion.py" \
    --sid_path="${sid_path}" \
    --feature_path="${feature_map_path}" \
    --output_file="${output_file}"

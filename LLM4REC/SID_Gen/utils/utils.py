import datetime
import json
import os
import re
from zoneinfo import ZoneInfo

# 预编译正则，大幅提升批量处理性能
_RE_CLEAN_STRICT = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]')
_RE_CLEAN_PUNCT = re.compile(
    r'[^\u4e00-\u9fa5'  # 汉字
    r'a-zA-Z0-9'  # 英文与数字
    r'\s'  # 空白符（空格/换行/制表）
    r'，。！？、；：“”‘’（）【】《》'  # 中文标点
    r'.,!?;:()\-\'\"/@#%&*+=]'  # 英文标点及常用符号
)
_RE_COLLAPSE_SPACE = re.compile(r'\s+')


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def set_color(log, color, highlight=True):
    # 定义颜色映射表
    colors = {
        "black": "0", "red": "1", "green": "2", "yellow": "3",
        "blue": "4", "pink": "5", "cyan": "6", "white": "7"
    }

    # 使用 .get() 方法，如果找不到颜色则默认为 "7" (白色)
    color_code = colors.get(color, "7")

    # 确定高亮模式
    mode = "1" if highlight else "0"

    # 拼接 ANSI 转义码
    # \033[{mode};3{code}m
    return f"\033[{mode};3{color_code}m{log}\033[0m"


def get_local_time(timezone_str="Asia/Shanghai"):
    r"""获取指定时区的当前时间

    Args:
        timezone_str (str): 时区名称，例如 "Asia/Shanghai" 或 "UTC"

    Returns:
        str: 格式化后的当前时间
    """
    # 获取指定时区的当前时间
    cur = datetime.datetime.now(ZoneInfo(timezone_str))
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


"""
All-in-One Agent 工具函数
"""
import sys
import yaml
from pathlib import Path

# 添加 models 目录到 Python 路径
from models.qf import get_llm as get_qianfan_llm
from models.gpt import get_llm as get_openai_llm
from models.gemini import get_llm as get_gemini_llm


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG = load_config()


def get_llm(model_name: str = None, temperature: float = None):
    """
    根据 model_name 动态获取对应的 LLM 实例

    Args:
        model_name: 模型名称，如果为 None 则使用配置中的默认模型
        temperature: 温度参数，如果为 None 则使用默认值

    Returns:
        LLM 实例
    """
    if model_name is None:
        model_name = CONFIG['llm']['model']

    if temperature is None:
        temperature = CONFIG['llm'].get('temperature', 0)

    if model_name.startswith('gpt'):
        return get_openai_llm(model_name, temperature=temperature)
    elif model_name.startswith('gemini'):
        return get_gemini_llm(model_name, temperature=temperature)
    else:
        return get_qianfan_llm(model_name, temperature=temperature)

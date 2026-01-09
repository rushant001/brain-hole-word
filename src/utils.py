import yaml
import os
from pathlib import Path
from models.qf import get_llm as get_qianfan_llm
from models.gpt import get_llm as get_openai_llm
from models.gemini import get_llm as get_gemini_llm

def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def ensure_output_dir():
    """确保输出目录存在"""
    output_dir = Path(CONFIG['workflow']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "cards").mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "video").mkdir(exist_ok=True)
    return output_dir

def get_output_path(subdir: str, filename: str) -> Path:
    """获取输出文件路径"""
    output_dir = ensure_output_dir()
    return output_dir / subdir / filename

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

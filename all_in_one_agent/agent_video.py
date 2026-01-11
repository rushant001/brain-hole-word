"""
Brain-Hole-Word Video Agent
使用 AI 图生视频能力，生成动态脑洞单词短视频
"""

import os
import json
import time
import base64
import logging
import httpx
import yaml
from pathlib import Path
from typing import Optional

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from openai import OpenAI

from utils import get_llm

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)


# === 加载配置 ===
def load_config():
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_prompts():
    prompts_path = Path(__file__).parent / 'prompts_video.yaml'
    with open(prompts_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


CONFIG = load_config()
PROMPTS = load_prompts()


def get_output_path(subdir: str, filename: str) -> Path:
    output_dir = Path(__file__).parent / CONFIG.get('output_dir', './output')
    (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir / subdir / filename


# === Tool 1: 创意规划（图片Prompt + 视频Prompt） ===
@tool
def brainstorm_video_content(word: str) -> str:
    """
    为英语单词创造脑洞记忆法，同时生成图片Prompt和视频Prompt。

    Args:
        word: 英语单词（如 "Ambulance"）

    Returns:
        JSON格式的创意内容，包含：
        - meaning: 中文含义
        - memory_method: 记忆方法（谐音/画面）
        - slogan: 记忆口号
        - image_prompt: 英文图片生成Prompt（竖屏，Pixar风格）
        - video_prompt: 中文视频动作Prompt（含角色台词，遵循蒸汽机公式）
        - duration: 推荐视频时长（5或10秒）
    """
    start_time = time.time()
    logger.info(f"[Tool] brainstorm_video_content 开始执行，单词: {word}")

    try:
        parser = JsonOutputParser()
        prompts = PROMPTS['tools']['brainstorm_video_content']

        llm = get_llm(temperature=0.8)

        messages = [
            SystemMessage(content=prompts['system']),
            HumanMessage(content=prompts['user'].format(word=word))
        ]

        response = llm.invoke(messages)
        content = response.text

        # 解析 JSON
        try:
            data = parser.parse(content)
            elapsed = time.time() - start_time
            logger.info(f"[Tool] brainstorm_video_content 完成，耗时: {elapsed:.2f}秒")
            return json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as e:
            logger.error(f"[Tool] brainstorm_video_content JSON解析失败: {e}")
            return f"Error: JSON解析失败 - {e}. 原始内容: {content[:500]}"
    except Exception as e:
        logger.error(f"[Tool] brainstorm_video_content 执行失败: {e}")
        raise


# === Tool 2: 生成竖屏图片 ===
@tool
def generate_vertical_image(image_prompt: str, word: str) -> str:
    """
    生成竖屏图片用于视频生成。

    Args:
        image_prompt: 英文图片生成Prompt
        word: 单词（用于文件命名）

    Returns:
        生成的图片本地路径
    """
    start_time = time.time()
    logger.info(f"[Tool] generate_vertical_image 开始执行，单词: {word}")

    try:
        video_config = CONFIG.get('video_generation', {})
        provider = video_config.get('image_provider', 'qwen')
        
        # 获取图片生成配置
        img_config = CONFIG['image_generation'].get(provider, {})
        
        client = OpenAI(
            base_url=img_config.get('base_url', 'https://qianfan.baidubce.com/v2'),
            api_key=os.getenv("QIANFAN_API_KEY")
        )
        
        # 使用竖屏尺寸
        size = video_config.get('image_size', '720x1280')
        
        response = client.images.generate(
            model=img_config.get('model', 'qwen-image'),
            prompt=image_prompt,
            size=size,
            n=1
        )
        image_url = response.data[0].url

        # 下载图片
        output_path = get_output_path("images", f"{word}_video.png")
        resp = httpx.get(image_url, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(resp.content)

        elapsed = time.time() - start_time
        logger.info(f"[Tool] generate_vertical_image 完成，耗时: {elapsed:.2f}秒，路径: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"[Tool] generate_vertical_image 执行失败: {e}")
        raise


# === Tool 3: 创建AI视频生成任务 ===
@tool
def create_video_task(image_path: str, video_prompt: str) -> str:
    """
    调用百度蒸汽机API创建AI视频生成任务。

    Args:
        image_path: 本地图片路径
        video_prompt: 视频动作描述Prompt（含角色台词）

    Returns:
        JSON格式的任务信息，包含task_id
    """
    start_time = time.time()
    logger.info(f"[Tool] create_video_task 开始执行")
    duration = 5
    try:
        # 读取图片并转为base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 获取图片格式
        if image_path.lower().endswith('.png'):
            image_prefix = "data:image/png;base64,"
        else:
            image_prefix = "data:image/jpeg;base64,"
        
        # 构建请求
        api_key = os.getenv("QIANFAN_API_KEY")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        video_config = CONFIG.get('video_generation', {})
        model = video_config.get('model', 'musesteamer-2.0-turbo-i2v-audio')
        
        payload = {
            "model": model,
            "content": [
                {
                    "type": "text",
                    "text": video_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_prefix + image_base64
                    }
                }
            ],
            "duration": duration
        }
        
        # 发送请求
        resp = httpx.post(
            "https://qianfan.baidubce.com/video/generations",
            headers=headers,
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        result = resp.json()
        
        elapsed = time.time() - start_time
        logger.info(f"[Tool] create_video_task 完成，耗时: {elapsed:.2f}秒，task_id: {result.get('task_id')}")
        logger.info(result)
        
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[Tool] create_video_task 执行失败: {e}")
        raise


# === Tool 4: 查询视频任务状态并下载 ===
@tool
def poll_and_download_video(task_id: str, word: str, max_wait: int = 300) -> str:
    """
    轮询视频生成任务状态，完成后下载视频。

    Args:
        task_id: 视频生成任务ID
        word: 单词（用于文件命名）
        max_wait: 最大等待时间（秒），默认300秒

    Returns:
        成功时返回视频本地路径，失败时返回错误信息
    """
    start_time = time.time()
    logger.info(f"[Tool] poll_and_download_video 开始执行，task_id: {task_id}")

    api_key = os.getenv("QIANFAN_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    poll_interval = 30  # 每30秒查询一次
    elapsed = 0

    while elapsed < max_wait:
        try:
            resp = httpx.get(
                f"https://qianfan.baidubce.com/video/generations?task_id={task_id}",
                headers=headers,
                timeout=30
            )
            resp.raise_for_status()
            result = resp.json()

            status = result.get('status', '')
            logger.info(f"[Tool] poll_and_download_video 状态: {status}，已等待: {elapsed}秒")

            if status == 'succeeded':
                # 下载视频
                video_url = result.get('content', {}).get('video_url', '')
                if not video_url:
                    return "Error: 视频URL为空"

                output_path = get_output_path("video", f"{word}_ai.mp4")
                video_resp = httpx.get(video_url, follow_redirects=True, timeout=120)
                video_resp.raise_for_status()
                with open(output_path, 'wb') as f:
                    f.write(video_resp.content)

                total_time = time.time() - start_time
                logger.info(f"[Tool] poll_and_download_video 完成，总耗时: {total_time:.2f}秒")
                return str(output_path)

            elif status == 'failed':
                error_msg = result.get('error', {}).get('message', '未知错误')
                logger.error(f"[Tool] poll_and_download_video 失败: {error_msg}")
                return f"Error: 视频生成失败 - {error_msg}"

            # 继续等待
            time.sleep(poll_interval)
            elapsed = time.time() - start_time

        except Exception as e:
            logger.error(f"[Tool] poll_and_download_video 查询出错: {e}")
            time.sleep(poll_interval)
            elapsed = time.time() - start_time

    return f"Error: 超时，等待{max_wait}秒后任务仍未完成"


# === 导出所有 Tools ===
ALL_VIDEO_TOOLS = [
    brainstorm_video_content,
    generate_vertical_image,
    create_video_task,
    poll_and_download_video
]


# === Agent 创建 ===
def create_video_agent(word: str):
    """
    创建脑洞单词视频 Agent

    Args:
        word: 要处理的单词

    Returns:
        配置好的 Agent
    """
    agent_prompt = PROMPTS['agent']['system_prompt'].format(word=word)

    llm = get_llm()

    # 使用 LangChain 1.2.0 的 create_agent API
    agent = create_agent(
        model=llm,
        tools=ALL_VIDEO_TOOLS,
        system_prompt=agent_prompt
    )

    return agent


def run_video_agent(word: str) -> dict:
    """
    运行 Video Agent 生成视频

    Args:
        word: 单词

    Returns:
        包含结果的字典
    """
    agent = create_video_agent(word)

    # LangChain 1.2.0 使用 messages 格式
    config = RunnableConfig(recursion_limit=30)
    result = agent.invoke({
        "messages": [HumanMessage(content=f"请为单词 '{word}' 生成一个脑洞记忆短视频。按顺序执行所有步骤。")]
    }, config=config)

    return result


def main():
    word = 'Kindergarten'
    print(f"\n🎬 启动脑洞单词视频 Agent...")
    print(f"📝 单词: {word}")
    print("=" * 60)

    result = run_video_agent(word)

    print("\n" + "=" * 60)
    print("✨ Agent 执行完成!")
    # 获取最后一条消息作为结果
    messages = result.get('messages', [])
    if messages:
        print(f"📹 结果: {result.get('messages')[-1].text if len(result.get('messages')) > 0 else None}")
    else:
        print(f"📹 结果: {result}")


if __name__ == "__main__":
    main()
    # ret = create_video_task.invoke({'video_prompt': '一个穿着蓝色背带裤的Q版小男孩坐在五彩斑斓的小板凳上，无聊地晃动着双腿，'
    #                                           '眼神期待地看着墙上的大时钟。镜头缓慢平稳地向小男孩面部推进。小男孩用稚嫩的声音说：'
    #                                           '"Kindergarten！读作：勤的-干-等！勤快的小朋友在幼儿园干等着放学呢！'
    #                                           '快跟我读：Kindergarten，幼儿园！"',
    #                           'image_path': '/Users/xuhong02/D/private_code/brain-hole-word/'
    #                                         'all_in_one_agent/output/images/Kindergarten_video.png'})
    # print(ret)

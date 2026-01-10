"""
Brain-Hole-Word Tools
将各功能封装为 LangChain Tool，供 ReAct Agent 调用
"""
import os
import json
import asyncio
import httpx
import yaml
import time
import logging
from pathlib import Path
from typing import Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont

from utils import get_llm as get_model_llm

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 屏蔽 httpx 的 INFO 日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)


# === 加载配置 ===

def load_config():
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_prompts():
    prompts_path = Path(__file__).parent / 'prompts.yaml'
    with open(prompts_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
PROMPTS = load_prompts()

def get_output_path(subdir: str, filename: str) -> Path:
    output_dir = Path(CONFIG['output_dir'])
    (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir / subdir / filename


# === Tool 1: 创意策划 ===

@tool
def creative_brainstorm(word: str) -> str:
    """
    为英语单词创造脑洞记忆法。音标由 LLM 自动生成。

    Args:
        word: 英语单词（如 "Ambulance"）

    Returns:
        JSON格式的创意内容，包含：
        - phonetic: 音标（自动生成）
        - strategy: 记忆策略（谐音/画面）
        - slogan: 记忆口号
        - story_scene: 场景描述
        - example_en/example_cn: 例句
        - segments: 分段脚本（用于TTS）
    """
    start_time = time.time()
    func_name = "creative_brainstorm"
    logger.info(f"[Tool] {func_name} 开始执行，开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    try:
        parser = JsonOutputParser()
        prompts = PROMPTS['tools']['creative_brainstorm']

        llm = get_model_llm(temperature=CONFIG['llm']['temperature'])

        messages = [
            SystemMessage(content=prompts['system']),
            HumanMessage(content=prompts['user'].format(word=word))
        ]

        response = llm.invoke(messages)
        content = response.text

        # 验证并返回 JSON
        try:
            data = parser.parse(content)
            # 确保必要字段存在
            required_fields = ['phonetic', 'slogan', 'story_scene', 'example_en', 'example_cn', 'segments']
            for field in required_fields:
                if field not in data:
                    data[field] = "" if field != 'segments' else []
            end_time = time.time()
            elapsed = end_time - start_time
            logger.info(f"[Tool] {func_name} 执行完成，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒")
            return json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as e:
            end_time = time.time()
            elapsed = end_time - start_time
            logger.error(f"[Tool] {func_name} 执行失败，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒，错误: {e}")
            return f"Error: JSON解析失败 - {e}. 原始内容: {content[:200]}"
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.error(f"[Tool] {func_name} 执行失败，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒，错误: {e}")
        raise


# === Tool 2: 生成绘图 Prompt ===

@tool
def generate_visual_prompt(word: str, slogan: str, story_scene: str) -> str:
    """
    将记忆场景转化为AI绘图Prompt。

    Args:
        word: 英语单词
        slogan: 记忆口号
        story_scene: 场景描述

    Returns:
        英文绘图Prompt
    """
    start_time = time.time()
    func_name = "generate_visual_prompt"
    logger.info(f"[Tool] {func_name} 开始执行，开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    try:
        prompts = PROMPTS['tools']['generate_visual_prompt']

        llm = get_model_llm(temperature=0.7)

        messages = [
            SystemMessage(content=prompts['system']),
            HumanMessage(content=prompts['user'].format(
                word=word, slogan=slogan, story_scene=story_scene
            ))
        ]

        response = llm.invoke(messages)
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[Tool] {func_name} 执行完成，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒")
        return response.text
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.error(f"[Tool] {func_name} 执行失败，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒，错误: {e}")
        raise


# === Tool 3: 生成图片 ===

@tool
def generate_image(visual_prompt: str, word: str) -> str:
    """
    调用AI生图API生成图片。

    Args:
        visual_prompt: 英文绘图Prompt
        word: 单词（用于文件命名）

    Returns:
        生成的图片本地路径
    """
    start_time = time.time()
    func_name = "generate_image"
    logger.info(f"[Tool] {func_name} 开始执行，开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    try:
        provider = CONFIG['image_generation']['provider']

        if provider == 'dall-e-3':
            config = CONFIG['image_generation']['dalle']
            client = OpenAI()
            response = client.images.generate(
                model="dall-e-3",
                prompt=visual_prompt,
                size=config['size'],
                quality=config['quality'],
                style=config['style'],
                n=1
            )
            image_url = response.data[0].url

        elif provider in ('flux', 'qwen', 'musesteamer'):
            config = CONFIG['image_generation'][provider]
            client = OpenAI(
                base_url=config['base_url'],
                api_key=os.getenv("QIANFAN_API_KEY")
            )
            response = client.images.generate(
                model=config['model'],
                prompt=visual_prompt,
                size=config['size'],
                n=1
            )
            image_url = response.data[0].url

        else:
            end_time = time.time()
            elapsed = end_time - start_time
            logger.error(f"[Tool] {func_name} 执行失败，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒，错误: Unknown provider {provider}")
            return f"Error: Unknown provider {provider}"

        # 下载图片
        output_path = get_output_path("images", f"{word}_main.png")
        resp = httpx.get(image_url, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(resp.content)

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[Tool] {func_name} 执行完成，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒")
        return str(output_path)
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.error(f"[Tool] {func_name} 执行失败，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒，错误: {e}")
        raise


# === Tool 4: 生成文字卡片 ===

@tool
def generate_cards(word: str, phonetic: str, example_en: str, example_cn: str) -> str:
    """
    生成视频所需的文字卡片（标题卡、例句卡、结尾卡）。

    Args:
        word: 单词
        phonetic: 音标
        example_en: 英文例句
        example_cn: 中文翻译

    Returns:
        JSON格式的卡片路径信息
    """
    start_time = time.time()
    func_name = "generate_cards"
    logger.info(f"[Tool] {func_name} 开始执行，开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    try:
        config = CONFIG['cards']
        width = CONFIG['video']['width']
        height = CONFIG['video']['height']

        # 加载字体
        try:
            if config['font_path']:
                title_font = ImageFont.truetype(config['font_path'], config['title_font_size'])
                content_font = ImageFont.truetype(config['font_path'], config['content_font_size'])
            else:
                title_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", config['title_font_size'])
                content_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", config['content_font_size'])
        except:
            title_font = ImageFont.load_default()
            content_font = ImageFont.load_default()

        bg_color = config['background_color']
        text_color = config['text_color']
        accent_color = config['accent_color']

        # 标题卡
        title_card = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(title_card)
        word_upper = word.upper()
        bbox = draw.textbbox((0, 0), word_upper, font=title_font)
        word_width = bbox[2] - bbox[0]
        draw.text(((width - word_width) // 2, height // 3), word_upper, fill=accent_color, font=title_font)
        if phonetic:
            bbox = draw.textbbox((0, 0), phonetic, font=content_font)
            draw.text(((width - (bbox[2] - bbox[0])) // 2, height // 3 + 150), phonetic, fill=text_color, font=content_font)
        title_path = get_output_path("cards", f"{word}_title.png")
        title_card.save(title_path)

        # 例句卡
        sentence_card = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(sentence_card)
        draw.text((80, height // 3), example_en, fill=accent_color, font=content_font)
        draw.text((80, height // 3 + 100), example_cn, fill=text_color, font=content_font)
        sentence_path = get_output_path("cards", f"{word}_sentence.png")
        sentence_card.save(sentence_path)

        # 结尾卡
        ending_card = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(ending_card)
        ending_text = "每天一个脑洞词\n关注我"
        bbox = draw.textbbox((0, 0), ending_text, font=content_font)
        draw.text(((width - (bbox[2] - bbox[0])) // 2, height // 2 - 50), ending_text, fill=accent_color, font=content_font, align="center")
        ending_path = get_output_path("cards", f"{word}_ending.png")
        ending_card.save(ending_path)

        result = {
            "title_card": str(title_path),
            "sentence_card": str(sentence_path),
            "ending_card": str(ending_path)
        }
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[Tool] {func_name} 执行完成，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.error(f"[Tool] {func_name} 执行失败，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒，错误: {e}")
        raise


# === Tool 5: 生成音频 ===

@tool
def generate_audio(word: str, segments_json: str) -> str:
    """
    为脚本分段生成TTS配音。

    Args:
        word: 单词（用于文件命名）
        segments_json: JSON格式的分段脚本，如 [{"id": "opening", "text": "...", "seconds": 2.5}, ...]

    Returns:
        JSON格式的音频信息，包含每段的路径和实际时长
    """
    start_time = time.time()
    func_name = "generate_audio"
    logger.info(f"[Tool] {func_name} 开始执行，开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    try:
        segments = json.loads(segments_json)
        audio_timings = []
        current_start = 0.0

        for segment in segments:
            segment_id = segment['id']
            text = segment['text']

            # 选择音色
            if segment_id == "opening":
                voice = CONFIG['audio']['voice_en']
            else:
                voice = CONFIG['audio']['voice_cn']

            audio_path = get_output_path("audio", f"{word}_{segment_id}.mp3")

            # 生成音频
            asyncio.run(_generate_tts(text, voice, str(audio_path)))

            # 获取实际时长
            audio = AudioSegment.from_mp3(str(audio_path))
            duration = len(audio) / 1000.0

            audio_timings.append({
                "id": segment_id,
                "path": str(audio_path),
                "duration": duration,
                "start_time": current_start
            })

            current_start += duration + 0.3  # 间隔

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[Tool] {func_name} 执行完成，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒")
        return json.dumps(audio_timings, ensure_ascii=False)
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.error(f"[Tool] {func_name} 执行失败，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒，错误: {e}")
        raise


async def _generate_tts(text: str, voice: str, output_path: str):
    import edge_tts
    rate = CONFIG['audio']['rate']
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


# === Tool 6: 合成视频 ===

@tool
def compose_video(
    word: str,
    main_image_path: str,
    cards_json: str,
    audio_timings_json: str
) -> str:
    """
    将图片、卡片、音频合成为最终视频。

    Args:
        word: 单词
        main_image_path: 主图路径
        cards_json: JSON格式的卡片路径
        audio_timings_json: JSON格式的音频时长信息

    Returns:
        最终视频路径
    """
    start_time = time.time()
    func_name = "compose_video"
    logger.info(f"[Tool] {func_name} 开始执行，开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    try:
        from moviepy import (
            ImageClip, AudioFileClip, CompositeVideoClip,
            concatenate_videoclips, CompositeAudioClip
        )
        import numpy as np

        config = CONFIG['video']
        width, height = config['width'], config['height']
        fps = config['fps']
        zoom_factor = config['ken_burns_zoom']

        cards = json.loads(cards_json)
        audio_timings = json.loads(audio_timings_json)

        timing_map = {t['id']: t for t in audio_timings}

        clips = []
        audio_clips = []

        # Opening (标题卡)
        opening = timing_map.get('opening', {'duration': 3, 'start_time': 0})
        title_clip = ImageClip(cards['title_card']).with_duration(opening['duration'] + 0.5).resized((width, height))
        clips.append(title_clip)
        if opening.get('path'):
            audio_clips.append(AudioFileClip(opening['path']).with_start(opening['start_time']))

        # Mnemonic (主图 + Ken Burns)
        mnemonic = timing_map.get('mnemonic', {'duration': 12, 'start_time': 3})
        mnemonic_duration = mnemonic['duration'] + 1

        def ken_burns(get_frame, t):
            progress = t / mnemonic_duration
            current_zoom = 1 + (zoom_factor - 1) * progress
            frame = get_frame(t)
            h, w = frame.shape[:2]
            new_h, new_w = int(h / current_zoom), int(w / current_zoom)
            start_y, start_x = (h - new_h) // 2, (w - new_w) // 2
            cropped = frame[start_y:start_y+new_h, start_x:start_x+new_w]
            img = Image.fromarray(cropped).resize((w, h), Image.Resampling.LANCZOS)
            return np.array(img)

        main_clip = ImageClip(main_image_path).with_duration(mnemonic_duration).resized((width, height))
        main_clip = main_clip.transform(ken_burns, apply_to=['mask'])
        clips.append(main_clip)
        if mnemonic.get('path'):
            audio_clips.append(AudioFileClip(mnemonic['path']).with_start(mnemonic['start_time']))

        # Sentence (例句卡)
        sentence = timing_map.get('sentence', {'duration': 8, 'start_time': 15})
        sentence_clip = ImageClip(cards['sentence_card']).with_duration(sentence['duration'] + 0.5).resized((width, height))
        clips.append(sentence_clip)
        if sentence.get('path'):
            audio_clips.append(AudioFileClip(sentence['path']).with_start(sentence['start_time']))

        # Ending (结尾卡)
        ending = timing_map.get('ending', {'duration': 4, 'start_time': 24})
        ending_clip = ImageClip(cards['ending_card']).with_duration(ending['duration'] + 0.5).resized((width, height))
        clips.append(ending_clip)
        if ending.get('path'):
            audio_clips.append(AudioFileClip(ending['path']).with_start(ending['start_time']))

        # 拼接
        final_video = concatenate_videoclips(clips, method="compose")
        if audio_clips:
            final_video = final_video.with_audio(CompositeAudioClip(audio_clips))

        output_path = get_output_path("video", f"{word}_final.mp4")
        final_video.write_videofile(
            str(output_path), fps=fps, codec='libx264', audio_codec='aac',
            logger=None
        )

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[Tool] {func_name} 执行完成，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒")
        return str(output_path)
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.error(f"[Tool] {func_name} 执行失败，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}，耗时: {elapsed:.2f}秒，错误: {e}")
        raise


# === 导出所有 Tools ===

ALL_TOOLS = [
    creative_brainstorm,
    generate_visual_prompt,
    generate_image,
    generate_cards,
    generate_audio,
    compose_video
]


if __name__ == "__main__":
    # 单独测试 generate_image 函数
    # 注意：generate_image 被 @tool 装饰器包装，需要通过 .func 访问原始函数
    visual_prompt = "A cute cartoon ambulance with a playful expression, driving through a colorful city street with sunshine and clouds"
    word = "ambulance"
    result = generate_image.invoke({
        "visual_prompt": visual_prompt,
        "word": word
    })
    print(f"Generated image saved to: {result}")

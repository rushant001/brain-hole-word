"""
Brain-Hole-Word Agent Nodes
使用 LangChain 1.2.0 的新特性：
- with_structured_output() 进行结构化输出
- 支持多种生图模型 (DALL-E, FLUX, Qwen, Muse)
- 精确的音视频同步机制
"""
import os
import asyncio
import httpx
from pathlib import Path
from typing import Literal, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI
from pydub import AudioSegment

from .state import AgentState, MnemonicContent, VisualPlan, AudioTiming
from .utils import CONFIG, get_output_path
from .prompts import (
    CREATIVE_SYSTEM_PROMPT, CREATIVE_USER_PROMPT,
    VISUAL_SYSTEM_PROMPT, VISUAL_USER_PROMPT
)

# === LLM Initialization with Structured Output ===

def get_creative_llm():
    """获取创意生成 LLM，使用结构化输出"""
    base_url = CONFIG['llm'].get('base_url')
    llm = ChatOpenAI(
        model=CONFIG['llm']['creative_model'],
        temperature=CONFIG['llm']['temperature'],
        base_url=base_url
    )
    return llm.with_structured_output(MnemonicContent)

def get_visual_llm():
    """获取视觉规划 LLM，使用结构化输出"""
    base_url = CONFIG['llm'].get('base_url')
    llm = ChatOpenAI(
        model=CONFIG['llm']['visual_model'],
        temperature=0.7,
        base_url=base_url
    )
    return llm.with_structured_output(VisualPlan)


# === Node 1: Creative_Brain ===

def creative_brain_node(state: AgentState) -> dict:
    """创意策划节点"""
    word = state['word']
    phonetic = state.get('phonetic', '')
    
    print(f"🧠 [Creative_Brain] 正在为 '{word}' 构思脑洞记忆法...")
    
    llm = get_creative_llm()
    
    messages = [
        SystemMessage(content=CREATIVE_SYSTEM_PROMPT),
        HumanMessage(content=CREATIVE_USER_PROMPT.format(word=word, phonetic=phonetic))
    ]
    
    result: MnemonicContent = llm.invoke(messages)
    
    print(f"   ✅ 策略: {result.strategy}")
    print(f"   ✅ 口号: {result.slogan}")
    print(f"   ✅ 脚本分段: {len(result.narration_segments)} 段")
    
    return {
        "mnemonic": result,
        "current_step": "creative_brain_done"
    }


# === Node 2: Visual_Planner ===

def visual_planner_node(state: AgentState) -> dict:
    """视觉规划节点"""
    mnemonic = state.get('mnemonic')
    if not mnemonic:
        return {"error": "No mnemonic content found"}
    
    print(f"🎨 [Visual_Planner] 正在生成绘图 Prompt...")
    
    llm = get_visual_llm()
    
    messages = [
        SystemMessage(content=VISUAL_SYSTEM_PROMPT),
        HumanMessage(content=VISUAL_USER_PROMPT.format(
            word=state['word'],
            slogan=mnemonic.slogan,
            story_scene=mnemonic.story_scene
        ))
    ]
    
    result: VisualPlan = llm.invoke(messages)
    
    print(f"   ✅ 主场景 Prompt: {result.main_scene_prompt[:80]}...")
    
    return {
        "visual_plan": result,
        "current_step": "visual_planner_done"
    }


# === Node 3: Image_Generator (多模型支持) ===

def image_generator_node(state: AgentState) -> dict:
    """图片生成节点：支持多种生图模型"""
    
    # 检查是否使用手动图片
    if state.get('use_manual_image') and state.get('manual_image_url'):
        print(f"🖼️ [Image_Generator] 使用用户提供的图片...")
        image_path = download_image(state['manual_image_url'], state['word'])
        return {"main_image_path": str(image_path)}
    
    visual_plan = state.get('visual_plan')
    if not visual_plan:
        return {"error": "No visual plan found"}
    
    provider = CONFIG['image_generation']['provider']
    print(f"🖼️ [Image_Generator] 使用 {provider} 生成图片...")
    
    prompt = visual_plan.main_scene_prompt
    
    if provider == 'dall-e-3':
        image_url = generate_dalle(prompt)
    elif provider == 'flux':
        image_url = generate_flux(prompt)
    elif provider == 'qwen':
        image_url = generate_qwen(prompt)
    elif provider == 'muse':
        image_url = generate_muse(prompt)
    else:
        return {"error": f"Unknown provider: {provider}"}
    
    if image_url:
        image_path = download_image(image_url, state['word'])
        print(f"   ✅ 图片已保存: {image_path}")
        return {"main_image_path": str(image_path)}
    else:
        return {"error": "Image generation failed"}


def generate_dalle(prompt: str) -> str:
    """DALL-E 3 生图"""
    config = CONFIG['image_generation']['dalle']
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=config['size'],
        quality=config['quality'],
        style=config['style'],
        n=1
    )
    return response.data[0].url


def generate_flux(prompt: str) -> str:
    """FLUX.1-schnell 生图 (兼容 OpenAI API 格式)"""
    config = CONFIG['image_generation']['flux']
    client = OpenAI(
        base_url=config['base_url'],
        api_key=os.getenv("FLUX_API_KEY", os.getenv("OPENAI_API_KEY"))
    )
    response = client.images.generate(
        model=config['model'],
        prompt=prompt,
        size=config['size'],
        n=1
    )
    return response.data[0].url


def generate_qwen(prompt: str) -> str:
    """Qwen/通义万象 生图"""
    config = CONFIG['image_generation']['qwen']
    client = OpenAI(
        base_url=config['base_url'],
        api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY"))
    )
    response = client.images.generate(
        model=config['model'],
        prompt=prompt,
        size=config['size'],
        n=1
    )
    return response.data[0].url


def generate_muse(prompt: str) -> str:
    """MuseSteamer-Air-Image 生图"""
    config = CONFIG['image_generation']['muse']
    client = OpenAI(
        base_url=config['base_url'],
        api_key=os.getenv("MUSE_API_KEY", os.getenv("OPENAI_API_KEY"))
    )
    response = client.images.generate(
        model=config['model'],
        prompt=prompt,
        size=config['size'],
        n=1
    )
    return response.data[0].url


def download_image(url: str, word: str) -> Path:
    """下载图片到本地"""
    output_path = get_output_path("images", f"{word}_main.png")
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        f.write(response.content)
    return output_path


# === Node 4: Card_Generator ===

def card_generator_node(state: AgentState) -> dict:
    """文字卡片生成节点"""
    from PIL import Image, ImageDraw, ImageFont
    
    print(f"📝 [Card_Generator] 生成文字卡片...")
    
    word = state['word']
    phonetic = state.get('phonetic', '')
    mnemonic = state.get('mnemonic')
    
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
    word_x = (width - word_width) // 2
    draw.text((word_x, height // 3), word_upper, fill=accent_color, font=title_font)
    if phonetic:
        bbox = draw.textbbox((0, 0), phonetic, font=content_font)
        phonetic_width = bbox[2] - bbox[0]
        draw.text(((width - phonetic_width) // 2, height // 3 + 150), phonetic, fill=text_color, font=content_font)
    title_path = get_output_path("cards", f"{word}_title.png")
    title_card.save(title_path)
    
    # 例句卡
    sentence_card = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(sentence_card)
    if mnemonic:
        draw.text((80, height // 3), mnemonic.example_sentence_en, fill=accent_color, font=content_font)
        draw.text((80, height // 3 + 100), mnemonic.example_sentence_cn, fill=text_color, font=content_font)
    sentence_path = get_output_path("cards", f"{word}_sentence.png")
    sentence_card.save(sentence_path)
    
    # 结尾卡
    ending_card = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(ending_card)
    ending_text = "每天一个脑洞词\n关注我"
    bbox = draw.textbbox((0, 0), ending_text, font=content_font)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, height // 2 - 50), ending_text, fill=accent_color, font=content_font, align="center")
    ending_path = get_output_path("cards", f"{word}_ending.png")
    ending_card.save(ending_path)
    
    print(f"   ✅ 已生成 3 张卡片")
    
    return {
        "title_card_path": str(title_path),
        "sentence_card_path": str(sentence_path),
        "ending_card_path": str(ending_path),
        "current_step": "card_generator_done"
    }


# === Node 5: Audio_Producer (带精确时长) ===

def audio_producer_node(state: AgentState) -> dict:
    """
    音频合成节点：分段生成音频，并记录精确时长用于视频同步
    """
    print(f"🎙️ [Audio_Producer] 生成音频...")
    
    word = state['word']
    mnemonic = state.get('mnemonic')
    
    if not mnemonic:
        return {"error": "No mnemonic content found"}
    
    provider = CONFIG['audio']['provider']
    audio_timings = []
    current_start = 0.0
    
    if provider == 'edge-tts':
        for segment in mnemonic.narration_segments:
            segment_id = segment.segment_id
            text = segment.text
            
            # 确定使用的音色
            if segment_id == "opening":
                voice = CONFIG['audio']['voice_en']
            else:
                voice = CONFIG['audio']['voice_cn']
            
            audio_path = get_output_path("audio", f"{word}_{segment_id}.mp3")
            
            # 生成音频
            asyncio.run(_generate_edge_tts_single(text, voice, str(audio_path)))
            
            # 获取实际时长
            audio = AudioSegment.from_mp3(str(audio_path))
            duration = len(audio) / 1000.0  # 转换为秒
            
            timing = {
                "segment_id": segment_id,
                "audio_path": str(audio_path),
                "duration_seconds": duration,
                "start_time": current_start
            }
            audio_timings.append(timing)
            
            print(f"   ✅ {segment_id}: {duration:.2f}s (从 {current_start:.2f}s 开始)")
            
            # 累加时间 (加上片段间隔)
            current_start += duration + CONFIG['video'].get('padding_seconds', 0.3)
        
        total_duration = current_start
        print(f"   📊 总时长: {total_duration:.2f}s")
        
        return {
            "audio_timings": audio_timings,
            "total_audio_duration": total_duration,
            "current_step": "audio_producer_done"
        }
    else:
        return {"error": f"Unknown audio provider: {provider}"}


async def _generate_edge_tts_single(text: str, voice: str, output_path: str):
    """生成单个 Edge TTS 音频"""
    import edge_tts
    rate = CONFIG['audio']['rate']
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


# === Node 6: Video_Composer (基于音频时长动态合成) ===

def video_composer_node(state: AgentState) -> dict:
    """
    视频合成节点：基于音频实际时长动态生成视频
    确保音画完美同步
    """
    from moviepy import (
        ImageClip, AudioFileClip, CompositeVideoClip, 
        concatenate_videoclips, CompositeAudioClip
    )
    import numpy as np
    
    print(f"🎬 [Video_Composer] 合成视频 (基于音频时长)...")
    
    word = state['word']
    config = CONFIG['video']
    width, height = config['width'], config['height']
    fps = config['fps']
    zoom_factor = config['ken_burns_zoom']
    
    audio_timings = state.get('audio_timings', [])
    if not audio_timings:
        return {"error": "No audio timings found"}
    
    # 构建时长映射
    timing_map = {t['segment_id']: t for t in audio_timings}
    
    main_image = state.get('main_image_path')
    title_card = state.get('title_card_path')
    sentence_card = state.get('sentence_card_path')
    ending_card = state.get('ending_card_path')
    
    if not all([main_image, title_card]):
        return {"error": "Missing required assets"}
    
    clips = []
    audio_clips = []
    
    # === Clip 1: Opening (标题卡 + 单词发音) ===
    opening_timing = timing_map.get('opening', {'duration_seconds': 3, 'start_time': 0})
    opening_duration = opening_timing['duration_seconds'] + 0.5  # 加缓冲
    
    title_clip = ImageClip(title_card).set_duration(opening_duration).resize((width, height))
    clips.append(title_clip)
    
    if opening_timing.get('audio_path'):
        audio_clips.append(
            AudioFileClip(opening_timing['audio_path']).set_start(opening_timing['start_time'])
        )
    
    # === Clip 2: Mnemonic (主图 + 脑洞解说 + Ken Burns) ===
    mnemonic_timing = timing_map.get('mnemonic', {'duration_seconds': 15, 'start_time': 3})
    mnemonic_duration = mnemonic_timing['duration_seconds'] + 1  # 加缓冲
    
    def ken_burns_effect(get_frame, t):
        """Ken Burns 推拉效果"""
        progress = t / mnemonic_duration
        current_zoom = 1 + (zoom_factor - 1) * progress
        
        frame = get_frame(t)
        h, w = frame.shape[:2]
        
        new_h = int(h / current_zoom)
        new_w = int(w / current_zoom)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        
        cropped = frame[start_y:start_y+new_h, start_x:start_x+new_w]
        
        from PIL import Image
        img = Image.fromarray(cropped)
        img = img.resize((w, h), Image.LANCZOS)
        return np.array(img)
    
    main_clip = ImageClip(main_image).set_duration(mnemonic_duration).resize((width, height))
    main_clip = main_clip.fl(ken_burns_effect, apply_to=['mask'])
    clips.append(main_clip)
    
    if mnemonic_timing.get('audio_path'):
        audio_clips.append(
            AudioFileClip(mnemonic_timing['audio_path']).set_start(mnemonic_timing['start_time'])
        )
    
    # === Clip 3: Sentence (例句卡) ===
    sentence_timing = timing_map.get('sentence', {'duration_seconds': 8, 'start_time': 18})
    sentence_duration = sentence_timing['duration_seconds'] + 0.5
    
    if sentence_card:
        sentence_clip = ImageClip(sentence_card).set_duration(sentence_duration).resize((width, height))
        clips.append(sentence_clip)
    
    if sentence_timing.get('audio_path'):
        audio_clips.append(
            AudioFileClip(sentence_timing['audio_path']).set_start(sentence_timing['start_time'])
        )
    
    # === Clip 4: Ending (结尾卡) ===
    ending_timing = timing_map.get('ending', {'duration_seconds': 4, 'start_time': 26})
    ending_duration = ending_timing['duration_seconds'] + 0.5
    
    if ending_card:
        ending_clip = ImageClip(ending_card).set_duration(ending_duration).resize((width, height))
        clips.append(ending_clip)
    
    if ending_timing.get('audio_path'):
        audio_clips.append(
            AudioFileClip(ending_timing['audio_path']).set_start(ending_timing['start_time'])
        )
    
    # 拼接视频
    final_video = concatenate_videoclips(clips, method="compose")
    
    # 合成音频轨道
    if audio_clips:
        combined_audio = CompositeAudioClip(audio_clips)
        final_video = final_video.set_audio(combined_audio)
    
    # 输出
    output_path = get_output_path("video", f"{word}_final.mp4")
    final_video.write_videofile(
        str(output_path),
        fps=fps,
        codec='libx264',
        audio_codec='aac',
        verbose=False,
        logger=None
    )
    
    total_duration = state.get('total_audio_duration', sum(c.duration for c in clips))
    print(f"   ✅ 视频已生成: {output_path}")
    print(f"   📊 视频总时长: {total_duration:.2f}s")
    
    return {
        "final_video_path": str(output_path),
        "current_step": "video_composer_done"
    }

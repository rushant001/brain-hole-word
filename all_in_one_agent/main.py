"""
Brain-Hole-Word 入口
支持两种模式：
1. Agent 模式：使用 ReAct Agent 自主执行（智能错误处理）
2. Pipeline 模式：按固定流程执行（高效可控）
"""
import argparse
import json
from pathlib import Path

from tools import (
    creative_brainstorm,
    generate_visual_prompt,
    generate_image,
    generate_cards,
    generate_audio,
    compose_video,
    CONFIG
)


def run_pipeline(word: str) -> str:
    """
    纯工程模式：按固定顺序执行所有步骤
    不使用 Agent，直接调用 Tool 函数
    
    优点：可控、高效、无额外 Token 消耗
    缺点：没有智能错误恢复
    """
    print(f"\n🚀 [Pipeline Mode] 开始处理: {word}")
    print("=" * 60)
    
    # Step 1: 创意策划（音标由 LLM 自动生成）
    print("\n📌 Step 1: 创意策划...")
    creative_result = creative_brainstorm.invoke({"word": word})
    
    if creative_result.startswith("Error"):
        print(f"   ❌ {creative_result}")
        return creative_result
    
    creative_data = json.loads(creative_result)
    print(f"   ✅ 音标: {creative_data.get('phonetic', 'N/A')}")
    print(f"   ✅ 策略: {creative_data.get('strategy', 'N/A')}")
    print(f"   ✅ 口号: {creative_data.get('slogan', 'N/A')}")
    
    # Step 2: 生成绘图 Prompt
    print("\n📌 Step 2: 生成绘图 Prompt...")
    visual_prompt = generate_visual_prompt.invoke({
        "word": word,
        "slogan": creative_data['slogan'],
        "story_scene": creative_data['story_scene']
    })
    print(f"   ✅ Prompt: {visual_prompt[:80]}...")
    
    # Step 3: 生成图片
    print("\n📌 Step 3: 生成图片...")
    main_image_path = generate_image.invoke({
        "visual_prompt": visual_prompt,
        "word": word
    })
    
    if main_image_path.startswith("Error"):
        print(f"   ❌ {main_image_path}")
        return main_image_path
    
    print(f"   ✅ 图片: {main_image_path}")
    
    # Step 4: 生成卡片
    print("\n📌 Step 4: 生成文字卡片...")
    cards_json = generate_cards.invoke({
        "word": word,
        "phonetic": creative_data.get('phonetic', ''),
        "example_en": creative_data['example_en'],
        "example_cn": creative_data['example_cn']
    })
    print(f"   ✅ 卡片已生成")
    
    # Step 5: 生成音频
    print("\n📌 Step 5: 生成音频...")
    segments_json = json.dumps(creative_data['segments'], ensure_ascii=False)
    audio_timings_json = generate_audio.invoke({
        "word": word,
        "segments_json": segments_json
    })
    audio_timings = json.loads(audio_timings_json)
    total_duration = sum(t['duration'] for t in audio_timings)
    print(f"   ✅ 音频总时长: {total_duration:.1f}s")
    
    # Step 6: 合成视频
    print("\n📌 Step 6: 合成视频...")
    video_path = compose_video.invoke({
        "word": word,
        "main_image_path": main_image_path,
        "cards_json": cards_json,
        "audio_timings_json": audio_timings_json
    })
    
    print("\n" + "=" * 60)
    print(f"🎉 完成! 视频路径: {video_path}")
    
    return video_path


def run_agent_mode(word: str) -> str:
    """
    Agent 模式：使用 ReAct Agent 自主执行
    
    优点：智能错误恢复、动态调整
    缺点：额外 Token 消耗、不完全可控
    """
    from agent import run_agent
    
    print(f"\n🤖 [Agent Mode] 启动 ReAct Agent...")
    print("=" * 60)
    
    result = run_agent(word)
    
    return result.get('output', 'Agent 执行完成')


def main(word, mode):
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🧠  脑 洞 单 词 Agent  (All-in-One)  🧠                  ║
║                                                              ║
║     输入单词，自动生成抖音短视频                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📝 单词: {word}")
    print(f"🔧 模式: {mode}")
    
    if mode == "agent":
        result = run_agent_mode(word)
    else:
        result = run_pipeline(word)
    
    print(f"\n✨ 最终结果: {result}")


if __name__ == "__main__":
    main('fleece', 'pipeline')

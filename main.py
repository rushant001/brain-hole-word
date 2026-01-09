"""
Brain-Hole-Word Agent 入口
支持 Human-in-the-loop 交互
"""
from src.graph import app
from src.utils import ensure_output_dir


def print_banner():
    """打印启动 Banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🧠  脑 洞 单 词 Agent  (Brain-Hole-Word)  🧠             ║
║                                                              ║
║     输入一个单词，自动生成抖音短视频素材                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def run_phase_1(word: str, phonetic: str, thread_config: dict) -> dict:
    """
    Phase 1: 创意与视觉规划
    运行到 Image_Generator 前暂停
    """
    print("\n" + "="*60)
    print("📌 Phase 1: 创意与视觉规划")
    print("="*60)
    
    initial_state = {
        "word": word,
        "phonetic": phonetic,
        "use_manual_image": False,
        "manual_image_url": None
    }
    
    for event in app.stream(initial_state, thread_config):
        for key, value in event.items():
            if key.startswith("__"):
                continue
            print(f"\n  ✅ [{key}] 完成")
            
            if key == "Creative_Brain" and value.get('mnemonic'):
                m = value['mnemonic']
                print(f"     策略: {m.strategy}")
                print(f"     口号: {m.slogan}")
                print(f"     场景: {m.story_scene[:80]}...")
            
            if key == "Visual_Planner" and value.get('visual_plan'):
                v = value['visual_plan']
                print(f"     主场景 Prompt:")
                print(f"     {v.main_scene_prompt[:100]}...")
    
    return app.get_state(thread_config)


def human_review(thread_config: dict) -> bool:
    """
    Human-in-the-loop: 用户确认视觉方案
    返回 True 表示继续，False 表示取消
    """
    state = app.get_state(thread_config)
    
    if not state.next:
        print("\n⚠️ 工作流未暂停，无需人工确认")
        return True
    
    print("\n" + "="*60)
    print("⏸️  Human-in-the-Loop: 请确认视觉方案")
    print("="*60)
    
    visual_plan = state.values.get('visual_plan')
    if visual_plan:
        print(f"\n📝 生成的绘图 Prompt:")
        print(f"\n   [主场景]")
        print(f"   {visual_plan.main_scene_prompt}")
        print(f"\n   [细节特写]")
        print(f"   {visual_plan.detail_shot_prompt}")
    
    print("\n" + "-"*60)
    print("请选择操作:")
    print("  [1] 使用 AI 自动生成图片 (DALL-E 3)")
    print("  [2] 我提供图片 URL (省钱模式)")
    print("  [3] 取消任务")
    print("-"*60)
    
    choice = input("请输入选项 (1/2/3): ").strip()
    
    if choice == "1":
        print("\n✅ 确认使用 AI 生图...")
        # 不需要更新 state，直接继续
        return True
    
    elif choice == "2":
        url = input("请输入图片 URL: ").strip()
        if url:
            # 更新 state，注入用户提供的 URL
            app.update_state(thread_config, {
                "use_manual_image": True,
                "manual_image_url": url
            })
            print(f"\n✅ 已设置手动图片: {url[:50]}...")
            return True
        else:
            print("❌ URL 为空，取消操作")
            return False
    
    else:
        print("❌ 任务已取消")
        return False


def run_phase_2(thread_config: dict):
    """
    Phase 2: 素材生成与视频合成
    从暂停点恢复执行
    """
    print("\n" + "="*60)
    print("📌 Phase 2: 素材生成与视频合成")
    print("="*60)
    
    # 传入 None 表示从暂停点恢复
    for event in app.stream(None, thread_config):
        for key, value in event.items():
            if key.startswith("__"):
                continue
            print(f"\n  ✅ [{key}] 完成")
            
            if key == "Image_Generator":
                print(f"     图片路径: {value.get('main_image_path')}")
            
            if key == "Card_Generator":
                print(f"     标题卡: {value.get('title_card_path')}")
            
            if key == "Audio_Producer":
                print(f"     解说音频: {value.get('narration_audio_path')}")
            
            if key == "Video_Composer":
                print(f"\n🎉 最终视频: {value.get('final_video_path')}")


def main():
    """主入口"""
    print_banner()
    ensure_output_dir()
    
    # 获取输入
    word = input("请输入单词 (例如: Ambulance): ").strip()
    if not word:
        print("❌ 单词不能为空")
        return
    
    phonetic = input("请输入音标 (可选，直接回车跳过): ").strip()
    
    # 配置线程
    thread_config = {"configurable": {"thread_id": f"word_{word}"}}
    
    try:
        # Phase 1: 创意规划
        run_phase_1(word, phonetic, thread_config)
        
        # Human Review
        if not human_review(thread_config):
            return
        
        # Phase 2: 素材生成
        run_phase_2(thread_config)
        
        print("\n" + "="*60)
        print("✨ 任务完成！")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        raise


if __name__ == "__main__":
    main()

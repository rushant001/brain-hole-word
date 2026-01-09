from typing import TypedDict, List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

# === Pydantic Models for Structured Output (LangChain 1.2.0 Feature) ===

class NarrationSegment(BaseModel):
    """解说脚本的分段结构 - 用于精确控制音视频同步"""
    segment_id: Literal["opening", "mnemonic", "sentence", "ending"]
    text: str = Field(description="这一段的解说文本")
    estimated_seconds: float = Field(description="预估朗读时长(秒)")


class MnemonicContent(BaseModel):
    """创意内容的结构化输出"""
    strategy: Literal["谐音", "画面"] = Field(description="记忆策略类型")
    pronunciation_mnemonic: str = Field(default="", description="谐音口诀，如果是画面策略则为空")
    story_scene: str = Field(description="详细的画面场景描述，越夸张越好")
    slogan: str = Field(description="一句简短的洗脑助记词")
    example_sentence_en: str = Field(description="英文例句")
    example_sentence_cn: str = Field(description="中文翻译")
    
    # 分段脚本 - 用于精确同步
    narration_segments: List[NarrationSegment] = Field(
        description="解说脚本分段，必须包含4段：opening(读单词)、mnemonic(脑洞讲解)、sentence(例句)、ending(结尾口号)"
    )


class VisualPlan(BaseModel):
    """视觉规划的结构化输出"""
    main_scene_prompt: str = Field(description="主场景的英文绘图Prompt")
    detail_shot_prompt: str = Field(description="细节特写的英文绘图Prompt")
    style_notes: str = Field(description="风格备注")


# === 音频时长信息 (运行时生成) ===

class AudioTiming(BaseModel):
    """音频时长信息 - 用于视频合成时的精确对齐"""
    segment_id: str
    audio_path: str
    duration_seconds: float
    start_time: float  # 在最终视频中的起始时间


# === LangGraph State Definition ===

class AgentState(TypedDict):
    """Agent 状态定义"""
    # Input
    word: str
    phonetic: Optional[str]
    
    # Node 1: Creative_Brain 输出
    mnemonic: Optional[MnemonicContent]
    
    # Node 2: Visual_Planner 输出
    visual_plan: Optional[VisualPlan]
    
    # Human-in-the-loop 控制
    use_manual_image: bool
    manual_image_url: Optional[str]
    
    # Node 3: Image_Generator 输出
    main_image_path: Optional[str]
    
    # Node 4: Card_Generator 输出
    title_card_path: Optional[str]
    sentence_card_path: Optional[str]
    ending_card_path: Optional[str]
    
    # Node 5: Audio_Producer 输出 (新增时长信息)
    audio_timings: Optional[List[dict]]  # List[AudioTiming] 序列化
    total_audio_duration: Optional[float]
    
    # Node 6: Video_Composer 输出
    final_video_path: Optional[str]
    
    # Metadata
    error: Optional[str]
    current_step: Optional[str]

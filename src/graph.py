"""
Brain-Hole-Word LangGraph Definition
使用 LangGraph 的最新特性：
- InMemorySaver checkpointer 实现 Human-in-the-loop
- interrupt_before 在指定节点前暂停
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import AgentState
from .nodes import (
    creative_brain_node,
    visual_planner_node,
    image_generator_node,
    card_generator_node,
    audio_producer_node,
    video_composer_node
)
from .utils import CONFIG


def build_graph() -> StateGraph:
    """构建 LangGraph 工作流"""
    
    workflow = StateGraph(AgentState)
    
    # === 添加节点 ===
    workflow.add_node("Creative_Brain", creative_brain_node)
    workflow.add_node("Visual_Planner", visual_planner_node)
    workflow.add_node("Image_Generator", image_generator_node)
    workflow.add_node("Card_Generator", card_generator_node)
    workflow.add_node("Audio_Producer", audio_producer_node)
    workflow.add_node("Video_Composer", video_composer_node)
    
    # === 定义边 ===
    workflow.set_entry_point("Creative_Brain")
    workflow.add_edge("Creative_Brain", "Visual_Planner")
    workflow.add_edge("Visual_Planner", "Image_Generator")
    workflow.add_edge("Image_Generator", "Card_Generator")
    workflow.add_edge("Card_Generator", "Audio_Producer")
    workflow.add_edge("Audio_Producer", "Video_Composer")
    workflow.add_edge("Video_Composer", END)
    
    return workflow


def compile_app():
    """编译 Graph 并配置 Checkpointer"""
    workflow = build_graph()
    
    # 使用 InMemorySaver 作为 Checkpointer (LangGraph v1 API)
    checkpointer = InMemorySaver()
    
    # 判断是否启用 Human-in-the-loop
    interrupt_nodes = []
    if CONFIG['workflow']['human_in_the_loop']:
        interrupt_nodes = ["Image_Generator"]  # 在生图前暂停
    
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes
    )
    
    return app


# 导出编译后的 App
app = compile_app()

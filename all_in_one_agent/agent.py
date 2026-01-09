"""
Brain-Hole-Word ReAct Agent
使用 LangChain v1 的 create_agent 创建自主执行的 Agent
"""

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from tools import ALL_TOOLS, PROMPTS
from utils import get_llm

def create_video_agent(word: str):
    """
    创建脑洞单词视频制作 Agent

    Args:
        word: 要处理的单词（音标由 LLM 自动生成）

    Returns:
        配置好的 Agent (LangChain v1)
    """
    # 获取 Agent System Prompt（只需要 word）
    agent_prompt = PROMPTS['agent']['system_prompt'].format(word=word)

    # 使用通用的 get_llm 获取模型
    llm = get_llm()

    # 创建 Agent (LangChain v1 API)
    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=agent_prompt,
        # checkpointer=InMemorySaver()
    )

    return agent


def run_agent(word: str) -> dict:
    """
    运行 Agent 生成视频
    
    Args:
        word: 单词（音标由 Agent 自动处理）
    
    Returns:
        包含结果和中间步骤的字典
    """
    
    agent = create_video_agent(word)
    
    # LangChain v1 使用 messages 格式
    config = RunnableConfig(recursion_limit=30)
    result = agent.invoke({
        "messages": [HumanMessage(content=f"请为单词 '{word}' 生成一个脑洞记忆短视频。按顺序执行所有步骤，注意检查每步结果。")]
    }, config=config)
    
    return result


if __name__ == "__main__":
    word = 'tree'
    print(f"\n🧠 启动脑洞单词 Agent...")
    print(f"📝 单词: {word}")
    print("=" * 60)
    
    result = run_agent(word)
    
    print("\n" + "=" * 60)
    print("✨ Agent 执行完成!")
    print(f"📹 结果: {result.get('output', 'N/A')}")

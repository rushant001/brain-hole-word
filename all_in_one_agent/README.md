# All-in-One Agent 版本

这是 Brain-Hole-Word 的 **ReAct Agent** 实现版本。

## 架构对比

| 特性 | LangGraph 版本 | All-in-One Agent 版本 |
|:---|:---|:---|
| 流程控制 | 显式 Graph 定义 | Agent 自主决策 |
| 错误处理 | 需手写重试逻辑 | Agent 自动分析重试 |
| Token 消耗 | 仅 LLM 调用消耗 | Agent 推理也消耗 |
| 可控性 | 高 | 中 |
| 适用场景 | 生产环境 | 需要智能决策时 |

## 文件结构

```
all_in_one_agent/
├── config.yaml      # 配置文件
├── prompts.yaml     # 所有 Prompt 定义
├── tools.py         # 功能封装为 LangChain Tools
├── agent.py         # ReAct Agent 定义
└── main.py          # 入口，支持 Agent/Pipeline 两种模式
```

## 使用方式

### Pipeline 模式（推荐，省 Token）

```bash
cd all_in_one_agent
python main.py Ambulance -p "/ˈæmbjələns/" --mode pipeline
```

### Agent 模式（智能决策）

```bash
python main.py Ambulance --mode agent
```

## Agent 的价值

在这个项目中，Agent 的核心价值是：

1. **智能错误恢复**：生图失败时，Agent 分析原因并重试
2. **质量自检**：检查创意是否合理
3. **动态调整**：发现谐音不好时自动切换策略

如果流程固定且不需要动态决策，**Pipeline 模式更高效**。

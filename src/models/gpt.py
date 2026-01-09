# -*- coding: utf-8 -*-

"""
OpenAI 系列模型
https://ku.baidu-int.com/d/_7RGPNO_gfsR3A
价格：https://platform.openai.com/docs/pricing
logits:https://chatgpt.com/share/68c28667-74c4-800d-822f-edcc0a703146
Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
author: xuhong02
contact: xuhong02@baidu.com
datetime: 2025/3/19 18:50
"""
import json
import httpx
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Optional, List

# 默认模型配置
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_TEMPERATURE = 0
DEFAULT_TIMEOUT = 120  # 降低到120秒，避免单个请求占用连接过久
DEFAULT_PROXY = "http://agent.baidu.com:8891"
DEFAULT_BASE_URL = "http://yy.dbh.baidu-int.com"


def get_llm(model_name: Optional[str] = None, temperature: float = None, max_retries=2, max_tokens=30000):
    """
    获取OpenAI模型实例（非单例模式，每次创建新实例）

    Args:
        model_name: OpenAI模型名称，如果为None则使用默认模型
        temperature: 温度参数，如果为None则使用默认值
        max_retries: 最大重试次数，默认2次
        max_tokens: 最大输出token数

    Returns:
        ChatOpenAI: OpenAI模型实例
    """
    model_name = model_name or DEFAULT_MODEL
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    
    # 创建自定义的 httpx.AsyncClient，配置适合低并发场景的小连接池
    http_async_client = httpx.AsyncClient(
        # proxy=proxy,  # openrouter不需要代理
        limits=httpx.Limits(
            max_connections=20,           # 小连接池，适合低并发场景
            max_keepalive_connections=5   # 保持少量活跃连接
        ),
        timeout=httpx.Timeout(
            timeout=DEFAULT_TIMEOUT,      # 总超时120秒
            pool=30.0                     # 等待获取连接的超时时间30秒
        )
    )
    
    # 每次创建新的 ChatOpenAI 实例，避免连接池共享导致的问题
    return ChatOpenAI(
        model=model_name,
        http_async_client=http_async_client,
        openai_proxy=None,  # 禁用 openai_proxy，因为代理已在 http_async_client 中配置
        temperature=temp,
        # timeout 已在 http_async_client 中配置，这里不需要重复设置
        base_url=DEFAULT_BASE_URL,
        max_tokens=max_tokens,
        max_retries=max_retries
    )


from pydantic import BaseModel, Field


class TechnicalContent(BaseModel):
    """技术内容结构"""

    concept_overview: str = Field(
        ...,
        description="技术概念概述 - 简明扼要地介绍技术是什么",
        examples=["Docker是一个开源的容器化平台，用于将应用程序及其依赖项打包到轻量级、可移植的容器中"]
    )

    key_features: List[str] = Field(
        ...,
        description="关键技术特性 - 列出技术的主要特点和功能",
        examples=["轻量级虚拟化", "可移植性", "版本控制", "快速部署"]
    )

    implementation_guide: str = Field(
        ...,
        description="实现指南 - 详细说明如何使用该技术，包括基本步骤和最佳实践",
        examples=["1. 安装Docker 2. 编写Dockerfile 3. 构建镜像 4. 运行容器..."]
    )

    common_use_cases: List[str] = Field(
        ...,
        description="常见应用场景 - 列出技术的主要使用场景",
        examples=["微服务部署", "开发环境统一", "CI/CD流水线", "云原生应用"]
    )


if __name__ == "__main__":
    template = """问题: {question}

    回答: 让我逐步思考这个问题。"""

    prompt = PromptTemplate.from_template(template)

    question = "大模型技术中endpoint是什么意思"

    # 1.让大模型按照自定义的 Pydantic 结构输出
    # llm_chain = prompt | get_llm().with_structured_output(TechnicalContent)
    # ret = llm_chain.invoke({'question': question})
    # print(json.dumps(ret.model_dump(), ensure_ascii=False, indent=2))

    _openai_llm_ = ChatOpenAI(
        model=DEFAULT_MODEL,
        # openai_proxy=DEFAULT_PROXY,
        temperature=DEFAULT_TEMPERATURE,
        timeout=DEFAULT_TIMEOUT,
        base_url=DEFAULT_BASE_URL,
        max_tokens=1024,
        max_retries=1,
        logprobs=True
    )
    llm_chain = prompt | _openai_llm_.with_structured_output(TechnicalContent, include_raw=True)
    ret = llm_chain.invoke({'question': question})
    print(json.dumps(ret['parsed'].model_dump(), ensure_ascii=False, indent=2))
    # log_probs_info = ret['raw'].response_metadata["logprobs"]["content"]
    # print(json.dumps(log_probs_info, ensure_ascii=False, indent=2))
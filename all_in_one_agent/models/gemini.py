# -*- coding: utf-8 -*-

"""
Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
author: xuhong02
contact: xuhong02@baidu.com
datetime: 2025/12/19 18:40
"""
import json
from typing import Optional, List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_TEMPERATURE = 0
DEFAULT_TIMEOUT = 120
DEFAULT_BASE_URL = "http://yy.dbh.baidu-int.com"

def get_llm(model_name: Optional[str] = 'gemini-3-flash-preview', temperature: float = None, max_retries=2, max_tokens=8192):
    """
    获取 Google Gemini 模型实例

    Args:
        model_name: 模型名称，默认 gemini-3-flash-preview
        temperature: 温度参数，如果为None则使用默认值
        max_retries: 最大重试次数
        max_tokens: 最大输出token数

    Returns:
        ChatGoogleGenerativeAI: Gemini模型实例
    """
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE

    # 配置安全设置，避免模型过度拒绝回答 (根据业务需求调整)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temp,
        timeout=DEFAULT_TIMEOUT,
        max_retries=max_retries,
        max_output_tokens=max_tokens,
        base_url=DEFAULT_BASE_URL,
        # safety_settings=safety_settings,
        # client_options={"api_endpoint": DEFAULT_BASE_URL},
    )

class TechnicalContent(BaseModel):
    """技术内容结构"""

    concept_overview: str = Field(
        ...,
        description="技术概念概述 - 100字简明扼要地介绍技术是什么",
        examples=["Docker是一个开源的容器化平台"]
    )

    key_features: List[str] = Field(
        ...,
        description="关键技术特性 - 列出3个技术的主要特点",
        examples=["轻量级虚拟化", "可移植性"]
    )

if __name__ == "__main__":
    template = """问题: {question}，用中文回答"""

    prompt = PromptTemplate.from_template(template)

    question = "Google Gemini 模型中的Context Window是什么意思"

    _gemini_llm = get_llm()

    llm_chain = prompt | _gemini_llm.with_structured_output(TechnicalContent, include_raw=True)

    print(f"正在调用模型: {_gemini_llm.model} ...")

    try:
        ret = llm_chain.invoke({'question': question})

        # 打印解析后的结构化数据
        print("\n=== 解析后的输出 (JSON) ===")
        print(json.dumps(ret['parsed'].model_dump(), ensure_ascii=False, indent=2))

        # 打印 Token 使用情况 (如果有)
        if ret.get('raw') and hasattr(ret['raw'], 'response_metadata'):
            print("\n=== 元数据 (Usage) ===")
            usage = ret['raw'].response_metadata.get('usage_metadata')
            print(usage)

    except Exception as e:
        print(f"调用出错: {e}")
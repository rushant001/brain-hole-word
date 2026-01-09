# -*- coding: utf-8 -*-

"""
千帆系列模型
兼容openai https://cloud.baidu.com/doc/qianfan-docs/s/1m9l6eex1
OpenAiSDK对话 https://cloud.baidu.com/doc/qianfan-docs/s/Fm9l6ocai
api_key鉴权 https://cloud.baidu.com/doc/qianfan-api/s/ym9chdsy5
api_key地址 https://console.bce.baidu.com/iam/#/iam/apikey/list
ak,sk地址 https://console.cloud.baidu-int.com/iam/intelligentCloud/aksk
模型广场 https://console.bce.baidu.com/qianfan/modelcenter/model/buildIn/list
所有模型: https://cloud.baidu.com/doc/qianfan/s/rmh4stp0j
模型计费: https://cloud.baidu.com/doc/qianfan/s/wmh4sv6ya
Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
author: xuhong02
contact: xuhong02@baidu.com
datetime: 2025/3/19 18:50
"""
import json
import os
import httpx

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

timeout = 120  # 降低到120秒，避免单个请求占用连接过久
DEFAULT_TEMPERATURE = 0.01
QF_BASE_URL = "https://qianfan.baidubce.com/v2"


def get_llm(model_name, temperature: float = None, max_retries=2, max_tokens=8192):
    """
    获取千帆模型实例（非单例模式，每次创建新实例）
    需要设置环境变量：QIANFAN_API_KEY

    Args:
        model_name: 模型名称
        temperature: 温度参数，如果为None则使用默认值
        max_retries: 最大重试次数，默认2次
        max_tokens: 最大输出token数

    Returns:
        ChatOpenAI: 千帆模型实例
    """
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    # 千帆 API 不需要代理（或从环境变量读取，如果需要的话）
    # proxy = os.getenv('QIANFAN_PROXY', None)  # 可选：如果千帆也需要代理
    
    # 获取当前对话ID作为 X-Trace-ID
    default_headers = {}

    # 创建自定义的 httpx.AsyncClient，配置适合低并发场景的小连接池
    http_async_client = httpx.AsyncClient(
        # proxy=proxy,  # 千帆通常不需要代理，如需要可取消注释
        limits=httpx.Limits(
            max_connections=20,           # 小连接池,适合低并发场景
            max_keepalive_connections=5   # 保持少量活跃连接
        ),
        timeout=httpx.Timeout(
            timeout=timeout,              # 总超时120秒
            pool=30.0                     # 等待获取连接的超时时间30秒
        )
    )
    
    # 每次创建新的 ChatOpenAI 实例，避免连接池共享导致的问题
    return ChatOpenAI(
        model=model_name,
        api_key=os.getenv('QIANFAN_API_KEY'),
        http_async_client=http_async_client,
        default_headers=default_headers,  # 使用 default_headers 传递自定义 header
        openai_proxy=None,  # 禁用 openai_proxy，因为千帆 API 不需要代理
        temperature=temp,
        # timeout 已在 http_async_client 中配置，这里不需要重复设置
        base_url=QF_BASE_URL,
        max_tokens=max_tokens,
        max_retries=max_retries
    )

if __name__ == "__main__":
    template = """问题: {question}
    回答: 让我逐步思考这个问题。"""

    prompt = PromptTemplate.from_template(template)

    question = "大模型技术中endpoint是什么意思"
    chat = get_llm('qwen3-coder-480b-a35b-instruct')
    llm_chain = prompt | chat
    ret = llm_chain.invoke({'question': question})
    print(ret)
    usage = ret.usage_metadata
    # {"input_tokens": 29, "output_tokens": 282, "total_tokens": 311}
    print(json.dumps(usage, ensure_ascii=False))
from typing import Optional

from pydantic import BaseModel, Field


class QuicklySiliconflowAiConfig(BaseModel):
    base_url: str = Field(default="https://api.siliconflow.cn/v1", description="硅基流动的请求地址")
    chat_model: str = Field(default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", description=" siliconflow的模型名称")
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-8B", description=" siliconflow的嵌入模型名称")
    key: str = Field(..., description=" siliconflow的密钥")

class QuicklyAzureAiConfig(BaseModel):
    base_url: str = Field(default="https://99313-mafhdjdl-eastus2.cognitiveservices.azure.com", description="亚马逊的请求地址")
    chat_model: str = Field(default="gpt-4.1", description="亚马逊的模型名称")
    key: str = Field(..., description="亚马逊的密钥")
    api_version: str = Field(default="2025-01-01-preview", description="微软的api版本")
    embedding_model: str = Field(default="qwen3:0.6b", description="亚马逊的嵌入模型名称")

class QuicklyOllamaAiConfig(BaseModel):
    base_url: str = Field(default="http://127.0.0.1:11434", description="ollama的请求地址")
    chat_model: str = Field(default="qwen3:0.6b", description="ollama的模型名称")
    embedding_model: str = Field(default="qwen3:0.6b", description="ollama的嵌入模型名称")
    key: str = Field(..., description="ollama的密钥")

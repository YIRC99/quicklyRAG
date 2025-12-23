import json
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger
from pydantic import BaseModel, Field

from quickly_rag.config.platform_config import MySiliconflowAiInfo


class QuicklyRerankerProvider(BaseModel):
    """
    硅基流动平台重排序模型服务提供者。
    提供对文档进行重排序的功能，使用 SiliconFlow 的 rerank API。
    """
    model: str = Field(default='Qwen/Qwen3-Reranker-0.6B', description="指定要使用的重排模型平台类型")
    base_url: str = Field(default=MySiliconflowAiInfo.base_url.rstrip('/'), description="请求的地址")
    api_key: str = Field(default=MySiliconflowAiInfo.key, description="平台的密钥")
    rerank_url: str = Field(default=MySiliconflowAiInfo.base_url.rstrip('/') + '/rerank', description="重排模型的地址")


    def rerank(self,
               query: str,
               documents: List[str],
               top_n: Optional[int] = None,
               instruction: str = "Please rerank the documents based on the query.",
               return_documents: bool = True,
               max_chunks_per_doc: Optional[int] = None,
               overlap_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        对文档进行重排序。
        
        Args:
            query: 查询语句
            documents: 待排序的文档列表
            top_n: 返回前N个结果，默认返回所有结果
            instruction: 指令文本
            return_documents: 是否返回文档内容
            max_chunks_per_doc: 每个文档的最大块数
            overlap_tokens: 重叠的token数
            
        Returns:
            重排序后的结果列表，每个元素包含索引、分数和可能的文档内容
        """
        # 准备请求数据
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "instruction": instruction,
            "return_documents": return_documents
        }
        
        # 添加可选参数
        if top_n is not None:
            payload["top_n"] = top_n
        if max_chunks_per_doc is not None:
            payload["max_chunks_per_doc"] = max_chunks_per_doc
        if overlap_tokens is not None:
            payload["overlap_tokens"] = overlap_tokens
            
        # 设置请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # 发送POST请求
            response = httpx.post(
                self.rerank_url,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            return result.get("results", [])
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred during reranking: {e}")
            logger.error(f"Response content: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error occurred during reranking: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in reranking response: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during reranking: {e}")
            raise

    async def arerank(self,
                      query: str,
                      documents: List[str],
                      top_n: Optional[int] = None,
                      instruction: str = "Please rerank the documents based on the query.",
                      return_documents: bool = True,
                      max_chunks_per_doc: Optional[int] = None,
                      overlap_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        异步对文档进行重排序。
        
        Args:
            query: 查询语句
            documents: 待排序的文档列表
            top_n: 返回前N个结果，默认返回所有结果
            instruction: 指令文本
            return_documents: 是否返回文档内容
            max_chunks_per_doc: 每个文档的最大块数
            overlap_tokens: 重叠的token数
            
        Returns:
            重排序后的结果列表，每个元素包含索引、分数和可能的文档内容
        """
        # 准备请求数据
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "instruction": instruction,
            "return_documents": return_documents
        }
        
        # 添加可选参数
        if top_n is not None:
            payload["top_n"] = top_n
        if max_chunks_per_doc is not None:
            payload["max_chunks_per_doc"] = max_chunks_per_doc
        if overlap_tokens is not None:
            payload["overlap_tokens"] = overlap_tokens
            
        # 设置请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # 发送异步POST请求
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.rerank_url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                # 检查响应状态
                response.raise_for_status()
                
                # 解析响应
                result = response.json()
                return result.get("results", [])
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred during async reranking: {e}")
            logger.error(f"Response content: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error occurred during async reranking: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in async reranking response: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during async reranking: {e}")
            raise
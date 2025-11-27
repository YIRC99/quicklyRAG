from typing import Optional

from langchain_core.embeddings import Embeddings

from quicklyRag.config.baseEnum.VectorMetricTypeEnum import VectorMetricType


class MyFaissConfig:
    def __init__(
            self,
            metric_type: Optional[VectorMetricType],
            embedding: Optional[Embeddings],
            **kwargs
    ):
        self.metric_type = metric_type
        self.embedding = embedding
        for key, value in kwargs.items():
            setattr(self, key, value)
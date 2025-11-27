from typing import Optional

from langchain_core.embeddings import Embeddings

from quicklyRag.config.baseEnum.VectorIndexTypeEnum import VectorIndexType


class MyMilvusConfig:
    def __init__(
            self,
            base_url: Optional[str],
            port: Optional[str],
            user: Optional[str],
            password: Optional[str],
            collection_name: Optional[str],
            auto_id: Optional[bool],    # 是否自动生成ID
            drop_old: Optional[bool],   # 是否删除已有集合   删除现有集合并重新创建
            metric_type: Optional[str],
            is_delete: Optional[bool],
            embedding_model: Optional[Embeddings],
            index_type: Optional[VectorIndexType],
            enable_dynamic_field: Optional[bool] = False,
            **kwargs
    ):
        self.base_url = base_url
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.auto_id = auto_id
        self.drop_old = drop_old
        self.metric_type = metric_type
        self.embedding_model = embedding_model
        self.index_type = index_type
        self.is_delete = is_delete
        self.enable_dynamic_field = enable_dynamic_field
        for key, value in kwargs.items():
            setattr(self, key, value)
from pydantic import BaseModel, Field


class RagDocumentInfo(BaseModel):
    chunk_size: int = Field(default=300, description="文档分块大小")
    chunk_overlap: int = Field(default=60, description="文档分块重叠大小")

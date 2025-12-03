from typing import Optional


class RagDocumentInfo:
    def __init__(
            self,
            chunk_size: Optional[int] = None,
            chunk_overlap: Optional[int] = None,
            **kwargs
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Dynamically set additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
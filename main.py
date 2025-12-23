from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from quickly_rag import api

app = FastAPI()


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


@app.post('/chat/stream')
def chat_stream(parms: ChatRequest):
    return StreamingResponse(
        # 一行代码接入流式对话
        api.llm_stream_chat(parms.question, parms.session_id),
        media_type="text/event-stream"
    )



if __name__ == '__main__':
    uvicorn.run('main:app', host="0.0.0.0", port=18000)
    pass
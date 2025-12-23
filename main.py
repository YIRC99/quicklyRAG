from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import quickly_rag

app = FastAPI()


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


@app.post('/chat/stream')
def chat_stream(parms: ChatRequest):
    return StreamingResponse(
        quickly_rag.llm_stream_chat(parms.question, parms.session_id),
        media_type="text/event-stream"  # 必须指定这个 header
    )

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

if __name__ == '__main__':
    uvicorn.run('main:app', host="0.0.0.0", port=18000)
    pass
from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import OllamaLLM, ChatOllama

model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)









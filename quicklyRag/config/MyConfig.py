from quicklyRag.config.baseClass.MyAzureAiConfig import MyAzureAiConfig
from quicklyRag.config.baseClass.MySiliconflowAiConfig import MySiliconflowAiConfig
from quicklyRag.config.baseClass.MyOllamaAiConfig import MyOllamaAiConfig


MySiliconflowAiInfo = MySiliconflowAiConfig()

# 创建亚马逊平台gpt的配置
MyAzureAiInfo = MyAzureAiConfig(
    base_url='https://99313-mafhdjdl-eastus2.cognitiveservices.azure.com',
    chat_model='gpt-4.1',
    key='亚马逊的Key',
    api_version="2025-01-01-preview"
)

# 创建本地Ollama的配置 可以使用默认的配置
MyOllamaInfo = MyOllamaAiConfig()

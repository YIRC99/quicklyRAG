

# 配置一下使用什么系统提示词  肯定使用文件的形式
# 配置一下是不是要添加向量检索,提问的时候要不要知识库
# 配置一下对话记忆的配置
import sqlite3
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from memori import Memori
def get_sqlite_connection():
    return sqlite3.connect("memori.db")
# 初始化 LangChain 的 LLM 客户端
llm = ChatOpenAI(
        model='Qwen/Qwen3-Omni-30B-A3B-Instruct',
        base_url='https://api.siliconflow.cn/v1',
        api_key='硅基流动的key',
    )
# 初始化 Memori 并注册 LangChain 的 LLM
mem = Memori(get_sqlite_connection).llm.register(llm)
mem.attribution(entity_id="user_001", process_id="langchain_agent")  # 设置归因
mem.config.storage.build()  # 构建数据库

# 定义 LangChain 链
prompt = PromptTemplate(
    input_variables=["question"],
    template="回答用户问题：{question}"
)
chain = prompt | llm

# 第一次交互：存储记忆
response = chain.invoke(input={"question": "我喜欢什么运动？"})
mem.augmentation.wait()  # 等待记忆增强完成

# 第二次交互：利用记忆回答
response = chain.invoke(input={"question": "我喜欢什么运动？"})
print(response)  # 应输出“你喜欢打篮球”

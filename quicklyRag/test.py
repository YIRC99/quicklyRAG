from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass

from quicklyRag.model.MyModel import siliconflow_llm, siliconflow_embedding

# checkpointer = InMemorySaver()
#
# @dataclass
# class Context:
#     """Custom runtime context schema."""
#     user_id: str

# agent = create_agent(
#     model=siliconflow_llm(),
#     system_prompt="我成功运行langchain1.0了 庆祝一下",
#     # tools=[get_user_location, get_weather_for_location],
#     context_schema=Context,
#     # response_format=ResponseFormat,
#     checkpointer=checkpointer
# )

# `thread_id` is a unique identifier for a given conversation.
# config = {"configurable": {"thread_id": "1"}}

# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "我成功运行langchain1.0了 庆祝一下"}]},
#     config=config,
#     context=Context(user_id="1")
# )

# print(response)
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# Note that we can continue the conversation using the same `thread_id`.
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "谢谢你"}]},
#     config=config,
#     context=Context(user_id="1")
# )

# print(response)
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )

if __name__ == '__main__':
    print(siliconflow_embedding().embed_query("你好请问你是"))
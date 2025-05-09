from typing import Annotated
from typing_extensions import TypedDict
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_deepseek import ChatDeepSeek
import os
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver


os.environ["DEEPSEEK_API_KEY"] = 'sk-34c6a63d778b4c328810c7ff8deb08bd'
os.environ["TAVILY_API_KEY"] = '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'

class State(TypedDict):
    messages: Annotated[list, add_messages]



graph_builder = StateGraph(State)

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# define tool
tool = TavilySearch(max_results=5)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

#define memory（保存到内存中） 实际产品中，需要替换成数据库


# def chatbot(state: State):
#     return {"messages": [llm_with_tools.invoke(state["messages"])]}
# #define memory（保存到内存中） 实际产品中，需要替换成数据库
def chatbot(state: State):
    user_question = state["messages"][-1].content
    # 启发式判断是否需要搜索，这里简单示例：检查问题中是否包含特定关键词
    search_keywords = ["最新", "实时", "当前", "最近","2025", "2024", "近期"]
    need_search = any(keyword in user_question for keyword in search_keywords)

    if need_search:
        # 调用Tavily进行搜索
        search_results = tool.invoke({"query": user_question})
        # 构建提示模板
        prompt_template = f"用户提问：{user_question}\n"
        for result in search_results["results"]:
            prompt_template += f"- 链接：{result['url']} ，标题：{result['title']} ，内容：{result['content']}\n"
        prompt_template += "请仔细梳理这些信息，提取出关键要点，针对用户问题进行准确、有条理的回答。回答时请突出重点信息，确保回答内容与搜索结果紧密相关。"
        # 将提示信息传递给模型
        new_messages = [{"role": "user", "content": prompt_template}]
        return {"messages": [llm_with_tools.invoke(new_messages)]}
    else:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

#chatbot node
graph_builder.add_node("chatbot", chatbot)

#tool node 已经封装好了，把工具传递进去即可
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

#这里表示从"chatbot"节点出发，根据tools_condition来确定边的条件和目标。（调用函数后从chatbox->tool node）
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
#tool node -->chatbox
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)
# graph = graph_builder.compile()
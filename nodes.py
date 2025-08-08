from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from knowledge_base import EmbeddingDBUtils

SYSTEM = """
你是一个膝关节外骨骼机器人领域的专业问答机器人，负责根据相关的英文文档，用中文回答用户的问题。不允许在答案中添加编造成分。
"""
QUERY = """
以下是与问题相关的论文文献片段：
{retrieved_results}

请基于上述文献内容，用中文回答用户问题：{user_question}
"""


# 本地部署
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
model_name = "/media/a822/82403B14403B0E83/Gwb/WechatRobot/Base_models/Qwen3-1.7B"


# 定义 State
class RAGState(TypedDict):
    user_input: str
    query: str
    docs: list
    answer: str
    continue_chat: str


# 用户输入节点
def get_user_input_node(state: RAGState) -> RAGState:
    user_input = input("\n👤 请输入你的问题（输入 exit 退出）：")
    if user_input.lower() in ["exit", "quit", "end", "滚"]:
        return {"continue_chat": "end"}
    return {"user_input": user_input,
            "continue_chat": "continue"}

# 节点1：提取query
def extract_query_node(state: RAGState) -> RAGState:
    rag_utils = EmbeddingDBUtils()
    query = rag_utils.query_extract(state["user_input"])
    print(query)
    return {"query": query}


# 节点2：文档检索
def document_retrieve_node(state: RAGState) -> RAGState:
    rag_utils = EmbeddingDBUtils()
    results = rag_utils.db_query(  # 使用全局工具类实例
        query=state["query"],
        top_k=3
    )
    return {"docs": results}

# 节点3：生成答案
def generate_answer_node(state: RAGState) -> RAGState:
    prompt = QUERY.format(
        retrieved_results=state["docs"],
        user_question=state["user_input"]
    )
    rag_utils = EmbeddingDBUtils()
    answer = rag_utils.single_chat(prompt)
    print(answer)
    return {"answer": answer}

# 判断节点
def check_continue_node(state: RAGState):
    user_input = input("\n👤 是否继续提问？(y/n): ")
    return {"continue_chat": "continue" if user_input.lower() == "y" else "end"}

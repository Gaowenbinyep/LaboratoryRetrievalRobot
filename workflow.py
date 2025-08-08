from langgraph.graph import StateGraph, END

from knowledge_base import EmbeddingDBUtils
from nodes import RAGState, get_user_input_node, extract_query_node, document_retrieve_node, generate_answer_node, check_continue_node




def build_workflow():
    # 创建工作流图
    graph = StateGraph(RAGState)
    
    graph.add_node("get_user_input", get_user_input_node)
    graph.add_node("extract_query", extract_query_node)
    graph.add_node("document_retrieve", document_retrieve_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("check_continue", check_continue_node)

    graph.set_entry_point("get_user_input")
    graph.add_edge("extract_query", "document_retrieve")
    graph.add_edge("document_retrieve", "generate_answer")
    graph.add_edge("generate_answer", "check_continue")

    graph.add_conditional_edges(
        "check_continue",
        lambda state: state["continue_chat"],
        {
            "continue": "get_user_input",
            "end": END
        }
    )
    graph.add_conditional_edges(
        "get_user_input",
        lambda state: state["continue_chat"],
        {
            "continue": "extract_query",
            "end": END
        }
    )
    # 构建工作流
    workflow = graph.compile()

    return workflow


if __name__ == "__main__":

    EmbeddingDBUtils.configure_llm(
        api_key="EMPTY",
        api_base="http://localhost:8888/v1",
        model_name="/media/a822/82403B14403B0E83/Gwb/WechatRobot/Base_models/Qwen3-1.7B",
        embedder_name="/media/a822/82403B14403B0E83/Gwb/RAG/embedding_model/bge-m3_ft",
        db_name="/media/a822/82403B14403B0E83/Gwb/RAG/chroma_db/bge-m3_ft"
    )
    # 执行工作流
    workflow = build_workflow()
    
    result = workflow.invoke({})
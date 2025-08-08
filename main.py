from workflow import build_workflow
from knowledge_base import EmbeddingDBUtils




def main():
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



if __name__ == "__main__":
    main()
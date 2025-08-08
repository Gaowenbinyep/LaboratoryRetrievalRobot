from openai import OpenAI, AsyncOpenAI
import json
import random
import pandas as pd
from langchain_chroma import Chroma
from FlagEmbedding import BGEM3FlagModel
from langchain.embeddings.base import Embeddings
from knowledge_base import LangChainBGE, EmbeddingDBUtils

QUERY = """
以下是与问题相关的论文文献片段：
{retrieved_results}

请基于上述文献内容，用中文回答用户问题：{user_question}
"""

# 本地部署
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
model_name = "/media/a822/82403B14403B0E83/Gwb/WechatRobot/Base_models/Qwen3-1.7B"
embedder_name = "/media/a822/82403B14403B0E83/Gwb/RAG/embedding_model/bge-m3_ft"
db_name = "/media/a822/82403B14403B0E83/Gwb/RAG/chroma_db/bge-m3_ft"

def get_score(doc_id, recall_ids, top_k):
    if doc_id in recall_ids:
        return (top_k - recall_ids.index(doc_id))/top_k
    return 0






if __name__ == "__main__":
     # 配置LLM参数（全局仅需调用一次）
    EmbeddingDBUtils.configure_llm(
        api_key=openai_api_key,
        api_base=openai_api_base,
        model_name=model_name,
        embedder_name=embedder_name,
        db_name = db_name
    )
    top_k = 3
    
    # 获取工具类实例（自动初始化Embedding/向量库/LLM）
    rag_utils = EmbeddingDBUtils()
    datas = pd.read_json(path_or_buf="./Test/QApairs.jsonl", lines=True)
    outputs = []
    for _, data in datas.iterrows():
        queries = data["query"]
        doc_id = data["id"]
        authors = data["author"][0]
        if authors == "None":
            continue
        for query in queries:
            author = authors.split(",")[random.choice([0, -1])].strip()
            query = f"{author} {query}"
            docs = rag_utils.db_query(query, top_k=top_k)
            recall_ids = [doc.id for doc in docs]
            score = get_score(doc_id=doc_id, recall_ids=recall_ids, top_k=top_k)
            outputs.append({
                "query": query,
                "doc_id": doc_id,
                "recall_ids": recall_ids,
                "score": score
            })
    pd.DataFrame(outputs).to_json("./Test/rag_output.jsonl", orient="records", lines=True, force_ascii=False)
    top_k_acc = sum([1 if o["score"] > 0 else 0 for o in outputs])/len(outputs)
    ndcg = sum([o["score"] for o in outputs])/len(outputs)
    # 计算top_k准确率和NDCG
    print(f"Top-{top_k}准确率: {top_k_acc:.4f}")
    print(f"NDCG: {ndcg:.4f}")
    


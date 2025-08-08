import pandas as pd
from knowledge_base import LangChainBGE
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

OUTPUT_FILE = "./Train/train_data.jsonl"



class TrainDataGen:
    def __init__(self):
        self.embedding = LangChainBGE(
            model_name="/media/a822/82403B14403B0E83/Gwb/RAG/embedding_model/bge-m3",
            use_fp16=True
        )
        self.db = Chroma(
            embedding_function=self.embedding,
            persist_directory="./chroma_db"
        )
    def get_content(self, id):
        return self.db.get(ids=[id])["documents"][0]
    

def gen_train_data(data_generater: TrainDataGen):
    datas = pd.read_json("./Test/rag_output.jsonl", lines=True)
    train_datas = []
    for _, data in tqdm(datas.iterrows(), total=len(datas), desc="生成训练数据"):
        query = data["query"]
        positive = data_generater.get_content(data["doc_id"])
        negatives = []
        for recall_id in data["recall_ids"]:
            if recall_id != data["doc_id"]:
                negatives.append(data_generater.get_content(recall_id))
        negative = negatives[:3]

        train_datas.append({
            "query": query,
            "pos": [positive],
            "neg": negative
        })
    return train_datas


if __name__ == "__main__":
    data_generater = TrainDataGen()
    train_datas = gen_train_data(data_generater)
    pd.DataFrame(train_datas).to_json(OUTPUT_FILE, orient="records", lines=True, force_ascii=False)
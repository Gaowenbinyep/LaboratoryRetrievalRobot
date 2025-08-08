import os
import shutil  # 新增：用于删除目录

from openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
import PyPDF2, random
from langchain_chroma import Chroma
from FlagEmbedding import BGEM3FlagModel
from langchain.embeddings.base import Embeddings

SYSTEM = """
你是一个膝关节外骨骼机器人领域的专业问答机器人，负责根据相关的英文文档，用中文回答用户的问题。不允许在答案中添加编造成分。
"""
QUERY = """
以下是与问题相关的论文文献片段：
{retrieved_results}

请基于上述文献内容，用中文回答用户问题：{user_question}
"""


class LangChainBGE(Embeddings):
    def __init__(self, model_name='BAAI/bge-m3', device='cuda', use_fp16=True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, query_max_length=8192, show_progress=False)
        self.device = device

    def embed_documents(self, documents):
        outputs = self.model.encode(documents, batch_size=12, max_length=8192)
        dense_vecs = outputs['dense_vecs']
        return dense_vecs.tolist() if hasattr(dense_vecs, 'tolist') else dense_vecs

    def embed_query(self, query):
        outputs = self.model.encode([query], batch_size=1, max_length=8192)
        dense_vec = outputs['dense_vecs'][0]
        return dense_vec.tolist() if hasattr(dense_vec, 'tolist') else dense_vec

def document_extract(file_path):
    loader = UnstructuredPDFLoader(file_path, strategy="fast")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "."]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

def db_build(folder_path):
    embedding = LangChainBGE(
        model_name="/media/a822/82403B14403B0E83/Gwb/RAG/embedding_model/bge-m3",
        use_fp16=True
    )
    db = Chroma(
        embedding_function=embedding,
        persist_directory="./chroma_db/bge-m3"
    )
    
    all_valid_docs = []
    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            file_path = os.path.join(folder_path, file)
            documents = document_extract(file_path)
            valid_docs = [doc for doc in documents if doc.page_content.strip()]
            
            for i, doc in enumerate(valid_docs):
                doc.metadata['source'] = file
                doc.metadata['page_id'] = i + 1
                doc.metadata['type'] = 'abstract' if i == 0 else 'content'
            
            if valid_docs:
                all_valid_docs.extend(valid_docs)  # 暂存文档块
                print(f"已缓存 {file} 的 {len(valid_docs)} 个非空文档块")
            else:
                print(f"⚠️ {file} 无有效文档块，已跳过")
    
    # 一次性嵌入并添加所有文档（提升GPU批量处理效率）
    if all_valid_docs:
        db.add_documents(all_valid_docs)  # 此处会触发批量嵌入
        print(f"数据库构建完成，共添加 {len(all_valid_docs)} 个文档块")
    else:
        print("数据库构建完成，无有效文档块")

def db_reencode(existing_db_path, new_embedding_model_path, batch_size=24):
    """
    用新 embedding 模型重新对 Chroma 中的文档进行编码（不修改 page_content 和 metadata，只更新 embedding）
    """
    new_embedding = LangChainBGE(
        model_name=new_embedding_model_path,
        use_fp16=True
    )

    db = Chroma(
        embedding_function=new_embedding,
        persist_directory=existing_db_path
    )

    all_docs = db.get()
    if not all_docs["ids"]:
        print("⚠️ 向量库为空，无需重新编码")
        return

    doc_ids = all_docs["ids"]
    updated_docs = [
        Document(
            page_content=text,
            metadata=meta
        ) for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    
    db.update_documents(
        ids=doc_ids,
        documents=updated_docs
    )
    print(f"✅ 向量库重新编码完成，共更新 {len(updated_docs)} 个文档")

class EmbeddingDBUtils:
    """Embedding模型、向量库与LLM工具类（单例模式）"""
    _instance = None
    _embedding = None
    _db = None
    _llm_client = None
    _model_name = None

    # 新增：类级方法用于配置LLM参数（需在首次实例化前调用）
    @classmethod
    def configure_llm(cls, api_key, api_base, model_name, embedder_name, db_name):
        cls._api_key = api_key
        cls._api_base = api_base
        cls._model_name = model_name
        cls._embedder_name = embedder_name
        cls._db_name = db_name

    def __new__(cls):
        """单例模式：确保全局仅一个实例"""
        if cls._instance is None:
            if not hasattr(cls, '_api_key'):
                raise ValueError("请先调用 configure_llm 设置LLM参数")
            
            cls._instance = super().__new__(cls)
            # 1. 初始化Embedding（仅首次实例化时执行）
            cls._embedding = LangChainBGE(
                model_name=cls._embedder_name,
                use_fp16=True
            )
            # 2. 初始化向量库（仅首次实例化时执行）
            cls._db = Chroma(
                embedding_function=cls._embedding,
                persist_directory=cls._db_name
            )
            # 3. 新增：初始化LLM客户端（OpenAI）
            cls._llm_client = OpenAI(
                api_key=cls._api_key,
                base_url=cls._api_base
            )
        return cls._instance

    @property
    def embedding(self):
        return self._embedding

    @property
    def db(self):
        return self._db

    def db_query(self, query, top_k=3):
        return self.db.similarity_search(
            query=query,
            k=top_k,
            filter={"type": "abstract"}
        )
    
    def query_extract(self, query):
        messages = [
            {"role": "system", "content": "你是知识检索助手，需从用户输入中提取检索用户检索的作者和关键词"},
            {"role": "user", "content": f"""
             用户输入：
             {query}
             仅返回检索作者（可选）和关键词，不添加额外内容，作者和不同关键词用空格隔开。
             
             示例：
             用户输入：
             
             给我找Federica Aprigliano的外骨骼机器人的步态稳定性控制策略

             输出：
             Federica Aprigliano 外骨骼机器人 步态稳定性控制策略
             """}
        ]
        response = self._llm_client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=256,
            temperature=0.8,
            top_p=0.95,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
        )
        return response.choices[0].message.content

    def single_chat(self, query, system_prompt=SYSTEM):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        response = self._llm_client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
        )
        return response.choices[0].message.content

def db_random_sample(n=1, filter=None):
    """从向量库随机抽取n条文档内容"""
    # 初始化与向量库相同的embedding
    embedding = LangChainBGE(
        model_name="/media/a822/82403B14403B0E83/Gwb/RAG/embedding_model/bge-m3",
        use_fp16=True
    )
    
    # 连接向量库
    db = Chroma(
        embedding_function=embedding,
        persist_directory="./chroma_db"
    )
    
    # 获取符合条件的所有文档（支持类型过滤）
    all_docs = db.get(where=filter)  # 返回格式: {"ids": [], "documents": [], "metadatas": []}
    
    if not all_docs["documents"]:
        return []  # 向量库为空时返回空列表
    
    # 随机抽取n条文档（避免样本量超过文档总数）
    
    sampled_docs = random.sample(
        list(zip(all_docs["ids"], all_docs["documents"])), 
        min(n, len(all_docs["ids"]))  # 取n和文档总数的较小值
    )
    
    return sampled_docs

def db_delete(db_path, force=False):
    """
    删除指定路径的Chroma向量库（增强版，处理残留文件）
    :param db_path: 向量库存储目录（如./chroma_db/bge-m3）
    :param force: 是否强制删除（默认False，需手动确认）
    """
    if not os.path.exists(db_path):
        print(f"⚠️ 向量库路径不存在：{db_path}")
        return
    
    # 安全确认
    if not force:
        confirm = input(f"⚠️ 确定要删除向量库 '{db_path}' 吗？(输入 'yes' 确认)：")
        if confirm.lower() != 'yes':
            print("❌ 删除已取消")
            return
    
    # 尝试1：使用shutil.rmtree（基础删除）
    try:
        shutil.rmtree(db_path)
        print(f"✅ 向量库 '{db_path}' 已成功删除")
        return
    except Exception as e:
        print(f"⚠️ shutil删除失败，尝试系统命令强制删除：{str(e)}")
    
    # 尝试2：使用系统命令强制删除（适用于Linux/macOS，处理残留文件）
    try:
        # 调用rm -rf强制删除（注意：此命令不可逆！）
        subprocess.run(
            ["rm", "-rf", db_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✅ 系统命令强制删除成功：{db_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 系统命令删除失败：{e.stderr}")
    except Exception as e:
        print(f"❌ 强制删除异常：{str(e)}")

if __name__ == "__main__":
    # db_build("/media/a822/82403B14403B0E83/Gwb/RAG/Papers")
    # db_delete("/media/a822/82403B14403B0E83/Gwb/RAG/chroma_db/bge-m3_ft")
    db_reencode("/media/a822/82403B14403B0E83/Gwb/RAG/chroma_db/bge-m3_ft", "/media/a822/82403B14403B0E83/Gwb/RAG/embedding_model/bge-m3_ft")
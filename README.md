# 实验室文献检索助手

## 项目简介
本项目是一个实验室内部使用的文献检索与问答助手，基于检索增强生成（RAG）技术，专注于实验室相关的文献。系统能够自动化处理文献、构建知识库，并根据用户问题返回基于文献内容的精准中文回答。

## 核心功能
- **文献处理**：支持PDF文献去重、自动分块与清洗
- **知识库构建**：基于Chroma向量数据库存储文献嵌入向量
- **智能检索**：使用BGE-M3多模态嵌入模型实现高效语义检索
- **问答生成**：结合Qwen3-1.7B大模型生成基于文献的中文回答
- **交互式对话**：命令行交互式问答界面，支持多轮对话
- **模型调优**：支持嵌入模型与大语言模型微调（新增）
- **测试评估**：提供检索准确率与问答质量评估工具（新增）

## 环境要求
- Python 3.8+
- 依赖库：langchain、langgraph、chromadb、FlagEmbedding、PyPDF2、vllm、tqdm、pandas、numpy等（补充测试与微调依赖）
- 硬件要求：建议GPU支持（显存≥10GB，用于模型部署与向量计算）

## 项目结构
```
RAG/
├── Papers/               # 存放PDF文献（需手动放入）
│   └── document deduplication.py  # 文献去重工具
├── embedding_model/      # 嵌入模型（BGE-M3 / BGE-M3_ft）
├── chroma_db/            # Chroma向量数据库
├── logs/                 # 日志文件（模型下载、部署等）
├── nodes.py              # 工作流节点定义（用户输入、检索、生成等）
├── workflow.py           # LangGraph工作流控制
├── rag.py                # LLM调用与检索逻辑
├── knowledge_base.py     # 文献处理与知识库构建
├── main.py               # 程序入口
├── model_deploy.sh       # LLM模型部署脚本
└── embedding_model_download.sh  # 嵌入模型下载脚本
```
## 安装部署

### 1. 项目克隆
<!-- ```bash -->
#### 克隆项目到本地
git clone https://github.com/Gaowenbinyep/LaboratoryRetrievalRobot.git

#### 模型下载：
    嵌入模型（BGE-M3）：
    # 后台下载BGE-M3嵌入模型（已配置脚本）
    sh embedding_model_download.sh
    # 查看下载进度
    tail -f logs/download_progress.log

    大语言模型（Qwen3-1.7B）:
    确保模型文件已放置于指定路径：
    model_path

#### 启动本地LLM服务：
    # 部署Qwen3-1.7B模型（使用vllm）
    sh model_deploy.sh
    # 查看模型服务日志
    tail -f logs/model_output.log


### 2. 模型微调（可选）
    <!-- ```bash -->
    # 嵌入模型微调（基于训练数据）
    python model_finetune.py --model_type embedding --data_path Train/embedding_train_data.jsonl

    # 大语言模型微调（需额外配置训练参数）
    python model_finetune.py --model_type llm --data_path Train/llm_train_data.jsonl


### 3. 知识库构建：
    # 准备文献：将PDF文献放入Papers/目录
    # （可选）文献去重
    python Papers/document_deduplication.py

    # 构建知识库（处理文献并写入向量数据库）
    python knowledge_base.py  # 需确保knowledge_base.py中db_build函数被调用

### 4. 使用说明
    启动问答助手：
    python main.py

    交互流程
    1. 程序启动后，输入问题（例如："找膝关节外骨骼机器人的步态辅助机制的文章，给我列出题目、作者和主要内容"）
    2. 系统自动检索相关文献并生成回答
    3. 回答完成后，可选择继续提问（输入y）或退出（输入n）
    4. 输入"exit"、"quit"或"end"可直接退出程序

### 5. 测试评估：
    # 生成QA测试对
    python QApairs_gen.py

    # 运行检索与问答评估
    python test.py

## 注意事项
1. 文献管理：建议定期备份Papers/目录下的文献，去重操作不可逆
2. 模型资源：BGE-M3模型约占用5GB存储空间，Qwen3-1.7B模型约占用8GB
3. 性能优化：检索结果数量（top_k）可在document_retrieve_node函数中调整（默认3）
4. 使用限制：本工具仅用于实验室内部学术研究，禁止用于商业用途

## 技术栈
    检索增强生成：LangChain + LangGraph
    嵌入模型：BGE-M3（支持 dense/sparse/multi-vector 检索）
    向量数据库：Chroma
    大语言模型：Qwen3-1.7B（本地部署）
    文献处理：Unstructured + RecursiveCharacterTextSplitter

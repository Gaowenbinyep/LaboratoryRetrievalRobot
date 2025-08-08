import pandas as pd
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from flagai.trainer import Trainer
from flagai.data.dataset import Dataset
from FlagEmbedding import BGEM3FlagModel
# ... existing code ...
from knowledge_base import LangChainBGE, EmbeddingDBUtils

# 添加微调相关参数
TRAIN_DATA_PATH = "./Train/train_data.jsonl"  # 微调数据集路径
FINETUNED_MODEL_PATH = "./fine_tuned_bge_m3"  # 微调后模型保存路径
TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# 新增：定义微调数据集类
class BGEFinetuneDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(data_path, lines=True).to_dict('records')
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negative": item.get("negative", [])
        }

# 新增：模型微调函数
def finetune_bge_model():
    # 加载训练数据
    train_dataset = BGEFinetuneDataset(TRAIN_DATA_PATH)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    
    # 加载基础模型
    model = BGEM3FlagModel(
        model_name_or_path="BAAI/bge-m3",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fp16=True
    )
    
    # 设置训练参数
    trainer = Trainer(
        model=model,
        train_dataset=train_loader,
        epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=TRAIN_BATCH_SIZE,
        log_interval=10,
        save_dir=FINETUNED_MODEL_PATH
    )
    
    # 开始训练
    print(f"开始微调BGE-M3模型，共{TRAIN_EPOCHS}个epoch...")
    trainer.train()
    
    # 保存最终模型
    model.save_pretrained(FINETUNED_MODEL_PATH)
    print(f"模型微调完成，已保存至{FINETUNED_MODEL_PATH}")

if __name__ == "__main__":
    # 新增：解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true", help="是否执行模型微调")
    args = parser.parse_args()
    
    # 新增：如果指定了微调参数，则执行微调
    if args.finetune:
        finetune_bge_model()
        exit()
import pandas as pd
import os
import json
import re
from tqdm import tqdm
from openai import OpenAI
from knowledge_base import db_random_sample


os.environ["DASHSCOPE_API_KEY"] = "sk-8be01996d47b4475810abaa3f0d2cb8e"

# ======================================
# 配置区
MODEL = "qwen-plus"  # 替换成可用的模型名
OUTPUT_FILE = "/media/a822/82403B14403B0E83/Gwb/RAG/Test/QApairs.jsonl"
# ======================================


def call_qwen(prompt):
    """ 调用 Qwen3-325B 返回结果 """
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=prompt,
        extra_body={"enable_thinking": False},
    )
    response = completion.model_dump_json()
    score = json.loads(response)["choices"][0]["message"]["content"]
    return score


def build_prompt(query):
    prompt = [{
                "role": "system",
                "content": f"""你是外骨骼控制领域的文献检索专家，专为实验室研究人员设计检索query。
                    任务：根据给定的外骨骼控制领域文献文本，生成3个专业、精准的检索query，并给出文章的作者，用于RAG系统查找相关学术文献。
                    要求：
                    1. 必须基于文本核心内容，聚焦外骨骼控制技术关键点（如控制算法、人机交互、动力学建模、运动意图识别等）
                    2. 使用领域专业术语（如"外骨骼机器人""肌电信号控制""阻抗控制""协同运动""穿戴舒适性"等）
                    3. 符合科研人员检索习惯，可包含组合关键词（如"外骨骼机器人 自适应控制 中风康复"）
                    4. 避免过于宽泛（如仅"外骨骼"）或过于狭窄（如包含具体实验数据）的表述
                    5. 每条query独立聚焦文本不同维度，形成检索策略组合
                    """
                },
            {
                "role": "user",
                "content": f"""<文献文本>
                {query}
                </文献文本>

                请输出3条检索query，作者，按如下格式返回，示例：
                <query>
                    髋关节外骨骼 自适应阻抗控制
                    肌电信号 运动意图识别
                    IMU传感器 膝关节外骨骼控制
                </query>
                <author>
                    所有作者的名称
                </author>
                请严格按照上述格式输出，不允许输出其他内容。
                """
            }]
    return prompt

if __name__ == "__main__":
    datas = db_random_sample(600, {"type": "abstract"})
    lines = []
    for data in tqdm(datas):
        prompt = build_prompt(data[1])
        
        res = call_qwen(prompt)
        pattern = re.compile(r'<query>(.*?)</query>', re.DOTALL)
        pattern_author = re.compile(r'<author>(.*?)</author>', re.DOTALL)
        matches = pattern.findall(res)
        author = pattern_author.findall(res)
        if not matches:
            for i in range(10):
                res = call_qwen(prompt)
                pattern = re.compile(r'<query>(.*?)</query>', re.DOTALL)
                matches = pattern.findall(res)
                if matches:
                    break
        if matches:
            if not author:
                author = "None"
            query = matches[0].strip().split("\n")
            query = [q.strip() for q in query]
            author = author[0].strip()
            lines.append({
                "query": query,
                "author": author,
                "content": data[1],
                "id": data[0],
            })
    pd.DataFrame(lines).to_json(OUTPUT_FILE, orient="records", lines=True, force_ascii=False)
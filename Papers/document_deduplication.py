import os
import shutil
from tqdm import tqdm
from PyPDF2 import PdfReader
from collections import defaultdict

def get_pdf_files():
    """获取当前目录下所有PDF文件并按大小排序"""
    pdf_files = []
    for file in os.listdir('.'):
        if file.lower().endswith('.pdf') and os.path.isfile(file):
            file_size = os.path.getsize(file)
            pdf_files.append((file, file_size))
    
    # 按文件大小升序排序
    pdf_files.sort(key=lambda x: x[1])
    return [file for file, _ in pdf_files]

def get_pdf_first_100_chars(pdf_path):
    """提取PDF文件的前100个字符"""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            if len(reader.pages) == 0:
                return ""
            
            page = reader.pages[0]
            text = page.extract_text()
            # 提取前100个非空白字符
            clean_text = ''.join(text.split())[:100]
            return clean_text
    except Exception as e:
        print(f"读取PDF错误 {pdf_path}: {str(e)}")
        return ""

def remove_duplicate_pdfs():
    """检测并删除重复的PDF文件"""
    pdf_files = get_pdf_files()
    if not pdf_files:
        print("未找到PDF文件")
        return
    
    # 存储内容哈希与对应的文件路径
    content_map = defaultdict(list)
    
    # 修改：添加进度条显示处理进度
    for pdf_file in tqdm(pdf_files, desc="处理PDF文件", unit="个"):
        content = get_pdf_first_100_chars(pdf_file)
        if content:  # 只处理能成功读取内容的文件
            content_map[content].append(pdf_file)
    
    # 处理重复文件
    for content, files in content_map.items():
        if len(files) > 1:
            print(f"\n发现重复内容: {content}")
            print(f"涉及文件 ({len(files)}个): {files}")
            
            # 保留第一个(最小的)，删除其余
            keep_file = files[0]
            delete_files = files[1:]
            
            print(f"保留文件: {keep_file}")
            for delete_file in delete_files:
                confirm = input(f"确定要删除 {delete_file} 吗? (y/n): ")
                if confirm.lower() == 'y':
                    try:
                        os.remove(delete_file)
                        print(f"已删除: {delete_file}")
                    except Exception as e:
                        print(f"删除失败 {delete_file}: {str(e)}")
                else:
                    print(f"已取消删除: {delete_file}")

if __name__ == "__main__":
    print("=== PDF文件去重工具 ===")
    print("注意: 此操作不可逆，请确保已备份重要文件！")
    confirm = input("是否继续? (y/n): ")
    if confirm.lower() == 'y':
        remove_duplicate_pdfs()
        print("\n操作完成")
    else:
        print("已取消操作")
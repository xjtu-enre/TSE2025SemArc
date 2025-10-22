import os
import json
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration,AutoTokenizer




tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base-multi-sum")

# 读取代码文件内容
def read_code(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 根据token长度裁剪文本
def chunk_code(code, max_tokens=256):
    tokens = tokenizer.tokenize(code)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])
        # 控制最大字符长度（避免生成时再超出）
        if len(chunk.strip()) > 10:  # 防止空内容生成
            chunks.append(chunk.strip())
    return chunks

# 使用 CodeT5+ 生成摘要
def summarize_code(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    summary_ids = model.generate(inputs["input_ids"], max_length=64)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 主流程：遍历路径下所有目标语言文件，生成摘要
def summarize_project_code(project_path, output_json_path,lang):
    if lang=='c':
        FILE_EXTENSIONS = [".c", ".cpp", ".h",".cc"]  # 支持多个扩展名，例如 C 和 C++ 文件
    elif lang=='java':
        FILE_EXTENSIONS = [".java"]
    elif lang=='python':
        FILE_EXTENSIONS = [".py"]
    summary_list = []
    total_files = sum(len(files) for _, _, files in os.walk(project_path))  # 计算总文件数
    analyzed_count = 0  # 已分析文件计数

    for root, _, files in os.walk(project_path):
        for file in files:
            if any(file.endswith(ext) for ext in FILE_EXTENSIONS):  # 检查文件扩展名是否匹配
                analyzed_count += 1
                print(f"正在分析：{file} ({analyzed_count}/{total_files})")  # 输出进度信息

                file_path = os.path.join(root, file)
                code = read_code(file_path)
                chunks = chunk_code(code)
                summaries = [summarize_code(chunk) for chunk in chunks]
                # 聚合摘要（简单拼接或去重后拼接）
                unique_summary = ". ".join(sorted(set(summaries)))
                summary_list.append({
                    "file": os.path.relpath(file_path, project_path),
                    "Functionality": unique_summary
                })

    result = {"summary": summary_list}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"摘要保存至：{output_json_path}")

# ==== 用法示例 ====
if __name__ == "__main__":
    # start_time = time.time()  # 记录开始时间
    
    summarize_project_code("D:\\SemArc\\data\\hadoop", "D:\\SemArc\\hadoop_code_summary.json",lang='java')
    summarize_project_code("D:\\SemArc\\data\\oodt", "D:\\SemArc\\oodt_code_summary.json",lang='java')
    summarize_project_code("D:\\SemArc\\data\\teammates", "D:\\SemArc\\teammates_code_summary.json",lang='java')
    summarize_project_code("D:\\SemArc\\data\\chromium", "D:\\SemArc\\chromium_code_summary.json",lang='c')

    # end_time = time.time()  # 记录结束时间
    # elapsed_time = end_time - start_time  # 计算运行时间
    # print(f"程序运行时间：{elapsed_time:.2f} 秒")
import tiktoken
import os

# 选择模型（以 gpt-3.5-turbo 为例）
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# 示例：统计整个目录下的 token 数量
root_dir = "your_project_source_dir"
token_stats = []

for root, _, files in os.walk(root_dir):
    for f in files:
        if f.endswith(".py") or f.endswith(".java") or f.endswith(".cpp"):  # 依语言过滤
            file_path = os.path.join(root, f)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                content = infile.read()
                token_count = count_tokens(content)
                token_stats.append((file_path, token_count))

# 输出统计信息
for path, count in token_stats:
    print(f"{path}: {count} tokens")

# 汇总
total_tokens = sum(c for _, c in token_stats)
avg_tokens = total_tokens / len(token_stats)
print(f"\nTotal files: {len(token_stats)}")
print(f"Total tokens: {total_tokens}")
print(f"Average tokens per file: {avg_tokens:.2f}")

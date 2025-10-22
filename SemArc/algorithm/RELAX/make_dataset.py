import os

def create_structure_from_tree(tree_str, root_path):
    stack = [root_path]
    prev_indent = 0
    for line in tree_str.strip().splitlines():
        # 计算缩进层级（每一级通常为4个空格或一个特殊符号）
        raw = line
        stripped = line.lstrip(' │├└─')
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        # 处理栈，找到正确的父目录
        while indent < prev_indent:
            stack.pop()
            prev_indent -= 4  # 假设每一级缩进为4
        if indent > prev_indent:
            prev_indent = indent
        # 当前目录
        parent = stack[-1]
        parts = stripped.split('/')
        name = parts[0]
        path = os.path.join(parent, name)
        if '.' in name:  # 文件
            os.makedirs(parent, exist_ok=True)
            if not os.path.exists(path):
                with open(path, 'w', encoding='utf-8') as f:
                    pass
        else:  # 目录
            os.makedirs(path, exist_ok=True)
            stack.append(path)
            prev_indent = indent + 4  # 进入下一级

if __name__ == "__main__":
    tree = """
concerns/
├── Global/
│   └── global_terms.txt
└── Pipeline/
    └── pipeline_terms.txt
    """
    target_root = r"d:\RELAX\concerns"
    create_structure_from_tree(tree, os.path.dirname(target_root))
    print("目录结构已创建。")
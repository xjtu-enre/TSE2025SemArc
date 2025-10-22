# -*- coding: utf-8 -*-
"""
RELAX 源码实体抽取与特征提取模块
"""
import os

def extract_code_entities(source_root, file_exts=['.java', '.py', '.cpp', '.c', '.h', '.cc']):
    """
    扫描源码目录，抽取所有代码文件实体
    :param source_root: 代码根目录
    :param file_exts: 支持的文件后缀
    :return: [(实体全路径, 文件内容)] 列表
    """
    entities = []
    for root, dirs, files in os.walk(source_root):
        for fname in files:
            if any(fname.endswith(ext) for ext in file_exts):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        entities.append((fpath, content))
                except Exception as e:
                    print("读取失败: ", fpath, e)
    return entities

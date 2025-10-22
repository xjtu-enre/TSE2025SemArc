# -*- coding: utf-8 -*-
"""
RELAX Concern-Oriented Architecture Recovery 主流程
用法示例:
    python main.py ./data/bash
输出:
    bash_RELAX.json
"""
import os, sys, json
from classifier import RelaxClassifier
from extractor import extract_code_entities
from cluster import assign_clusters

def main(proj_path: str):
    # ========== 1. 加载 / 训练分类器 ==========
    clf = RelaxClassifier()
    model_path = "./relax_classifier.pkl"
    try:
        clf.load(model_path)
        print("已加载训练好的分类器")
    except FileNotFoundError:
        clf.train(base_dir="./concerns/", save_path=model_path)

    # ========== 2. 扫描源码 ==========
    entities = extract_code_entities(proj_path)
    print(f"共发现 {len(entities)} 个代码实体")

    # ========== 3. 亲和力预测 ==========
    texts   = [txt for _, txt in entities]
    probas  = clf.predict_proba(texts)

    # ========== 4. 聚类分配 ==========
    labels  = assign_clusters(probas, clf.concerns)

    # ========== 5. 整理 {Concern: [paths]} ==========
    cluster_map = {}
    for (abs_path, _), label in zip(entities, labels):
        rel_path = os.path.relpath(abs_path, proj_path)
        cluster_map.setdefault(label, []).append(rel_path)

    # ========== 6. 组装 JSON ==========
    structure = [
        {
            "@type": "group",
            "name": grp,
            "nested": [{"@type": "item", "name": p} for p in sorted(files)]
        }
        for grp, files in sorted(cluster_map.items())
    ]
    result = {
        "@schemaVersion": "1.0",
        "name": "clustering",
        "structure": structure
    }

    # ========== 7. 保存 ==========
    proj_name = os.path.basename(os.path.abspath(proj_path))
    out_file  = f"{proj_name}_RELAX.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    print(f"聚类结果已保存至 {out_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python main.py <project_path>")
        sys.exit(1)
    project_path = sys.argv[1]
    if not os.path.isdir(project_path):
        print(f"路径无效: {project_path}")
        sys.exit(1)
    main(project_path)

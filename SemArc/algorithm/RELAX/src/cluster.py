# -*- coding: utf-8 -*-
"""
RELAX concern聚类分配模块
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def assign_clusters(probas, concerns, threshold=0.1):
    """
    将每个实体分配到最相似concern聚类
    :param probas: [n_entities, n_concerns] 亲和力矩阵
    :param concerns: 关注点名称列表
    :param threshold: 分配到“Unknown”聚类的阈值
    :return: 每个实体分配的concern聚类名称
    """
    clusters = []
    cluster_vectors = np.eye(len(concerns))  # 单位向量
    unknown_vector = np.zeros(len(concerns)).reshape(1, -1)
    for i, p in enumerate(probas):
        sims = cosine_similarity([p], cluster_vectors)[0]
        max_idx = np.argmax(sims)
        max_sim = sims[max_idx]
        # 如果所有亲和力都很低，或相似度过低，则分到“Unknown”
        if max_sim < threshold:
            clusters.append("Unknown")
        else:
            clusters.append(concerns[max_idx])
    return clusters

# -*- coding: utf-8 -*-
"""
RELAX方法文本分类器训练与预测模块
使用多类别朴素贝叶斯，支持训练和持久化
"""
import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

class RelaxClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.model = MultinomialNB()
        self.concerns = []

    def load_training_data(self, base_dir):
        """从concerns目录加载训练数据"""
        texts, labels = [], []
        concerns = []
        for concern in os.listdir(base_dir):
            concern_path = os.path.join(base_dir, concern)
            if not os.path.isdir(concern_path): continue
            concerns.append(concern)
            for fname in os.listdir(concern_path):
                with open(os.path.join(concern_path, fname), encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
                    labels.append(concern)
        self.concerns = concerns
        return texts, labels

    def train(self, base_dir, save_path=None):
        """训练多类别文本分类器"""
        texts, labels = self.load_training_data(base_dir)
        X = self.vectorizer.fit_transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("训练完成，测试集准确率: {:.2f}".format(acc))
        print("混淆矩阵:\n", confusion_matrix(y_test, y_pred, labels=self.concerns))
        if save_path:
            joblib.dump((self.model, self.vectorizer, self.concerns), save_path)
            print("模型已保存至: ", save_path)

    def load(self, path):
        """加载已训练模型"""
        self.model, self.vectorizer, self.concerns = joblib.load(path)

    def predict_proba(self, texts):
        """预测每个文本对各 concern 的概率（亲和力向量）"""
        X = self.vectorizer.transform(texts)
        probas = self.model.predict_proba(X)
        return probas  # shape: [n_samples, n_concerns]

if __name__ == "__main__":
    import argparse, pathlib

    parser = argparse.ArgumentParser(
        description="Train RELAX multi-class classifier and save to disk")
    parser.add_argument(
        "--data", "-d", default="./concerns",
        help="训练语料目录（concern 子目录结构）")
    parser.add_argument(
        "--out", "-o", default="relax_classifier.pkl",
        help="输出模型文件名")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"训练数据目录不存在: {data_dir}")

    clf = RelaxClassifier()
    clf.train(base_dir=str(data_dir), save_path=args.out)

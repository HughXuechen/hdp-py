import os
import numpy as np
import pandas as pd
from hdp_py import HDP
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm

# 确保结果文件夹存
results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# 1. 读取数据
def read_documents(folder_path):
    documents = []
    file_names = []
    for filename in tqdm(os.listdir(folder_path), desc="Reading documents"):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
                file_names.append(filename)
    return documents, file_names

folder_path = 'demo-data'
documents, file_names = read_documents(folder_path)

# 2. 文本向量化
print("Vectorizing documents...")
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()

# 3. 准备HDP输入
x = X.toarray().astype(int)
j = np.arange(len(documents))

# 4. 初始化和运行HDP模型
print("Running HDP model...")
hdp = HDP.HDP(gamma=0.5, alpha0=0.5, f='multinomial', hypers=(len(vocab), np.full(len(vocab), 0.5)))
hdp = hdp.gibbs_direct(x, j, iters=100, Kmax=20, verbose=True)

# 5. 获取主题-词分布
beta = hdp.beta_samples[-1]

# 6. 获取文档-主题分布
theta = hdp.direct_samples[-1]['theta']

# 7. 输出结果
def print_topics(beta, vocab, n_words=10):
    topics_output = []
    for i, topic in enumerate(beta):
        top_words = [vocab[j] for j in topic.argsort()[:-n_words-1:-1]]
        topics_output.append(f"Topic {i}: {', '.join(top_words)}")
    return topics_output

def get_representative_docs(theta, file_names, n_docs=3):
    docs_output = []
    for i in range(theta.shape[1]):
        topic_probs = theta[:, i]
        top_docs = np.argsort(topic_probs)[-n_docs:][::-1]
        docs_output.append(f"Topic {i} representative documents:")
        for doc_idx in top_docs:
            docs_output.append(f"  - {file_names[doc_idx]} (probability: {topic_probs[doc_idx]:.4f})")
    return docs_output

def generate_wordcloud(beta, vocab, topic_idx):
    word_freq = {vocab[i]: beta[topic_idx][i] for i in range(len(vocab))}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Topic {topic_idx}')
    plt.savefig(os.path.join(results_folder, f'topic_{topic_idx}_wordcloud.png'))
    plt.close()

# 保存主题和文档信息到文件
with open(os.path.join(results_folder, 'topics_and_docs.txt'), 'w', encoding='utf-8') as f:
    f.write("Top words for each topic:\n")
    topics_output = print_topics(beta, vocab)
    f.write("\n".join(topics_output) + "\n\n")
    
    f.write("Representative documents for each topic:\n")
    docs_output = get_representative_docs(theta, file_names)
    f.write("\n".join(docs_output) + "\n")

print("\nGenerating word clouds for each topic...")
for i in tqdm(range(beta.shape[0]), desc="Generating word clouds"):
    generate_wordcloud(beta, vocab, i)

print("All results have been saved in the 'results' folder.")

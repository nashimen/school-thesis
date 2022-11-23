import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True
debugLength = 30


# 读取词向量
def load_w2v(words):
    t1 = time.time()
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer('/testcbd021_zhangjunming/dataset/BERT/distiluse-base-multilingual-cased-v1')
    print("加载w2v文件耗时：", (time.time() - t1) / 60.0, "minutes")

    embeddings = []
    for word in words:
        embeddings.append(model.encode(word, convert_to_tensor=False))
        # embeddings.append(model.encode(word, convert_to_numpy=True))

    print("w2v loading finished...")

    return embeddings


# 输入candidate和seed words的词向量，找到最相似的seed word的下标&相似度
def find_most_similar_seed(current_candidate, embeddings):
    i = 0
    similarity_max = 0
    for j in range(len(embeddings)):
        sim = cosine_similarity([current_candidate], [embeddings[j]])[0][0]
        if sim > similarity_max:
            i = j
            similarity_max = sim

    return i, similarity_max


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取seed words
    seed_path = "data/2 Seed words.xlsx"
    seed_words = pd.read_excel(seed_path, engine="openpyxl")["Words"]
    print("seed_words:", seed_words)

    # 读取candidates
    candidates_path = "test/2 candidates-test.txt" if debug else "result/1 candidates.txt"
    file = open(candidates_path, 'r', encoding='utf-8')
    candidates = eval(file.read())
    file.close()
    print("candidates:", candidates)

    # Transformer词向量表示
    embeddings_seed_words = load_w2v(seed_words)
    embeddings_candidates = load_w2v(candidates)

    # 相似度计算,遍历每一个candidate与所有seed words的相似度
    length = len(candidates)
    similar_words = []
    similarities = []
    for i in range(length):
        return_value = find_most_similar_seed(embeddings_candidates[i], embeddings_seed_words)
        index = return_value[0]
        similarity = return_value[1]
        seed_word = seed_words[index]
        similar_words.append(seed_word)
        similarities.append(similarity)
        if i % 300 == 0:
            print("正在计算第", i, "个candidate的seed word")

    # 保存最相似的seed word & 相似度
    result = {"Candidates": candidates, "Seed word": similar_words, "Similarity": similarities}
    result = pd.DataFrame(result)

    save_path = "test/3 Kansei-words-test.xlsx" if debug else "result/2 Kansei-words.xlsx"
    result.to_excel(save_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


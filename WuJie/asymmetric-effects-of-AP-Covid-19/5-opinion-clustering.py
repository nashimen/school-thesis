import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True
debugLength = 100


# 读取词向量
def load_w2v(words):
    t1 = time.time()
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer('/testcbd021_zhangjunming/dataset/BERT/distiluse-base-multilingual-cased-v1')
    print("加载w2v文件耗时：", (time.time() - t1) / 60.0, "minutes")

    embeddings = []
    for word in words:
        embeddings.append(model.encode(word, convert_to_numpy=True))

    return embeddings


# 作图
def drawing(data, label):
    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(range(2, k), data, marker="o", label=label)
    plt.tick_params(labelsize=20)
    plt.xlabel("K", size=20)
    plt.ylabel("SSE", size=20)
    plt.show()


# 删除0值和长度小于2的数据
def preprocess(current_list):
    current_result = []
    for current_value in current_list:
        if len(str(current_value)) < 2:
            continue
        current_result.append(current_value)
    return current_result


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 直接读取文件中的words
    path_global = "data/观点抽取结果.xlsx"
    data_global = pd.read_excel(path_global, nrows=debugLength if debug else None, engine="openpyxl")
    words_global = data_global["Safety"].tolist()
    if debug:
        print("before:", words_global)
    # 删除0值和长度小于2的列
    words_global = preprocess(words_global)
    if debug:
        print("after:", words_global)

    # 提取词向量
    embeddings_global = load_w2v(words_global)
    # if debug:
    #     print(embeddings_global)
    # embeddings_global = np.array(embeddings_global)
    # if debug:
    #     print(embeddings_global)

    # 判断聚类簇数
    k = 25
    SSE = []  # 根据手肘法判断结果优劣
    SSC = []  # 轮廓系数
    for i in range(2, k):
        km_cluster = KMeans(n_clusters=i, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
        # 返回各自文本的所被分配到的类索引
        # result = km_cluster.fit_predict(tfidf_matrix)
        result = km_cluster.fit(embeddings_global)
        print("Predicting result: ", result)
        SSC.append(silhouette_score(embeddings_global, result.labels_, metric="euclidean"))
        SSE.append(result.inertia_)

    drawing(SSE, "SSE")
    # pca降维
    pca = PCA(n_components=2)
    embeddings_global = pca.fit_transform(embeddings_global)

    # 确定了聚类簇数之后，执行下面代码进行类别判断与保存
    '''
    df = pd.DataFrame()
    # 聚类
    s_path_global = "test/5-clustering-test.xlsx" if debug else "result/5-clustering.xlsx"
    final_cluster = KMeans(n_clusters=16, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
    result = final_cluster.fit(embeddings_global)
    print(result.labels_)
    print(type(result.labels_))
    df["category"] = pd.Series(result.labels_)
    df["words"] = pd.Series(words_global)

    # print(reduced_embeddings)
    # print(type(reduced_embeddings))
    df["X"] = pd.Series(reduced_embeddings[:, 0])
    df["Y"] = pd.Series(reduced_embeddings[:, 1])
    df.to_excel(s_path_global, index=False)
    '''

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


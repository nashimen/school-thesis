import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


# 生成停用词表
stop_words = []
f = open(file='stopwords.txt', mode='r', encoding='utf-8')  # 文件为123.txt
sourceInLines = f.readlines()
f.close()
for line in sourceInLines:
    temp = line.strip('\n')
    stop_words.append(temp)


def calculate_tfidf(texts):
    # print("texts:", texts)

    segs = []
    # 去标点符号→分词→去停用词
    for text in texts:
        for word in jieba.cut(text):
            if word not in stop_words and not word.isdigit():
                segs.append(word)
        # segs.extend([word for word in jieba.cut(text) if word not in stop_words])
    corpus = [" ".join(segs)]
    print("corpus' length = ", len(corpus))
    # print("corpus:", corpus)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print("words:", words)
    # print("weight:", weight)
    print("weight's length = ", len(weight))
    print("words' length = ", len(words))

    return words


# 读取词向量
w2v_path = "../sources/Tencent_AILab_ChineseEmbedding_5w.bin" if debug else "../sources/Tencent_AILab_ChineseEmbedding_300w.bin"
def load_w2v(words):
    t1 = time.time()
    wv_from_text = KeyedVectors.load(w2v_path)
    wv_from_text.init_sims(replace=True)  # 神奇，很省内存，可以运算most_similar
    print("加载w2v文件耗时：", (time.time() - t1) / 60.0, "minutes")
    w2v = wv_from_text.wv

    # 创建词向量索引字典
    embeddings_index = {}
    # 遍历得到word对应的embedding
    for word in w2v.vocab.keys():
        embeddings_index[word] = wv_from_text[word]
    # print("词向量维度：", len(embeddings_index.values()[0]))
    embeddings = []
    for word in words:
        # embeddings.append(wv_from_text[word])
        embeddings.append(embeddings_index.get(word, np.zeros(200)))
        # embeddings[word] = embeddings_index.get(word, np.array(200))

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


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    '''
    # 读取文件
    print(">>>正在读取数据。。。")
    path_global = "data/data-focal-v3.xlsx"
    name_global = "原始数据"
    data_global = pd.read_excel(path_global, name_global)

    # 处理指定属性
    attribute_global = "taste"
    attribute_label_global = attribute_global + "_label"
    texts_global = data_global.loc[(data_global[attribute_label_global] < 0.5) & (data_global[attribute_label_global] > -1)][attribute_global]
    texts_global = texts_global.tolist()

    # if debug:
    #     texts_global = ["原始数据", "是否助农产品", "助农商家", "助农扶贫馆", "海盐颗粒花生酱"]

    # 提取tfidf矩阵和词语+保存
    words_global = calculate_tfidf(texts_global)
    '''

    # 直接读取文件中的words
    path_global = "data/4-hot-words.xlsx"
    data_global = pd.read_excel(path_global, "drawing")
    words_global = data_global["taste"].tolist()

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


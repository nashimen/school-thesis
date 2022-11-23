import time, codecs, csv, math, numpy as np, random, datetime, os, pandas as pd, jieba, re, sys, string, lexical_diversity
# import paddlehub as hub
# from pylab import *
import emoji
from LAC import LAC
from gensim import utils
import gensim.models
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 150

# 读词向量CBOW200
model = gensim.models.Word2Vec.load('model/gensim_cbow200_5.model')
word_vector = model.wv


def split_row(data, column):
    """拆分成行

    :param data: 原始数据
    :param column: 拆分的列名
    :type data: pandas.core.frame.DataFrame
    :type column: str
    """
    row_len = list(map(len, data[column].values))
    rows = []
    for i in data.columns:
        if i == column:
            row = np.concatenate(data[i].values)
        else:
            row = np.repeat(data[i].values, row_len)
        rows.append(row)
    return pd.DataFrame(np.dstack(tuple(rows))[0], columns=data.columns)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取名词
    nouns = pd.read_csv('data/4 nouns.txt')
    nouns.columns = ['nouns']
    nouns = nouns[~ nouns['nouns'].isna()]
    nouns['temp'] = '1'
    print("1 nouns.shape:", nouns.shape)
    print(nouns.head())
    # 提取评论中的名词转为DF
    nouns_split = nouns.drop("nouns", axis=1).join(
        nouns["nouns"].str.split(" ", expand=True).stack().reset_index(level=1, drop=True).rename("noun"))
    print("2 nouns_split.shape:", nouns_split.shape)
    print(nouns_split.head())
    nouns_split = nouns_split.drop('temp', axis=1)
    # 名词去重
    nouns_split = nouns_split.drop_duplicates(subset='noun')
    print("3 nouns_split.shape:", nouns_split.shape)
    print(nouns_split.head())

    # 将名词转为词向量,可能有些名词没有词向量
    nouns_split = nouns_split[nouns_split['noun'].apply(lambda x: x in model.wv.index_to_key)]
    print("4 nouns_split.shape:", nouns_split.shape)
    print(nouns_split.head())
    nouns_split['CBOW200'] = nouns_split['noun'].apply(lambda x: model.wv[x])
    print("5 nouns_split.shape:", nouns_split.shape)
    print(nouns_split.head())
    nouns_split['CBOW200'] = nouns_split['CBOW200'].apply(lambda x: ' '.join([str(number) for number in x]))
    print("6 nouns_split.shape:", nouns_split.shape)
    # print(nouns_split["CBOW200"])
    print(nouns_split["noun"])

    # 准备聚类用的词向量
    # tfidf_matrix =
    cbow200 = nouns_split['CBOW200'].apply(lambda x: [float(number) for number in x.split(' ')]).tolist()

    # KMeans聚类
    print("开始训练聚类模型")
    k = 15
    SSE = []  # 根据手肘法判断结果优劣
    SSC = []  # 轮廓系数
    for i in range(2, k):
        km_cluster = KMeans(n_clusters=i, max_iter=300, n_init=40, init='k-means++')
        # 返回各自文本的所被分配到的类索引
        result = km_cluster.fit(cbow200)
        print("Predicting result: ", result)
        SSC.append(silhouette_score(cbow200, result.labels_, metric="euclidean"))
        SSE.append(result.inertia_)

    # 绘制曲线
    plt.plot(range(2, k), SSE, marker="o", label=u'SSE')
    plt.xlabel("K值——簇数量", size=15)
    plt.ylabel("簇内误方差SSE", size=15)
    plt.show()
    plt.plot(range(2, k), SSC, marker="*", label=u'SSC')
    plt.xlabel("K值——簇数量", size=15)
    plt.ylabel("轮廓系数SSC", size=15)
    plt.show()

    '''

    # 确定了聚类簇数之后，执行下面代码进行类别判断与保存
    df = pd.DataFrame()
    # 聚类
    s_path_global = "result/1-clustering_50 times_" + str(k) + ".xlsx"
    final_cluster = KMeans(n_clusters=k, max_iter=300, n_init=40, init='k-means++')
    result = final_cluster.fit(cbow200)
    # print(result.labels_)
    print(type(result.labels_))
    df["category"] = pd.Series(result.labels_)
    df["words"] = pd.Series(nouns_split['noun'].tolist())

    df.to_excel(s_path_global, index=False)
    '''

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    sys.exit(10086)


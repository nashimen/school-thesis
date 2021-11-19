import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 100


# 判断是否包含中文字符
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


# 计算情感
# 初始化接口
senta = hub.Module(name='senta_cnn')
def calculate_sentiment(row):
    if debug:
        return 0.5
    if not is_Chinese(row):
        print("row:", row)
        return 0.5

    input_dict = {"text": [row]}
    res = senta.sentiment_classify(data=input_dict)
    positive_probs = []
    for r in res:
        positive_probs.append(r['positive_probs'])

    return positive_probs[0]


# 获取停用词
stoplist = pd.read_csv('../stopwords.txt').values
# 计算词汇丰富度，去停用词后，按照如下公式计算
# 公式：richness=不同的词汇个数/所有词汇个数
def calculate_richness(row):
    total_length = len(row)
    if total_length < 1:
        return 0
    length = 0
    for word in row:
        if word not in stoplist:
            length += 1
    return length / total_length


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # path_global = "test/data-test.xlsx" if debug else "data/data.xlsx"
    path_global = "data/data_sentiment-v2.xlsx"
    # s_path_global = "test/data_sentiment-test.xlsx" if debug else "result/data_sentiment.xlsx"

    # 读取数据
    data_global = pd.read_excel(path_global, nrows=debugLength if debug else None)

    # print("1-商品链接总数为：", len(set(data_global["商品链接"])))

    s_path_global = "test/data_density-test.xlsx" if debug else "result/data_density.xlsx"
    # 计算情感
    # data_global["Sentiment"] = data_global.apply(lambda row_global: calculate_sentiment(str(row_global["评论"])), axis=1)

    # 计算词汇丰富度
    # 去标点符号→分词
    data_global["Words"] = data_global["评论文本"].apply(lambda x: re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？、~@#￥%……&*（）．；：】【|]+", "", str(x)))
    data_global["Words"] = data_global["Words"].apply(lambda x: list(jieba.cut(x)))
    data_global["Richness"] = data_global.apply(lambda row_global: calculate_richness(row_global["Words"]), axis=1)

    # 保存结果
    data_global.to_excel(s_path_global, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


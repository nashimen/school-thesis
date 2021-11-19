import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


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


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    path_global = "test/data-test.xlsx" if debug else "data/data.xlsx"
    # s_path_global = "test/data_sentiment-test.xlsx" if debug else "result/data_sentiment.xlsx"

    # 读取数据
    data_global = pd.read_excel(path_global)

    # print("1-商品链接总数为：", len(set(data_global["商品链接"])))

    # 计算情感
    # 依次计算每个产品
    products = set(data_global["产品"].tolist())
    for product in products:
        s_path_global = "test/data_sentiment_" + product + "-test.xlsx" if debug else "result/data_sentiment_" + product + ".xlsx"
        if os.path.exists(s_path_global):
            print("current product has finished:", product)
            continue
        print("current product is ", product)
        current_data = data_global.loc[data_global["产品"] == product]
        # 分别计算助农&非助农数据
        tags = set(current_data["是否助农"].tolist())
        for tag in tags:
            s_path_global = "test/data_sentiment_" + product + "-test.xlsx" if debug else "result/data_sentiment_" + product + "_" + tag + ".xlsx"
            current_data_tag = current_data.loc[current_data["是否助农"] == tag]
            current_data_tag["sentiment"] = current_data_tag.apply(lambda row_global: calculate_sentiment(str(row_global["评论文本"])), axis=1)
            current_data_tag.to_excel(s_path_global, index=False)

        # current_data["sentiment"] = current_data.apply(lambda row_global: calculate_sentiment(str(row_global["评论文本"])), axis=1)
        # 保存当前产品结果
        # current_data.to_excel(s_path_global, index=False)
    # data_global["sentiment"] = data_global.apply(lambda row_global: calculate_sentiment(str(row_global["评论文本"])), axis=1)

    # print("2-商品链接总数为：", len(set(data_global["商品链接"])))

    # 保存结果
    # data_global.to_excel(s_path_global, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


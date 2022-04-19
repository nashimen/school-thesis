import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
import paddlehub as hub
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True
debugLength = 30


# 计算短文本情感
senta = hub.Module(name='senta_cnn')
def calculate_sentiment(text):
    if pd.isna(text):
        return -1
    # print("text:", text)
    input_dict = {"text": [text]}
    res = senta.sentiment_classify(data=input_dict)
    positive_probs = []
    for r in res:
        # print(r["sentiment_label"], r["sentiment_key"], r['positive_probs'], r["negative_probs"], r["text"])
        positive_probs.append(r['positive_probs'])
    if len(positive_probs) > 1:
        print("出错啦！怎么会有那么多！")
    return positive_probs[0]


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    print(">>>正在读取数据。。。")
    path = "data/test/test.xlsx" if debug else "data/爱卡汽车汇总.xlsx"
    current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")

    # drop一些属性
    current_data.drop(["品牌", "车系", "作者", "外观", "内饰", "空间", "舒适", "能耗", "动力", "操控", "性价比", "发表时间", "车型", "购车时间", "地区", "价格", "油耗里程", "购车目的", "综述", "最满意", "最不满意", "点赞数"], axis=1, inplace=True)

    s_path = "data/test/sentiment-test.xlsx" if debug else "result/sentiment.xlsx"

    # 遍历属性→计算情感→依次保存
    attributes = ["外观评论", "内饰评论", "空间评论", "舒适评论", "能耗评论", "动力评论", "操控评论", "性价比评论"]
    for attribute in attributes:
        print("正在处理：", attribute)
        current_label = attribute + "_senti"
        current_data[current_label] = current_data.apply(lambda row: calculate_sentiment(row[attribute]), axis=1)

        current_data.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")



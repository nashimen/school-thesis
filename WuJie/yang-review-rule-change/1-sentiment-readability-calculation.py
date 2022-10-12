import time, codecs, csv, math, numpy as np, random, datetime, os, pandas as pd, jieba, re, sys, string, lexical_diversity
import paddlehub as hub
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 30


# 计算短文本情感
senta = hub.Module(name='senta_cnn')
def calculate_sentiment(text):
    if pd.isna(text) or str.isdigit(text):
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


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


# 合并属性评论
def merge_attribute(current_row, attrs):
    merged = ""
    for attribute in attrs:
        if pd.isna(current_row[attribute]) or is_number(current_row[attribute]):
            continue
        # 判断当前属性评论是否包含句子符号，如果没有则最后加一个句号
        number = current_row[attribute].count(".") + current_row[attribute].count("。") + \
                 current_row[attribute].count("？") + current_row[attribute].count("?") + \
                 current_row[attribute].count("！") + current_row[attribute].count("!") + \
                 current_row[attribute].count(";") + current_row[attribute].count("；")
        if number == 0:
            merged += str(current_row[attribute]) + "。"
        else:
            merged += str(current_row[attribute])

    return merged


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    print(">>>正在读取数据。。。")

    current_website = "Ctrip"

    if current_website == "qunar":
        path = "test/merged_qunar-test.xlsx" if debug else "data/merged_qunar.xlsx"
    else:
        # Ctrip
        path = "test/merged_ctrip-test.xlsx" if debug else "data/merged_ctrip.xlsx"
    current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")

    # 遍历属性→计算情感→依次保存
    s_path = "test/sentiment-" + current_website + "-test.xlsx" if debug else "result/sentiment-" + current_website + ".xlsx"
    current_data["评论文本_sentiment"] = current_data.apply(lambda row: calculate_sentiment(row["评论文本"]), axis=1)
    if current_website == "qunar":
        current_data["题目_sentiment"] = current_data.apply(lambda row: calculate_sentiment(row["题目"]), axis=1)
        # Calculate sentiments of merged text
        current_data["merged"] = current_data.apply(lambda row: merge_attribute(row, ["题目", "评论文本"]), axis=1)
        current_data["merged_sentiment"] = current_data.apply(lambda row: calculate_sentiment(row["merged"]), axis=1)

    if current_website == "qunar":
        current_data.drop(["题目", "评论文本", "merged", "点赞数", "作者", "地区", "出行目的", "评论数", "链接地址", "图片地址", "发布日期"], axis=1, inplace=True)
    else:
        current_data.drop(["作者", "评论文本", "房型", "发布日期", "出行目的", "作者点评数", "点赞数", "酒店回复"], axis=1, inplace=True)

    current_data.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")



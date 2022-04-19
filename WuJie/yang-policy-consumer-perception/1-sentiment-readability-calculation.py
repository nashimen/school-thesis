import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys, string, lexical_diversity
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


# 计算评论合并之后的长度
def calculate_length(current_row):
    return len(current_row)


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


# 统计句子数量
def calculate_sentence_number(current_row):
    number = current_row.count(".") + current_row.count("。") + current_row.count("？") + current_row.count("?") + \
             current_row.count("！") + current_row.count("!") + current_row.count(";") + current_row.count("；")
    if number == 0:
        print("current_row:", current_row)
    return number


# 统计句子平均长度
def calculate_average_sentence_length(current_number, current_length):
    # print("current_number:", current_number, ", current_length:", current_length)
    if current_number == 0:
        return -1
    return current_length / current_number


# 计算平均词语数量
def calculate_average_word_number(current_row, current_length):
    # 去标点符号
    line = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(current_row))
    # 分词
    line = jieba.cut(line)
    # print(type(line),", line:", line)
    line = [word.strip() for word in line]

    if len(line) == 0:
        return -1

    # 计算数量
    return current_length / len(line)


# 计算可读性
def calculate_readability(current_sentence_length, current_word_number):
    if current_sentence_length <= 0 or current_word_number <= 0:
        return -1
    return (0.39 * current_sentence_length) + (11.8 * current_word_number) - 15.59


# 使用python自带的lexical_diversity函数计算
def calculate_readability2(text):
    return lexical_diversity(text)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    print(">>>正在读取数据。。。")

    current_website = "爱卡汽车"

    if current_website == "爱卡汽车":
        path = "data/test/爱卡汽车-test.xlsx" if debug else "data/爱卡汽车汇总.xlsx"
        current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")
        # drop一些属性
        current_data.drop(["品牌", "车系", "作者", "外观", "内饰", "空间", "舒适", "能耗", "动力", "操控", "性价比", "发表时间", "车型", "购车时间", "地区", "价格", "油耗里程", "购车目的", "综述", "最满意", "最不满意", "点赞数"], axis=1, inplace=True)
        attributes = ["外观评论", "内饰评论", "空间评论", "舒适评论", "能耗评论", "动力评论", "操控评论", "性价比评论"]
    elif current_website == "汽车之家":
        path = "data/test/汽车之家-test.xlsx" if debug else "data/汽车之家感知表.xlsx"
        current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")
        # drop一些属性
        current_data.drop(["题目", "链接", "发帖人_链接", "发帖人_编号","外观评分", "内饰评分", "空间评分", "舒适性评分", "能耗评分", "动力评分", "操控评分", "性价比评分", "发帖时间", "发帖人","车系", "车型","购买时间", "地点", "价格", "购车原因", "最满意评论", "最不满意评论", "支持", "浏览", "评论"], axis=1, inplace=True)
        attributes = ["外观评论", "内饰评论", "空间评论", "舒适性评论", "能耗评论", "动力评论", "操控评论", "性价比评论"]
    else:
        # 太平洋汽车
        path = "data/test/太平洋汽车-test.xlsx" if debug else "data/太平洋汽车汇总.xlsx"
        current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")
        # drop一些属性
        current_data.drop(["车系", "车型", "购买时间", "购买地点", "价格", "平均能耗", "行驶里程", "作者", "发表时间", "总分", "外观", "内饰", "空间", "配置", "动力", "操控", "能耗", "舒适", "优点", "缺点", "点赞数", "回复数"], axis=1, inplace=True)
        attributes = ["外观评论", "内饰评论", "空间评论", "配置评论", "动力评论", "操控评论", "能耗评论", "舒适评论"]

    # 遍历属性→计算情感→依次保存
    s_path = "data/test/sentiment-" + current_website + "-test.xlsx" if debug else "result/sentiment-" + current_website + ".xlsx"
    for attribute in attributes:
        print("正在处理：", attribute)
        current_label = attribute + "_senti"
        current_data[current_label] = current_data.apply(lambda row: calculate_sentiment(row[attribute]), axis=1)

        current_data.to_excel(s_path, index=False)
    '''
    '''

    # 计算可读性
    # 遍历每一行→属性用句号拼接起来→统计评论字数→
    # for index, row in current_data.iterrows():
    current_data["merged"] = current_data.apply(lambda row: merge_attribute(row, attributes), axis=1)
    current_data["length"] = current_data.apply(lambda row: calculate_length(row["merged"]), axis=1)

    # 1.1统计句子数量（保存）→求平均句子长度
    current_data["sentence_number"] = current_data.apply(lambda row: calculate_sentence_number(row["merged"]), axis=1)
    current_data["average_sentence_length"] = current_data.apply(lambda row: calculate_average_sentence_length(row["sentence_number"], row["length"]), axis=1)
    # 1.2分词→统计词语个数→求平均词语字数
    current_data["average_word_number"] = current_data.apply(lambda row: calculate_average_word_number(row["merged"], row["length"]), axis=1)
    # 2.计算可读性
    current_data["readability"] = current_data.apply(lambda row: calculate_readability(row["sentence_number"], row["average_word_number"]), axis=1)
    # 3.使用lexical_diversity计算一遍多样性,这个好像是针对英文的？
    # current_data["readability_ld"] = current_data.apply(lambda row: calculate_readability2(row["merged"]), axis=1)

    s_path_readability = "data/test/readability-" + current_website + "-test.xlsx" if debug else "result/readability-" + current_website + ".xlsx"
    current_data.to_excel(s_path_readability, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")



import time, codecs, csv, math, numpy as np, random, datetime, os, pandas as pd, jieba, re, sys, string, lexical_diversity
import paddlehub as hub
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True
debugLength = 30


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
    # print("content = ", content)
    # 使用逗号替换空格
    content = re.sub(r" +", ",", str(current_row))
    pattern = r'\.|;|\?|!|。|；|！|\?'
    docs = list(re.split(pattern, content))
    # print("docs pre = ", docs)
    docs = list(filter(lambda x: len(str(x).strip()) > 0, docs))
    # 去掉不包含中文字符的短文本
    for doc in docs:
        if not is_Chinese(doc):
            print("非中文：", doc)
            docs.remove(doc)

    if len(docs) == 0:
        print("current_row:", current_row)

    return len(docs)


# 判断是否包含中文字符
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


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

    # 计算可读性
    # 遍历每一行→属性用句号拼接起来→统计评论字数→
    # for index, row in current_data.iterrows():
    # current_data["merged"] = current_data.apply(lambda row: merge_attribute(row, attributes), axis=1)
    current_data["length"] = current_data.apply(lambda row: calculate_length(row["评论文本"]), axis=1)
    # 1.1统计句子数量（保存）→求平均句子长度
    current_data["sentence_number"] = current_data.apply(lambda row: calculate_sentence_number(row["评论文本"]), axis=1)
    current_data["average_sentence_length"] = current_data.apply(lambda row: calculate_average_sentence_length(row["sentence_number"], row["length"]), axis=1)
    # 1.2分词→统计词语个数→求平均词语字数
    current_data["average_word_number"] = current_data.apply(lambda row: calculate_average_word_number(row["评论文本"], row["length"]), axis=1)
    # 2.计算可读性
    current_data["readability"] = current_data.apply(lambda row: calculate_readability(row["sentence_number"], row["average_word_number"]), axis=1)

    # 如果是qunar，则再进一步计算题目的可读性和汇总的可读性
    if current_website == "qunar":
        current_data["length-题目"] = current_data.apply(lambda row: calculate_length(row["题目"]), axis=1)
        # 1.1统计句子数量（保存）→求平均句子长度
        current_data["sentence_number-题目"] = current_data.apply(lambda row: calculate_sentence_number(row["题目"]), axis=1)
        current_data["average_sentence_length-题目"] = current_data.apply(lambda row: calculate_average_sentence_length(row["sentence_number-题目"], row["length-题目"]), axis=1)
        # 1.2分词→统计词语个数→求平均词语字数
        current_data["average_word_number-题目"] = current_data.apply(lambda row: calculate_average_word_number(row["题目"], row["length-题目"]), axis=1)
        # 2.计算可读性
        current_data["readability-题目"] = current_data.apply(lambda row: calculate_readability(row["sentence_number-题目"], row["average_word_number-题目"]), axis=1)

        # 计算merged后可读性
        current_data["merged"] = current_data.apply(lambda row: merge_attribute(row, ["题目", "评论文本"]), axis=1)
        current_data["length-merged"] = current_data.apply(lambda row: calculate_length(row["merged"]), axis=1)
        # 1.1统计句子数量（保存）→求平均句子长度
        current_data["sentence_number-merged"] = current_data.apply(lambda row: calculate_sentence_number(row["merged"]), axis=1)
        current_data["average_sentence_length-merged"] = current_data.apply(lambda row: calculate_average_sentence_length(row["sentence_number-merged"], row["length-merged"]), axis=1)
        # 1.2分词→统计词语个数→求平均词语字数
        current_data["average_word_number-merged"] = current_data.apply(lambda row: calculate_average_word_number(row["merged"], row["length-merged"]), axis=1)
        # 2.计算可读性
        current_data["readability-merged"] = current_data.apply(lambda row: calculate_readability(row["sentence_number-merged"], row["average_word_number-merged"]), axis=1)

    if current_website == "qunar":
        if debug:
            current_data.drop(["点赞数", "作者", "地区", "出行目的", "评论数", "链接地址", "图片地址", "发布日期"], axis=1, inplace=True)
        else:
            current_data.drop(["点赞数", "作者", "地区", "评论文本", "题目", "merged", "出行目的", "评论数", "链接地址", "图片地址", "发布日期"], axis=1, inplace=True)
    else:
        if debug:
            current_data.drop(["作者", "房型", "发布日期", "出行目的", "作者点评数", "点赞数", "酒店回复"], axis=1, inplace=True)
        else:
            current_data.drop(["作者", "房型", "发布日期", "评论文本", "出行目的", "作者点评数", "点赞数", "酒店回复"], axis=1, inplace=True)

    s_path_readability = "test/readability-" + current_website + "-test.xlsx" if debug else "result/readability-" + current_website + ".xlsx"
    current_data.to_excel(s_path_readability, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")



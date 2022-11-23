import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
import jieba.posseg as pseg
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 30


stoplist = []
f = open('stopwords.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    stoplist.append(line.strip())


# 输入一条评论数据，输出其中的candidates
def obtain_candidates_single_review(current_review):
    current_result = []
    words = pseg.cut(current_review)
    # print("words:", words)
    try:
        for word, flag in words:
            # if str(flag) is 'a' or str(flag) is 'd':
            if str(flag) is 'a':
                if debug:
                    print(word, flag)
                if word not in stoplist:
                    current_result.append(word)
    except Exception as e:
        print("Exception:", e)

    return current_result


# process online reviews: segment word -> 词性判断 -> remain adj/adv -> remove stop words
def obtain_candidates(current_corpus, attribute_list):
    print(current_corpus.head())

    # 保存所有candidates
    current_candidates = set()

    # 遍历所有属性
    for current_attribute in attribute_list:
        print("正在处理：", current_attribute)
        current_reviews = current_corpus[current_attribute + "评论"]
        # 分词→判断词性→保存adj/adv至Set
        for current_review in current_reviews:
            current_candidates_single = obtain_candidates_single_review(current_review)
            if len(current_candidates_single) < 1:
                continue
            for word in current_candidates_single:
                current_candidates.add(word)

    return current_candidates


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # read corpus, namely online reviews
    corpus_path = "test/1 Corpus-test.xlsx" if debug else "data/1 Corpus.xlsx"
    corpus = pd.read_excel(corpus_path, engine="openpyxl")

    # 属性
    attributes = ["外观", "内饰", "空间", "动力", "操控", "能耗", "舒适性", "性价比"]

    # process online reviews: segment word -> 词性判断 -> remain adj/adv -> remove stop words
    candidates = obtain_candidates(corpus, attributes)
    if debug:
        print(candidates)
    print("candidates:", len(candidates))

    # save candidates
    save_path = "test/2 candidates-test.txt" if debug else "result/1 candidates(only adjectives).txt"
    file = open(save_path, 'w')
    file.write(str(list(candidates)))
    file.close()

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")
    sys.exit(10086)


import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
import jieba.posseg as pseg
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True
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
            if str(flag) is 'a' or str(flag) is 'd':
                print(word, flag)
                if word not in stoplist:
                    current_result.append(word)
    except Exception as e:
        print("Exception:", e)

    return current_result


# process online reviews: segment word -> 词性判断 -> remain adj/adv -> remove stop words
def obtain_candidates(current_corpus):
    print(current_corpus.head())
    # 分词→判断词性→保存adj/adv至Set
    current_candidates = set()
    for current_review in current_corpus["评论文本"]:
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

    # process online reviews: segment word -> 词性判断 -> remain adj/adv -> remove stop words
    candidates = obtain_candidates(corpus)
    if debug:
        print(candidates)
    print("candidates:", len(candidates))

    # save candidates
    save_path = "test/2 candidates-test.txt" if debug else "result/1 candidates.txt"
    file = open(save_path, 'w')
    file.write(str(list(candidates)))
    file.close()

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


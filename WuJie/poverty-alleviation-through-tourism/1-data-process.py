import time, codecs, csv, math, numpy as np, random, datetime, os, pandas as pd, jieba, re, sys, string, lexical_diversity
# import paddlehub as hub
# from pylab import *
import emoji
from LAC import LAC

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 150


def get_nouns(lac):
    word, part = lac[0], lac[1]
    nouns = []
    for word_token, part_token in zip(word, part):
        if part_token == 'n':
            nouns.append(word_token)
    return nouns


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件+预处理
    raw_data_path = "data/1 raw data.xlsx"
    raw_data = pd.read_excel(raw_data_path, nrows=debugLength if debug else None, engine="openpyxl")
    raw_data = raw_data[~ raw_data['评论文本'].isnull()]
    print("1 raw_data's shape:", raw_data.shape)
    raw_data['评论文本'] = raw_data['评论文本'].apply(lambda x: "".join(emoji.replace_emoji(x).split()))
    print("2 raw_data's shape:", raw_data.shape)
    raw_data = raw_data[~ raw_data['评论文本'].str.contains(r'[a-zA-z]+://[^\s]*')]
    raw_data = raw_data[~ raw_data['评论文本'].isna()]
    print("3 raw_data's shape:", raw_data.shape)

    # 保存预处理之后的文件
    # raw_data_processed_path = "data/2 raw data-processed.xlsx"
    # raw_data.to_excel(raw_data_processed_path, index=False)

    # 分词
    lac = LAC(mode='lac')
    raw_data['LAC'] = raw_data['评论文本'].apply(lambda x: lac.run(x))
    raw_data['word'], raw_data['part'] = raw_data['LAC'].apply(lambda x: x[0]), raw_data['LAC'].apply(lambda x: x[1])
    print("4 raw_data's shape:", raw_data.shape)
    # 去停用词
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = f.readlines()
    stop_words = [stop_word.strip('\n') for stop_word in stop_words]
    raw_data['without_stop_word'] = raw_data['word'].apply(lambda x: list(filter(lambda word: word not in stop_words, x)))
    # 获取名词
    raw_data['nouns'] = raw_data['LAC'].apply(get_nouns)
    raw_data['nouns_corpus'] = raw_data['nouns'].apply(lambda x: ' '.join(x))
    # 准备训练词向量的语料库
    raw_data['word2vec_corpus'] = raw_data['without_stop_word'].apply(lambda x: ' '.join(x))

    # 保存所有结果
    raw_data.to_excel('data/2 data_POS_corpora.xlsx', index=False)
    raw_data['word2vec_corpus'].to_csv('data/3 word2vec_corpus.txt', index=False)
    raw_data['nouns_corpus'].to_csv('data/4 nouns.txt', index=False, header=False)
    raw_data['word'].apply(lambda x: ' '.join(x)).to_csv('data/5 word2vec_corpus_with_stoped_punctuation.txt', index=False, header=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    sys.exit(10086)


import time, codecs, csv, math, numpy as np, random, datetime, os, pandas as pd, jieba, re, sys, string, lexical_diversity
# import paddlehub as hub
# from pylab import *
import emoji
from LAC import LAC
from gensim import utils
import gensim.models

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 150


class MyCorpus:
    def __iter__(self):
        corpus_path = 'data/5 word2vec_corpus_with_stoped_punctuation.txt'
        for line in open(corpus_path, encoding='utf-8'):
            yield utils.simple_preprocess(line)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    sentences = MyCorpus()

    # 训练词向量
    # CBOW 200
    min_count = 50
    cbow200 = gensim.models.Word2Vec(sentences=sentences, vector_size=200, min_count=min_count)
    model_save_path = "model/gensim_cbow200_" + str(min_count) + ".model"
    cbow200.save(model_save_path)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    sys.exit(10086)


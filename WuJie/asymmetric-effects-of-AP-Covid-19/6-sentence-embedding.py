import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)


def test():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sent1 = "我是中国人"
    sent2 = "他是新加坡人"
    embedding1 = model.encode(sent1, convert_to_numpy=True)
    embedding2 = model.encode(sent2, convert_to_tensor=True)
    print(embedding1)
    print("*" * 50)
    print(embedding2)
    print("*" * 50)
    print("sent1's length = ", len(embedding1))
    print("sent2's length = ", len(embedding2))


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    test()

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


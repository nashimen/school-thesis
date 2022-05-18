import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


def similarity_test():
    a = [1, 0, 2, 1]
    b = [1, 0, 1, 1]

    sim = cosine_similarity([a], [b])

    print("sim = ", sim[0][0])


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    similarity_test()

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


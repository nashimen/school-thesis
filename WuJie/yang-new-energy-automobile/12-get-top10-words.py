import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib;matplotlib.use('tkagg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)


def fetching(path):
    data = pd.read_excel(path)
    values = list(set(data["Value"].tolist()))
    aspects = list(set(data["Aspect"].tolist()))
    years = list(set(data["Year"].tolist()))
    for value in values:
        data_value = data.loc[data["Value"] == value]
        for year in years:
            data_year = data_value.loc[data_value["Year"] == year]
            for aspect in aspects:
                if aspect == "目的类别":
                    continue
                data_aspect = data_year.loc[data_year["Aspect"] == aspect]
                data_aspect = data_aspect.sort_values(by="Tfidf", ascending=False)
                print(value, year, aspect, ":", ','.join(data_aspect["Keyword"].tolist()[:10]))


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    current_type = "purpose"
    path_global = "data/importance_words_" + current_type + "-20211017.xlsx"
    fetching(path_global)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


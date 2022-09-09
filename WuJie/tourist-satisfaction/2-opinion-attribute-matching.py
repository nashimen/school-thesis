import time, codecs, csv, math, numpy as np, random, datetime, os, gc, jieba, re, sys, pandas as pd
import jieba.posseg as pseg

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True
debugLength = 30

# 匹配原则：
# 1.只保留同时包括评价对象和观点的数据
# 2.


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 1. 读取观点文件
    path = "data/1 opinion extraction results.xlsx"
    data_global = pd.read_excel(path, engine="openpyxl")
    print(data_global.head())

    # 2. 依次处理每行，判断其中观点所属的属性（一共5个属性，增加5列）

    # 3. 保存到文件

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


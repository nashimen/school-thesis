import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


# 读取文件夹路径下所有文件名，合并所有数据
# 文件名的格式为：店铺名 商品名+1/2/3，1-好评&2-中评&3-差评
def read_write(path, s_path):
    for root, dirs, files in os.walk(path):
        pass
        # print(files)
        # print(type(files))

    result = pd.DataFrame()
    for file_name in files:
        current_path = path + "/" + file_name
        temp = file_name.split(" ")
        shop_name = temp[0]
        product_name = "".join(temp[1:])
        # print(shop_name, product_name)
        current = pd.read_excel(current_path)
        current["shop"] = shop_name
        current["product"] = product_name
        result = result.append(current)

    result.to_excel(s_path)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    path_global = "data/origin"
    s_path_global = "data/merged.xlsx"
    read_write(path_global, s_path_global)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)


def process_file(current_file_dir, name):
    result = pd.DataFrame(columns=["name", "comment_date", "comment", "readCnt", "userLevel", "star"])
    f = open(current_file_dir + "/" + name, 'rb')
    name = name.split("_")[0]
    mylist = pickle.load(f)
    for l in mylist:
        comment = l.get("comment")
        readCnt = l.get("readCnt")
        star = l.get("star")
        userLevel = l.get("userLevel")
        comment_date = datetime.datetime.strptime(l.get("create_time"), "%Y-%m-%d %H:%M:%S")
        result = result.append([{"name": name, "comment_date": comment_date, "userLevel": userLevel, "readCnt": readCnt, "star": star, "comment": comment}])

    return result


# 加载和保存数据
def load_save(path, s_path):
    # final = pd.DataFrame(columns=["name", "comment_date", "comment", "readCnt", "userLevel", "star"])
    # 获取当前路径下所有文件名
    for root, dirs, file_names in os.walk(path):
        print(file_names)
    for name in file_names:
        print("正在处理：", name)
        # 读取当前文件并处理
        current_data = process_file(path, name)
        current_data.to_csv(s_path, index=False, mode="a", encoding="utf_8_sig")
        # final.append(current_data)

    # final.to_excel(s_path, index=False)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    path = "data/originFiles"
    s_path = "data/merged_file.csv"
    load_save(path, s_path)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

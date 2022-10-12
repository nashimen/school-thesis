import time, codecs, csv, math, numpy as np, random, datetime, os, pandas as pd, jieba, re, sys, string, lexical_diversity
import paddlehub as hub
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True
debugLength = 30


# 计算短文本情感
def calculate_image(text):
    if pd.isna(text) or str.isdigit(text):
        return 0
    # if len(text) == 0:
    #     return 0

    return text.count("img_")


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    print(">>>正在读取数据。。。")

    current_website = "qunar"

    if current_website == "qunar":
        path = "test/merged_qunar-test.xlsx" if debug else "data/merged_qunar.xlsx"
    else:
        # Ctrip
        path = "data/merged_ctrip.xlsx"
    current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")

    # 计算可读性
    current_data["image"] = current_data.apply(lambda row: calculate_image(row["图片地址"]), axis=1)

    if current_website == "qunar":
        current_data.drop(["点赞数", "作者", "地区", "出行目的", "评论数", "链接地址", "发布日期"], axis=1, inplace=True)
    else:
        current_data.drop(["作者", "房型", "发布日期", "出行目的", "作者点评数", "点赞数", "酒店回复"], axis=1, inplace=True)

    s_path_image = "test/image-" + current_website + "-test.xlsx" if debug else "result/image-" + current_website + ".xlsx"
    current_data.to_excel(s_path_image, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")



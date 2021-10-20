import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy
import pandas as pd
from aip import AipNlp
import sys
import datetime
import traceback
import time
sys.setrecursionlimit(5000000)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib;matplotlib.use('tkagg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False

""" 你的 APPID AK SK """
APP_ID = '24520514'
API_KEY = 'NH1mjaWqSj1WWuBB74vEWBoj'
SECRET_KEY = 'gQnGvSfR2KRhD6GOhR1Wbj9lXw8YTz4n'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
options = dict()
options["type"] = 10


# 访问百度接口,入参为评论文本
def baidu_api_access(text):
    local_result_api = ''
    try:
        local_result_api = client.commentTag(text, options)
        error_code = 'error_code'
        i = 0
        while error_code in local_result_api.keys():
            if i > 50:
                print(local_result_api)
                break
            local_result_api = client.commentTag(text, options)
            i = i + 1
        # if debug:
        #     print("接口访问了", i, '次:', text)
        if i > 8:
            print("接口访问了", i, '次:', text)
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt text = ", text)
        print("KeyboardInterrupt local_result_api = ", local_result_api)
        traceback.print_exc()
    except Exception as e:
        print("Exception text = ", text)
        print("Exception local_result_api = ", local_result_api)
        traceback.print_exc()
    return local_result_api


# 访问百度接口，抽取评论观点
def extract_opinion(path, s_path, name):
    data = pd.read_excel(path, usecols=[14, 17, 19, 21, 23, 25, 27, 29, 31], sheet_name=name)
    # print(data.head())
    data = data.loc[data["购买年份-fix"] == 2017]
    # 只用2017年数据
    cols = data.columns
    print("cols:", cols)

    for col in cols:
        if col == "购买年份-fix":
            continue
        component_result = {}
        texts = data[col].tolist()
        for text in texts:
            text = str(text)
            if len(text.strip()) <= 1:
                continue
            # 调用接口
            result_api = baidu_api_access(text)
            if debug:
                print(text, ":", len(result_api), result_api)
            if result_api is None or len(result_api) <= 1:
                continue
            items = result_api.get('items')
            if debug:
                print("items:", items)
            if items is None or len(items) == 0:
                continue
            for current_dict in items:
                component = current_dict.get('prop')
                opinion = current_dict.get('adj', 'no_value')
                if len(opinion) == 0:
                    opinion = 'no_value'
                if component in component_result.keys():  # 字典中已有部件
                    if opinion in component_result.get(component).keys():  # 当前部件有此感知单元时
                        count = component_result[component][opinion]
                        component_result[component][opinion] = count + 1
                    else:  # 当前部件无此感知单元时
                        component_result[component][opinion] = 1
                else:
                    component_result[component] = {opinion: 1}  # 字典中暂无部件时
        # 将当前得到的component_result转为DataFrame
        result = {'attribute': [], 'component': [], 'opinion': [], 'count': []}
        for key, value in component_result.items():
            for k, v in value.items():
                result['attribute'].append(col)
                result['component'].append(key)
                result['opinion'].append(k)
                result['count'].append(v)
        print("result:", result)
        result = pd.DataFrame(result)
        # 追加写入文件
        print("正在写入文件：", col)
        result.to_csv(s_path, mode='a', encoding='utf_8_sig', header=None, index=False)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    path_global = "test/秦新能源数据-test.xlsx" if debug else "data/秦新能源数据.xlsx"
    s_path_global = "test/秦新能源数据-观点抽取结果-test.csv" if debug else "result/秦新能源数据-观点抽取结果.csv"
    sheet_name_global = "秦新能源"
    extract_opinion(path_global, s_path_global, sheet_name_global)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


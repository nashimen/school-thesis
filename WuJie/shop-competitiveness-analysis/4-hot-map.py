import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub
from pyheatmap.heatmap import HeatMap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import random
from pyecharts.charts import HeatMap
from pyecharts import options as opts

import warnings

warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False

import seaborn as sns
sns.set()
sns.set_style("whitegrid", {"font.sans-serif": ["simhei", "Arial"]})
def heatMapTest(data):
    df = pd.DataFrame(
        np.random.rand(4, 7),
        index=["天安门", "故宫", "奥林匹克森林公园", "八达岭长城"],
        columns=["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    )
    plt.figure(figsize=(10, 4))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="coolwarm")


# 获取颜色


# RGB格式颜色转换为16进制颜色格式
def RGB_to_Hex(rgb):
    RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    # print(color)
    return color


# RGB格式颜色转换为16进制颜色格式
def RGB_list_to_Hex(RGB):
    # RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    # print(color)
    return color


# 16进制颜色格式颜色转换为RGB格式
def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    rgb = str(r) + ',' + str(g) + ',' + str(b)
    # print(rgb)
    return rgb, [r, g, b]


def gradient_color(color_list, color_sum=15):
    color_center_count = len(color_list)
    # if color_center_count == 2:
    #     color_center_count = 1
    color_sub_count = int(color_sum / (color_center_count - 1))
    color_index_start = 0
    color_map = []
    for color_index_end in range(1, color_center_count):
        color_rgb_start = Hex_to_RGB(color_list[color_index_start])[1]
        color_rgb_end = Hex_to_RGB(color_list[color_index_end])[1]
        r_step = (color_rgb_end[0] - color_rgb_start[0]) / color_sub_count
        g_step = (color_rgb_end[1] - color_rgb_start[1]) / color_sub_count
        b_step = (color_rgb_end[2] - color_rgb_start[2]) / color_sub_count
        # 生成中间渐变色
        now_color = color_rgb_start
        color_map.append(RGB_list_to_Hex(now_color))
        for color_index in range(1, color_sub_count):
            now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
            color_map.append(RGB_list_to_Hex(now_color))
        color_index_start = color_index_end
    return color_map



# 使用matplot画图
# 散点图→根据TFIDF上色→加入文本标签
def draw_scatter(data):
    input_colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#99004c"]
    length_rank = len(set(data["Rank"]))
    colors = gradient_color(input_colors, length_rank)
    if length_rank != len(colors):
        colors = gradient_color(input_colors, length_rank + 1)
    print("colors' length = ", len(colors))

    ax = plt.figure().add_subplot()
    # 依次画每个点
    length = len(data)
    X = data["X"]
    Y = data["RandomY"]
    Value = data["TFIDF"]
    Tag = data["taste"]
    Rank = data["Rank"]
    for i in range(length):
        current_x = X[i]
        current_y = Y[i]
        current_value = Value[i]
        current_tag = Tag[i]
        current_rank = Rank[i]
        current_color = colors[len(colors) - current_rank]
        ax.scatter(current_x, current_y, color=current_color, s=2000*current_value, alpha=0.75)

    # ax.scatter(data["X"], data["RandomY"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.patch.set_facecolor("#6495ED")

    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    path_global = "data/hot-words.xlsx"
    name = "drawing"
    data_global = pd.read_excel(path_global, sheet_name=name)
    draw_scatter(data_global)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


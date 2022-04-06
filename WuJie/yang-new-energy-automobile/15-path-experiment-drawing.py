import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as font_manager

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


# 计算中位数
def median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2


# import matplotlib;
# matplotlib.use('tkagg')
# plt.rcParams['font.sans-serif']=['SimHei']  # 指定默认字体 SimHei为黑体
# plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
from matplotlib.pyplot import MultipleLocator
# 画某款车的IPA数据，秦新能源
def drawing_route(path, name):
    font = {'family': 'Times New Roman', 'size': 12}
    font2 = {'family': 'Times New Roman', 'size': 18}
    font3 = font_manager.FontProperties(family='Times New Roman', style='normal', size=15)
    # colors = {"Appearance": "b", "Comfort": "g", "CP": "r", "EC": "c", "Handling": "m", "Interiors": "y", "Power": "k", "Space": "yellowgreen"}
    markers = {1: "o", 2: "*", 3: "s"}
    data = pd.read_excel(path, sheet_name=name, engine="openpyxl")
    aspects = set(data[name].tolist())

    figure = plt.figure(1)

    ax = figure.add_subplot(111)
    for aspect in aspects:
        current_data_aspect = data.loc[data[name] == aspect]
        x = current_data_aspect["平均值"].tolist()
        y = current_data_aspect["前三"].tolist()
        if len(x) <= 2:
            continue
        # 折线
        ax.plot(x, y, linewidth=1.0)
        # 箭头
        length = len(x)
        for j in range(length):
            if j == 0:
                continue
            start_x = (x[j] + x[j - 1])/2.0
            start_y = (y[j] + y[j - 1])/2.0
            plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", mutation_scale=20))
            # plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=colors.get(aspect), mutation_scale=20))
        # 散点
        # shape = current_data_aspect["Shape"].tolist()
        length = len(x)
        for j in range(length):
            current_x = x[j]
            current_y = y[j]
            # current_shape = markers.get(shape[j])
            ax.scatter(current_x, current_y, s=45, marker="o")
            # ax.scatter(current_x, current_y, s=45, marker=current_shape)
            # ax.scatter(current_x, current_y, color=colors.get(aspect), s=45, marker=current_shape)
        ax.set_xlabel("平均值", fontdict=font2)
        ax.set_ylabel("前三", fontdict=font2)

    # 画中位线
    min_x = min(data["平均值"].tolist()) - 0.02
    max_x = max(data["平均值"].tolist()) + 0.02
    median_x = median(data["平均值"].tolist())
    min_y = min(data["前三"].tolist()) - 0.2
    max_y = max(data["前三"].tolist()) + 0.2
    median_y = median(data["前三"].tolist())
    ax.plot([median_x, median_x], [min_y, max_y], color="black", linewidth=1.0)
    ax.plot([min_x, max_x], [median_y, median_y], color="black", linewidth=1.0)

    # 设置坐标轴范围
    ax.set_xlim(min_x + 0.02, max_x - 0.02)
    ax.set_ylim(min_y + 0.02, max_y - 0.02)

    # leg = plt.legend(aspects, loc="lower center", prop=font3, bbox_to_anchor=(1.21, 0.35), borderaxespad=0., markerscale=5)  # 设置图例
    # leg.get_frame().set_linewidth(0.0)  # 去掉图例边框
    plt.tight_layout()  # 调整子图之间距离
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    path = "data/路径实验.xlsx"
    sheet_names = ["插电式混合动力", "纯电动", "本土产进口品牌", "本土品牌", "进口"]
    for sheet_name in sheet_names:
        print("current sheet:", sheet_name)
        drawing_route(path, sheet_name)
        # break

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as font_manager
from haishoku.haishoku import Haishoku
from colormap import rgb2hex

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


def drawing(path, name):
    font = {'family': 'Times New Roman', 'size': 30}
    font2 = {'family': 'Times New Roman', 'size': 35}
    font3 = font_manager.FontProperties(size=25)
    data = pd.read_excel(path, sheet_name=name)
    aspects = list(set(data["Aspect"]))
    figure = plt.figure(1)

    # x = [2015, 2016, 2017, 2018, 2019]
    ax2 = figure.add_axes([0.1, 0.15, 0.7, 0.8])  # 无背景色
    # 设置边框颜色和宽度
    linewidth = 2
    ax2.spines['right'].set_color("black")
    ax2.spines['right'].set_linewidth(linewidth)
    ax2.spines['left'].set_color("black")
    ax2.spines['left'].set_linewidth(linewidth)
    ax2.spines['top'].set_color("black")
    ax2.spines['top'].set_linewidth(linewidth)
    ax2.spines['bottom'].set_color("black")
    ax2.spines['bottom'].set_linewidth(linewidth)

    legends = []
    for aspect in aspects:
        current_data_aspect = data.loc[data["Aspect"] == aspect]
        x = current_data_aspect["Competition advantage"].tolist()
        y = current_data_aspect["Attractiveness"].tolist()
        # print(aspect, ":", x)
        # print(x)
        # print(y)
        current_color = "#" + str(current_data_aspect["Color"].tolist()[0])  # 颜色
        # 折线
        # legends.append(ax2.plot(x, y, linewidth=2.5))
        legends.append(ax2.plot(x, y, color=current_color, linewidth=2.5))
        # 箭头
        length = len(x)
        for j in range(length):
            if j == 0:
                continue
            start_x = (x[j] + x[j - 1])/2.0
            start_y = (y[j] + y[j - 1])/2.0
            plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=current_color, mutation_scale=25))
            # plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color="black", mutation_scale=35))
        # 散点
        length = len(x)
        # print(current_data_aspect["Shape"])
        Shapes = current_data_aspect["Shape"].tolist()
        # print(Shapes)
        for j in range(length):
            current_shape = Shapes[j]
            # print(current_shape)
            if current_shape == "One-Dimension":
                current_shape = "o"
            elif current_shape == "Attractive":
                current_shape = "^"
            elif current_shape == "Basic":
                current_shape = "s"
            else:
                current_shape = "+"
            current_x = x[j]
            current_y = y[j]
            ax2.scatter(current_x, current_y, color=current_color, s=25, marker=current_shape)

    ax2.set_xlabel("Competition advantage", fontdict=font2)
    ax2.set_ylabel("Market attractiveness", fontdict=font2)

    # 画中位线
    min_x = min(data["Competition advantage"].tolist()) - 0.002
    max_x = max(data["Competition advantage"].tolist()) + 0.002
    print(min_x, max_x)
    min_y = min(data["Attractiveness"].tolist()) - 0.002
    max_y = max(data["Attractiveness"].tolist()) + 0.002
    median_y = median(data["Attractiveness"].tolist())
    print(median_y)
    ax2.plot([0, 0], [min_y, max_y], color="blue", linewidth=1.0)
    ax2.plot([min_x, max_x], [median_y, median_y], color="red", linewidth=1.0)
    # 上三分位数

    # 下三分位数

    # 设置坐标轴范围
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)

    # 设置坐标轴与刻度值的宽度
    ax2.tick_params(axis="y", pad=1)

    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False
    leg = plt.legend(aspects, loc="lower center", bbox_to_anchor=(1.15, -0.02), prop=font3, borderaxespad=0., markerscale=5)  # 设置图例
    leg.get_frame().set_linewidth(0.0)  # 去掉图例边框
    plt.tick_params(labelsize=25)  # 设置坐标轴刻度值大小
    # plt.subplots_adjust(wspace=2, hspace=2)
    # plt.tight_layout(h_pad=2.5, w_pad=2.5)  # 调整子图之间距离
    # plt.legend(["o", "*", "□"], ["1", "2", "3"], prop=font3, bbox_to_anchor=(1.24, 0.55), borderaxespad=0.)
    plt.show()


# 计算中位数
def median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    path_global = "data/GE矩阵.xlsx"
    table_name = "分年份-drawing"

    drawing(path_global, table_name)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


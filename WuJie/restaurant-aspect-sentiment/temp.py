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


def drawing():
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.array([0, 1, 2])
    Y = np.array([4, 5, 6])
    X, Y = np.meshgrid(X, Y)  # 形成二维的网格点
    Z = np.array([(4, 5, 6), (5, 6, 7), (6, 7, 8)])  # 计算Z=X+Y的值
    print("X:", X)
    print("Y:", Y)
    print("Z:", Z)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='#FF6347')  # 绘制曲面图
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm')  # 绘制曲面图
    # ax.contour(X, Y, Z, zdir='z', offset=4, cmap=plt.get_cmap('coolwarm'))  # 绘制投影（到Z=-2上)
    X = np.array([2, 1, 0])
    Y = np.array([6, 5, 4])
    X, Y = np.meshgrid(X, Y)  # 形成二维的网格点
    Z = np.array([(6, 7, 8), (5, 6, 7), (4, 5, 6)])  # 计算Z=X+Y的值
    print("X:", X)
    print("Y:", Y)
    print("Z:", Z)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='#1E90FF')  # 绘制曲面图

    plt.show()


# 读取数据绘图
def drawing2(path, name):
    restaurant = [1, 2]
    data = pd.read_excel(path, sheet_name=name)
    for rest in restaurant:
        print("正在画：", rest)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    path_global = "data/GE矩阵-20211217.xlsx"
    table_name = "分年份(16-19)-drawing2"

    drawing()
    # drawing2(path_global, table_name)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


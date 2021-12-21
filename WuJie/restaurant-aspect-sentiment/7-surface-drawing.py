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
    Z = np.array([(5, 6, 7)])  # 计算Z=X+Y的值
    # Z = np.array([(4, 5, 6), (5, 6, 7), (6, 7, 8)])  # 计算Z=X+Y的值
    print("X:", X)
    print("Y:", Y)
    print("Z:", Z)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='#FF6347')  # 绘制曲面图
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm')  # 绘制曲面图
    # ax.contour(X, Y, Z, zdir='z', offset=4, cmap=plt.get_cmap('coolwarm'))  # 绘制投影（到Z=-2上)
    X = np.array([2, 1, 0])
    Y = np.array([6, 5, 4])
    X, Y = np.meshgrid(X, Y)  # 形成二维的网格点
    Z = np.array([(6, 7, 8)])  # 计算Z=X+Y的值
    print("X:", X)
    print("Y:", Y)
    print("Z:", Z)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='#1E90FF')  # 绘制曲面图

    plt.show()


# 读取数据绘图
def drawing2(path, name):
    font = {'family': 'Times New Roman', 'size': 30}
    font2 = {'family': 'Times New Roman', 'size': 35}
    font3 = font_manager.FontProperties(size=30)
    fig = plt.figure()
    ax = Axes3D(fig)

    restaurant = [1, 2]
    data = pd.read_excel(path, sheet_name=name)
    for rest in restaurant:
        X = [1, 2, 3, 4]
        Y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        print("正在画：", rest)
        data_rest = data.loc[data["Restaurant"] == rest]
        # print(data_rest)

        # 生成Z值
        Z = []
        for x in X:
            temp = []
            for y in Y:
                z = data_rest.loc[(data_rest["TimeNo"] == x) & (data_rest["AspectNo"] == y)]["Attractiveness"].tolist()[0]
                # print("z:", z)
                # print("z's type:", type(z))
                temp.append(z)
            # print("temp:", temp)
            Z.append(temp)

        X = np.array(X)
        Y = np.array(Y)

        X, Y = np.meshgrid(X, Y)
        print(X.shape)
        print(Y.shape)
        Z = np.array(Z)
        Z = Z.T
        # print(Z)
        print(Z.shape)

        if rest == 1:
            color = "#FF6347"
            alpha = 1
        else:
            color = "#1E90FF"
            alpha = 0.5
        ax.plot_surface(X, Y, Z, rstride=64, cstride=64, color=color, antialiased=False, alpha=alpha)  # 绘制曲面图

    ax.set_xlabel("Time", font2, labelpad=12.5)
    ax.set_ylabel("Aspect", font2, labelpad=15.5)
    ax.set_zlabel("Performance", font2, labelpad=19.5, rotation=90)

    ax.tick_params(labelsize=25)

    ax.set_xticklabels(["1", "2", "3", "4"], fontdict=font)
    ax.set_xticks([1, 2, 3, 4])

    # ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"], fontdict=font)
    # ax.set_zticklabels(["0.4", "0.6", "0.8", "1"], fontdict=font)

    ax.tick_params(axis="z", pad=10)
    ax.tick_params(axis="y", pad=1.5)

    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    path_global = "data/GE矩阵-20211217.xlsx"
    table_name = "分年份(16-19)-drawing2"

    # drawing()
    drawing2(path_global, table_name)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


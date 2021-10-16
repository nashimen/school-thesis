import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib;matplotlib.use('tkagg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)


# 二维IPA
def drawing(data_path, name):
    data = pd.read_excel(data_path, sheet_name=name)
    hotel_types = ["Economy", "Midscale", "Luxury"]
    trip_types = ["Business", "Leisure"]
    attributes = list(set(data["Attributes"].tolist()))
    figure = plt.figure(1)

    # 画散点图
    i = 1
    for trip_type in trip_types:
        current_data_trip = data.loc[data["Trip type"] == trip_type]
        for hotel_type in hotel_types:
            current_data_hotel = current_data_trip.loc[current_data_trip["Hotel type"] == hotel_type]
            # print(current_data_hotel)
            quadrant = "23" + str(i)
            # print("quadrant = ", quadrant)
            ax = figure.add_subplot(quadrant)  # 2行3列 第一个
            i += 1
            # 画中位线
            min_x = min(current_data_hotel["Complaint index"].tolist()) - 0.01
            max_x = max(current_data_hotel["Complaint index"].tolist()) + 0.01
            median_x = median(current_data_hotel["Complaint index"].tolist())
            min_y = min(current_data_hotel["Importance"].tolist()) - 0.1
            max_y = max(current_data_hotel["Importance"].tolist()) + 0.1
            median_y = median(current_data_hotel["Importance"].tolist())
            ax.plot([median_x, median_x], [min_y, max_y], color="blue", linewidth=1.0)
            ax.plot([min_x, max_x], [median_y, median_y], color="red", linewidth=1.0)
            for attribute in attributes:
                xComplaintIndex = current_data_hotel.loc[current_data_hotel["Attributes"] == attribute]["Complaint index"]
                yImportance = current_data_hotel.loc[current_data_hotel["Attributes"] == attribute]["Importance"]
                marker = list(current_data_hotel.loc[current_data_hotel["Attributes"] == attribute]["Markers"])[0]
                current_marker = "." if str(marker) == "circle" else "*"
                ax.scatter(xComplaintIndex, yImportance, color="black", marker=current_marker)
                # 根据坐标位置设置ha和va
                ax.text(xComplaintIndex, yImportance - 0.02, attribute, ha="center", va="top", fontsize=18)
            ax.set_xlabel("Complaint Index", fontsize=20)
            ax.set_ylabel("Importance", fontsize=20)

            # 去掉边缘空白
            ax.margins(0, 0)
            ax.set_xlim(xmin=min_x, xmax=max_x)
            ax.set_ylim(ymin=min_y, ymax=max_y)

            # 坐标轴字体设置
            ax.tick_params(axis="x", labelsize=18)
            ax.tick_params(axis="y", labelsize=18)

    plt.show()


# 计算中位数
def median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    path = "data/IPA作图.xlsx"
    sheet_name = "data"
    drawing(path, sheet_name)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


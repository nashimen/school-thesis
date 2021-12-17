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

# 颜色提取器
def extract_color():
    image = ""
    haishoku = Haishoku.loadHaishoku(image)


from matplotlib.pyplot import MultipleLocator
# 二维&三维同时画
def drawing_2D_3D(path, name):
    font = {'family': 'Times New Roman', 'size': 12}
    font2 = {'family': 'Times New Roman', 'size': 18}
    font3 = font_manager.FontProperties(family='Times New Roman', style='normal', size=15)
    data = pd.read_excel(path, sheet_name=name)
    aspects = list(set(data["属性"]))
    figure = plt.figure(1)

    x = [2018, 2019, 2020, 2021]
    # 先画3D图
    quadrant = "121"
    ax = figure.add_subplot(quadrant, projection='3d')
    for aspect in aspects:
        current_data_aspect = data.loc[data["属性"] == aspect]
        y = current_data_aspect["GAP"].tolist()
        z = current_data_aspect["PD"].tolist()
        # print("y:", y)
        # print("z:", z)
        # 3D折线图
        current_color = "#" + str(current_data_aspect["Color"].tolist()[0])
        print("current_color:", current_color)
        ax.plot(x, y, z, linewidth=1.5, color=current_color, linestyle='-', label=aspect)  # 设置图例
        # 3D散点图
        length = len(y)
        for j in range(length):
            current_x = x[j]
            current_y = y[j]
            current_z = z[j]
            # ax.scatter(x, y, z, linewidth=1, color=colors.get(aspect), linestyle="-", label=aspect)  # 设置图例
            ax.plot(current_x, current_y, current_z, linewidth=0.5, marker='o', markeredgecolor=current_color, markerfacecolor=current_color, markersize=4, label=aspect)  # 设置图例

    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.zaxis.set_major_locator(MultipleLocator(0.01))

    ax.set_xlabel("Year", fontdict=font2, labelpad=85)
    ax.set_ylabel("GAP", fontdict=font2, labelpad=85)
    ax.set_zlabel("PD", fontdict=font2, labelpad=85)
    title = "(a)"
    print(title)
    ax.set_title(title, fontdict=font2)  # 设置标题

    ax.set_xticklabels(["Before 2019", "2019", "2020", "After 2021"], fontdict=font)
    ax.set_xticks([2018, 2019, 2020, 2021])
    ax.dist = 8.65

    # 再画2D图
    quadrant = "122"
    ax = figure.add_subplot(quadrant)

    legends = []
    for aspect in aspects:
        current_data_aspect = data.loc[data["属性"] == aspect]
        x = current_data_aspect["GAP"].tolist()
        y = current_data_aspect["PD"].tolist()
        current_color = "#" + str(current_data_aspect["Color"].tolist()[0])
        # 折线
        legends.append(ax.plot(x, y, color=current_color, linewidth=1.5))
        # 箭头
        length = len(x)
        for j in range(length):
            if j == 0:
                continue
            start_x = (x[j] + x[j - 1])/2.0
            start_y = (y[j] + y[j - 1])/2.0
            plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=current_color, mutation_scale=20))
            # plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=colors.get(aspect)))
        # 散点
        length = len(x)
        for j in range(length):
            current_x = x[j]
            current_y = y[j]
            ax.scatter(current_x, current_y, color=current_color, s=25, marker='o')

    ax.set_xlabel("GAP", fontdict=font2)
    ax.set_ylabel("PD", fontdict=font2)
    title = "(b)"
    ax.set_title(title, fontdict=font2)  # 设置标题
    # ax.set_title(title, fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 12, 'style': 'italic'})  # 设置标题
    # ax.legend(legends, aspects, loc="upper left")  # 设置图例

    # 画中位线
    min_x = min(data["GAP"].tolist()) - 0.002
    max_x = max(data["GAP"].tolist()) + 0.002
    min_y = min(data["PD"].tolist()) - 0.002
    max_y = max(data["PD"].tolist()) + 0.002
    median_y = median(data["PD"].tolist())
    ax.plot([0, 0], [min_y, max_y], color="black", linewidth=1.0)
    ax.plot([min_x, max_x], [0, 0], color="black", linewidth=1.0)

    # 设置坐标轴范围
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # 去掉边缘空白
    # plt.margins(0, 0)
    # plt.xlim(xmin=min_x, xmax=max_x)
    # plt.ylim(ymin=min_y, ymax=max_y)

    leg = plt.legend(aspects, loc="lower center", prop=font3, bbox_to_anchor=(1.21, 1), borderaxespad=0., markerscale=5)  # 设置图例
    leg.get_frame().set_linewidth(0.0)  # 去掉图例边框
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.tight_layout(h_pad=2.5, w_pad=2.5)  # 调整子图之间距离
    # plt.legend(["o", "*", "□"], ["1", "2", "3"], prop=font3, bbox_to_anchor=(1.24, 0.55), borderaxespad=0.)
    plt.show()


# 二维&三维同时画
def drawing_2D_3D_2(path, name):
    font = {'family': 'Times New Roman', 'size': 20}
    font2 = {'family': 'Times New Roman', 'size': 25}
    font3 = font_manager.FontProperties(size=25)
    data = pd.read_excel(path, sheet_name=name)
    aspects = list(set(data["属性"]))
    figure = plt.figure(1)

    x = [2018, 2019, 2020, 2021]
    # 先画3D图
    # quadrant = "121"
    # ax = figure.add_subplot(quadrant, projection='3d')
    ax1 = figure.add_axes([-0.1, 0.15, 0.65, 0.7], projection='3d')
    for aspect in aspects:
        current_data_aspect = data.loc[data["属性"] == aspect]
        y = current_data_aspect["PD"].tolist()
        z = current_data_aspect["GAP"].tolist()
        # print("y:", y)
        # print("z:", z)
        # 3D折线图
        current_color = "#" + str(current_data_aspect["Color"].tolist()[0])
        print("current_color:", current_color)
        ax1.plot(x, y, z, linewidth=1.5, color=current_color, linestyle='-', label=aspect)  # 设置图例
        # 3D散点图
        length = len(y)
        for j in range(length):
            current_x = x[j]
            current_y = y[j]
            current_z = z[j]
            # ax.scatter(x, y, z, linewidth=1, color=colors.get(aspect), linestyle="-", label=aspect)  # 设置图例
            ax1.plot(current_x, current_y, current_z, linewidth=0.5, marker='o', markeredgecolor=current_color, markerfacecolor=current_color, markersize=4, label=aspect)  # 设置图例

    # ax1.xaxis.set_major_locator(MultipleLocator(0.01))
    # ax1.yaxis.set_major_locator(MultipleLocator(0.03))
    # ax1.zaxis.set_major_locator(MultipleLocator(0.01))

    ax1.set_xlabel("Year", fontdict=font2, labelpad=15.5)
    ax1.set_ylabel("PD", fontdict=font2, labelpad=15.5)
    ax1.set_zlabel("GAP", fontdict=font2, labelpad=11.5, rotation=90)  # 设置轴标签与轴的距离

    ax1.tick_params(labelsize=15)

    title = "(a)"
    print(title)
    ax1.set_title(title, fontdict=font2)  # 设置标题

    ax1.set_xticklabels(["Before 2019", "2019", "2020", "After 2021"], fontdict=font)
    ax1.set_xticks([2018, 2019, 2020, 2021])
    ax1.dist = 8.65

    # 再画2D图
    # quadrant = "122"
    # ax = figure.add_subplot(quadrant)
    ax2 = figure.add_axes([0.52, 0.17, 0.35, 0.67], facecolor='#f2f2f2')

    legends = []
    for aspect in aspects:
        current_data_aspect = data.loc[data["属性"] == aspect]
        x = current_data_aspect["PD"].tolist()
        y = current_data_aspect["GAP"].tolist()
        current_color = "#" + str(current_data_aspect["Color"].tolist()[0])
        # 折线
        legends.append(ax2.plot(x, y, color=current_color, linewidth=1))
        # 箭头
        length = len(x)
        for j in range(length):
            if j == 0:
                continue
            start_x = (x[j] + x[j - 1])/2.0
            start_y = (y[j] + y[j - 1])/2.0
            plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=current_color, mutation_scale=20))
            # plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=colors.get(aspect)))
        # 散点
        length = len(x)
        for j in range(length):
            current_x = x[j]
            current_y = y[j]
            ax2.scatter(current_x, current_y, color=current_color, s=25, marker='o')

    ax2.set_xlabel("PD", fontdict=font2)
    ax2.set_ylabel("GAP", fontdict=font2)
    # ax2.set_ylabel("GAP", fontdict=font2, rotation=0, ha="right")
    title = "(b)"
    ax2.set_title(title, fontdict=font2)  # 设置标题
    # ax.set_title(title, fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 12, 'style': 'italic'})  # 设置标题
    # ax.legend(legends, aspects, loc="upper left")  # 设置图例

    # 画中位线
    min_x = min(data["PD"].tolist()) - 0.002
    max_x = max(data["PD"].tolist()) + 0.002
    min_y = min(data["GAP"].tolist()) - 0.002
    max_y = max(data["GAP"].tolist()) + 0.002
    median_y = median(data["PD"].tolist())
    ax2.plot([0, 0], [min_y, max_y], color="blue", linewidth=1.0)
    ax2.plot([min_x, max_x], [0, 0], color="red", linewidth=1.0)

    # 设置坐标轴范围
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)

    # 去掉边缘空白
    # plt.margins(0, 0)
    # plt.xlim(xmin=min_x, xmax=max_x)
    # plt.ylim(ymin=min_y, ymax=max_y)

    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False
    leg = plt.legend(aspects, loc="lower center", bbox_to_anchor=(1.2, 0.13), prop=font3, borderaxespad=0., markerscale=5)  # 设置图例
    leg.get_frame().set_linewidth(0.0)  # 去掉图例边框
    plt.tick_params(labelsize=15)  # 设置坐标轴刻度值大小
    # plt.subplots_adjust(wspace=2, hspace=2)
    # plt.tight_layout(h_pad=2.5, w_pad=2.5)  # 调整子图之间距离
    # plt.legend(["o", "*", "□"], ["1", "2", "3"], prop=font3, bbox_to_anchor=(1.24, 0.55), borderaxespad=0.)
    plt.show()


# 二维&三维同时画
def drawing_2D_3D_3(path, name):
    font = {'family': 'Times New Roman', 'size': 20}
    font2 = {'family': 'Times New Roman', 'size': 25}
    font3 = font_manager.FontProperties(size=25)
    data = pd.read_excel(path, sheet_name=name)
    aspects = list(set(data["属性"]))
    figure = plt.figure(1)

    x = [2019, 2020, 2021]
    # 先画3D图
    # quadrant = "121"
    # ax = figure.add_subplot(quadrant, projection='3d')
    ax1 = figure.add_axes([-0.1, 0.15, 0.65, 0.7], projection='3d')
    # ax1.grid(False)  # 去掉背景的网格线
    for aspect in aspects:
        current_data_aspect = data.loc[data["属性"] == aspect]
        y = current_data_aspect["PD"].tolist()
        z = current_data_aspect["GAP"].tolist()
        # print("y:", y)
        # print("z:", z)
        # 3D折线图
        current_color = "#" + str(current_data_aspect["Color"].tolist()[0])
        print("current_color:", current_color)
        ax1.plot(x, y, z, linewidth=1.5, color=current_color, linestyle='-', label=aspect)  # 设置图例
        # 3D散点图
        length = len(y)
        for j in range(length):
            current_x = x[j]
            current_y = y[j]
            current_z = z[j]
            # ax.scatter(x, y, z, linewidth=1, color=colors.get(aspect), linestyle="-", label=aspect)  # 设置图例
            ax1.plot(current_x, current_y, current_z, linewidth=0.5, marker='o', markeredgecolor=current_color, markerfacecolor=current_color, markersize=4, label=aspect)  # 设置图例

    # ax1.xaxis.set_major_locator(MultipleLocator(0.01))
    # ax1.yaxis.set_major_locator(MultipleLocator(0.03))
    # ax1.zaxis.set_major_locator(MultipleLocator(0.01))

    ax1.set_xlabel("Year", fontdict=font2, labelpad=15.5)
    ax1.set_ylabel("PD", fontdict=font2, labelpad=15.5)
    ax1.set_zlabel("GAP", fontdict=font2, labelpad=11.5, rotation=90)  # 设置轴标签与轴的距离

    ax1.tick_params(labelsize=15)

    title = "(a)"
    print(title)
    ax1.set_title(title, fontdict=font2)  # 设置标题

    ax1.set_xticklabels(["Before 2020", "2020", "2021"], fontdict=font)
    ax1.set_xticks([2019, 2020, 2021])
    ax1.dist = 8.65

    # 再画2D图
    # quadrant = "122"
    # ax = figure.add_subplot(quadrant)
    ax2 = figure.add_axes([0.52, 0.18, 0.35, 0.65])  # 无背景色
    # ax2 = figure.add_axes([0.52, 0.17, 0.35, 0.67], facecolor='#f2f2f2')
    # 设置边框颜色和宽度
    linewidth = 1.5
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
        current_data_aspect = data.loc[data["属性"] == aspect]
        x = current_data_aspect["PD"].tolist()
        y = current_data_aspect["GAP"].tolist()
        current_color = "#" + str(current_data_aspect["Color"].tolist()[0])
        # 折线
        legends.append(ax2.plot(x, y, color=current_color, linewidth=1))
        # 箭头
        length = len(x)
        for j in range(length):
            if j == 0:
                continue
            start_x = (x[j] + x[j - 1])/2.0
            start_y = (y[j] + y[j - 1])/2.0
            plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=current_color, mutation_scale=20))
            # plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=colors.get(aspect)))
        # 散点
        length = len(x)
        for j in range(length):
            current_x = x[j]
            current_y = y[j]
            ax2.scatter(current_x, current_y, color=current_color, s=25, marker='o')

    ax2.set_xlabel("PD", fontdict=font2)
    ax2.set_ylabel("GAP", fontdict=font2)
    # ax2.set_ylabel("GAP", fontdict=font2, rotation=0, ha="right")
    title = "(b)"
    ax2.set_title(title, fontdict=font2)  # 设置标题
    # ax.set_title(title, fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 12, 'style': 'italic'})  # 设置标题
    # ax.legend(legends, aspects, loc="upper left")  # 设置图例

    # 画中位线
    min_x = min(data["PD"].tolist()) - 0.002
    max_x = max(data["PD"].tolist()) + 0.002
    min_y = min(data["GAP"].tolist()) - 0.002
    max_y = max(data["GAP"].tolist()) + 0.002
    median_y = median(data["PD"].tolist())
    ax2.plot([0, 0], [min_y, max_y], color="blue", linewidth=1.0)
    ax2.plot([min_x, max_x], [0, 0], color="red", linewidth=1.0)

    # 设置坐标轴范围
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)

    # 去掉边缘空白
    # plt.margins(0, 0)
    # plt.xlim(xmin=min_x, xmax=max_x)
    # plt.ylim(ymin=min_y, ymax=max_y)

    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False
    leg = plt.legend(aspects, loc="lower center", bbox_to_anchor=(1.2, 0.13), prop=font3, borderaxespad=0., markerscale=5)  # 设置图例
    leg.get_frame().set_linewidth(0.0)  # 去掉图例边框
    plt.tick_params(labelsize=15)  # 设置坐标轴刻度值大小
    # plt.subplots_adjust(wspace=2, hspace=2)
    # plt.tight_layout(h_pad=2.5, w_pad=2.5)  # 调整子图之间距离
    # plt.legend(["o", "*", "□"], ["1", "2", "3"], prop=font3, bbox_to_anchor=(1.24, 0.55), borderaxespad=0.)
    plt.show()


# 二维&三维同时画-示例
def drawing_2D_3D_4(path, name):
    font = {'family': 'Times New Roman', 'size': 30}
    font2 = {'family': 'Times New Roman', 'size': 35}
    font3 = font_manager.FontProperties(size=35)
    data = pd.read_excel(path, sheet_name=name)
    aspects = list(set(data["属性"]))
    figure = plt.figure(1)

    x = [2019, 2020, 2021]
    # 先画3D图
    # quadrant = "121"
    # ax = figure.add_subplot(quadrant, projection='3d')
    ax1 = figure.add_axes([-0.1, 0.15, 0.65, 0.7], projection='3d')
    # ax1.grid(False)  # 去掉背景的网格线
    for aspect in aspects:
        current_data_aspect = data.loc[data["属性"] == aspect]
        y = current_data_aspect["PD"].tolist()
        z = current_data_aspect["GAP"].tolist()
        # print("y:", y)
        # print("z:", z)
        # 3D折线图
        current_color = "#" + str(current_data_aspect["Color"].tolist()[0])
        print("current_color:", current_color)
        ax1.plot(x, y, z, linewidth=1.5, color=current_color, linestyle='-', label=aspect)  # 设置图例
        # 3D散点图
        length = len(y)
        for j in range(length):
            current_x = x[j]
            current_y = y[j]
            current_z = z[j]
            # ax.scatter(x, y, z, linewidth=1, color=colors.get(aspect), linestyle="-", label=aspect)  # 设置图例
            ax1.plot(current_x, current_y, current_z, linewidth=0.5, marker='o', markeredgecolor=current_color, markerfacecolor=current_color, markersize=4, label=aspect)  # 设置图例

    # ax1.xaxis.set_major_locator(MultipleLocator(0.01))
    # ax1.yaxis.set_major_locator(MultipleLocator(0.03))
    # ax1.zaxis.set_major_locator(MultipleLocator(0.01))

    ax1.set_xlabel("Time period", fontdict=font2, labelpad=17.5)
    ax1.set_ylabel("PD", fontdict=font2, labelpad=22.5)
    ax1.set_zlabel("GAP", fontdict=font2, labelpad=27.5, rotation=90)  # 设置轴标签与轴的距离

    ax1.tick_params(labelsize=30)
    ax1.tick_params(axis="z", pad=15)

    # title = "(a)"
    # print(title)
    # ax1.set_title(title, fontdict=font2)  # 设置标题

    ax1.set_xticklabels(["1", "2", "3"], fontdict=font)
    ax1.set_xticks([2019, 2020, 2021])
    ax1.dist = 8.65

    # 再画2D图
    # quadrant = "122"
    # ax = figure.add_subplot(quadrant)
    ax2 = figure.add_axes([0.61, 0.18, 0.35, 0.64])  # 无背景色
    # ax2 = figure.add_axes([0.52, 0.17, 0.35, 0.67], facecolor='#f2f2f2')
    # 设置边框颜色和宽度
    linewidth = 1.5
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
        current_data_aspect = data.loc[data["属性"] == aspect]
        x = current_data_aspect["PD"].tolist()
        y = current_data_aspect["GAP"].tolist()
        current_color = "#" + str(current_data_aspect["Color"].tolist()[0])
        # 折线
        legends.append(ax2.plot(x, y, color=current_color, linewidth=1))
        # 箭头
        length = len(x)
        for j in range(length):
            if j == 0:
                continue
            start_x = (x[j] + x[j - 1])/2.0
            start_y = (y[j] + y[j - 1])/2.0
            plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=current_color, mutation_scale=35))
            # plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=colors.get(aspect)))
        # 散点
        length = len(x)
        for j in range(length):
            current_x = x[j]
            current_y = y[j]
            ax2.scatter(current_x, current_y, color=current_color, s=30, marker='o')

    ax2.set_xlabel("PD", fontdict=font2)
    ax2.set_ylabel("GAP", fontdict=font2)
    # ax2.set_ylabel("GAP", fontdict=font2, rotation=0, ha="right")
    # title = "(b)"
    # ax2.set_title(title, fontdict=font2)  # 设置标题
    # ax.set_title(title, fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 12, 'style': 'italic'})  # 设置标题
    # ax.legend(legends, aspects, loc="upper left")  # 设置图例

    # 画中位线
    min_x = min(data["PD"].tolist()) - 0.002
    max_x = max(data["PD"].tolist()) + 0.002
    min_y = min(data["GAP"].tolist()) - 0.002
    max_y = max(data["GAP"].tolist()) + 0.002
    median_y = median(data["PD"].tolist())
    ax2.plot([0, 0], [min_y, max_y], color="blue", linewidth=1.0)
    ax2.plot([min_x, max_x], [0, 0], color="red", linewidth=1.0)

    # 设置坐标轴范围
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)

    # 设置坐标轴与刻度值的宽度
    ax2.tick_params(axis="y", pad=1)

    # 去掉边缘空白
    # plt.margins(0, 0)
    # plt.xlim(xmin=min_x, xmax=max_x)
    # plt.ylim(ymin=min_y, ymax=max_y)

    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False
    # leg = plt.legend(aspects, loc="lower center", bbox_to_anchor=(1.2, 0.13), prop=font3, borderaxespad=0., markerscale=5)  # 设置图例
    # leg.get_frame().set_linewidth(0.0)  # 去掉图例边框
    plt.tick_params(labelsize=30)  # 设置坐标轴刻度值大小
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

    path_global = "data/属性情感-汇总后-v3（示例）.xlsx"
    table_name = "drawing"

    drawing_2D_3D_4(path_global, table_name)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


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


# 三维折线图
def drawing_broken_line_graph():
    theta = numpy.linspace(0, 3.14*4, 200)
    r = numpy.linspace(0, 1, 200)
    x = r * numpy.cos(theta)
    y = r * numpy.sin(theta)
    z = numpy.linspace(0, 2, 200)

    performance = [4.738, 4.839, 4.875, 4.898, 4.917]
    importance = [3.917, 3.949, 3.784, 3.612, 3.615]
    year = [2017, 2018, 2019, 2020, 2021]
    # year = ["2017", "2018", "2019", "2020", "2021"]

    # fig = plt.figure(figsize=(12, 7))
    plt.xlabel("Performance")
    plt.ylabel("Importance")
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax1 = plt.axes(projection='3d')
    ax1.plot(performance, year, importance)

    performance = [4.38, 4.39, 4.5, 4.98, 4.17]
    importance = [3, 3, 4, 3.6, 3.15]
    year = [2017, 2018, 2019, 2020, 2021]
    # year = ["2017", "2018", "2019", "2020", "2021"]

    # fig = plt.figure(figsize=(12, 7))
    ax1 = plt.axes(projection='3d')
    ax1.plot(performance, year, importance)
    # ax1.plot(x, y, z)
    plt.show()


def drawing_v2():
    ax = plt.figure().add_subplot(projection='3d')

    # Plot a sin curve using the x and y axes.
    x = [2017, 2018, 2019, 2020, 2021]
    y = [4.38, 4.39, 4.5, 4.98, 4.17]
    ax.plot(x, y, zs=[3, 3, 4, 3.6, 3.15], zdir='z', label='curve in (x, y)')
    x2 = [2017, 2018, 2019, 2020, 2021]
    y2 = [4.738, 4.839, 4.875, 4.898, 4.917]
    ax.plot(x2, y2, zs=[3.917, 3.949, 3.784, 3.612, 3.615], zdir='z')
    # ax.plot(x2, y2, zs=[3.917, 3.949, 3.784, 3.612, 3.615], zdir='z')

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
    ax.set_xlabel("Year")
    ax.set_ylabel("Performance")
    ax.set_zlabel("Importance")
    plt.show()


def drawing_v3(path=None):
    # marker
    markers = {"Appearance": "d", "Comfort": "^", "CP": "s", "EC": "d", "Handling": "s", "Interiors": "^", "Power": ".", "Space": "."}
    styles = {"Appearance": "--", "Comfort": "-", "CP": "-", "EC": "-", "Handling": "--", "Interiors": "--", "Power": "--", "Space": "-"}
    colors = {"Appearance": "b", "Comfort": "g", "CP": "r", "EC": "c", "Handling": "m", "Interiors": "y", "Power": "k", "Space": "yellowgreen"}
    current_sheet = "Electric"
    data = pd.read_excel(path, sheet_name=current_sheet)
    print(data.head())
    aspects = markers.keys()
    # x = ["Before 2018", 2018, 2019, 2020, 2021]
    x = [2017, 2018, 2019, 2020, 2021]

    ax = plt.figure().add_subplot(projection='3d')
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        y = current_data["Performance"]
        z = current_data["Importance"]
        ax.plot(x, y, z, linewidth=0.5, color=colors.get(aspect), marker=markers.get(aspect), markersize=3.5, linestyle=styles.get(aspect), label=aspect)  # 设置图例

    # ax.legend(loc='lower left')

    '''
    y = [4.38, 4.39, 4.5, 4.98, 4.17]
    z = [3, 3, 4, 3.6, 3.15]
    ax.plot(x, y, z, linewidth=1, color='C1', marker='o', markersize=3, linestyle='-', label='line')  # 设置图例

    y2 = [4.738, 4.839, 4.875, 4.898, 4.917]
    z2 = [3.917, 3.949, 3.784, 3.612, 3.615]
    ax.plot(x, y2, z2, linewidth=1, color='green', marker='^', markersize=3, linestyle='-.', label='line2')
    '''

    ax.dist = 9.5

    # 设置标签
    ax.set_xlabel("Year")
    ax.set_ylabel("Performance")
    ax.set_zlabel("Importance")

    # 设置刻度
    # x = ["Before 2018", "2018", "2019", "2020", "2021"]
    # ax.set_xticklabels(["Before2018", "2018", "2019", "2020", "2021"])
    ax.set_xticks([2017, 2018, 2019, 2020, 2021])
    # ax.set_yticks([3.25, 4.15, 5.0])
    # ax.set_zticks([3.25, 4.15, 5.0])

    plt.show()


def drawing_v4(path, name):
    colors = {"Appearance": "b", "Comfort": "g", "CP": "r", "EC": "c", "Handling": "m", "Interiors": "y", "Power": "k", "Space": "yellowgreen"}
    data = pd.read_excel(path, sheet_name=name)
    aspects = colors.keys()
    x = [2017, 2018, 2019, 2020, 2021]

    ax = plt.figure().add_subplot(projection='3d')
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        y = current_data["Performance"]
        z = current_data["Importance"]
        ax.plot(x, y, z, linewidth=0.5, color=colors.get(aspect), marker=markers.get(aspect), markersize=3.5, linestyle=styles.get(aspect), label=aspect)  # 设置图例


import numpy as np
import matplotlib.pyplot as plt


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    print("paths = ", paths)
    return sc



def plotMatrixPoint(Mat, Label):
    """
    输入二维点矩阵和标签，能够改变不同形状
    :param Mat:
    :param Label:
    :return:
    """
    x = Mat[:, 0]
    y = Mat[:, 1]
    map_size = {-1: 50, 1: 100}
    size = list(map(lambda x: map_size[x], Label))
    map_color = {-1: 'r', 1: 'g'}
    color = list(map(lambda x: map_color[x], Label))
    map_marker = {-1: 'o', 1: 's'}
    markers = list(map(lambda x: map_marker[x], Label))
    print("markers:", markers)
    mscatter(np.array(x), np.array(y), s=size, c=color, m=markers)  # scatter函数只支持array类型数据
    plt.show()


def loadSimpData():
    datMat = np.matrix([[1., 2.1], [1.5, 1.6], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    plotMatrixPoint(datMat, classLabels)
    return datMat, classLabels


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import matplotlib;matplotlib.use('tkagg')
plt.rcParams['font.sans-serif']=['SimHei']  # 指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
# 三维散点图
def scatter3D():
    print("scatter3D...")
    # 数据1
    data1 = np.arange(24).reshape((8, 3))
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    z1 = data1[:, 2]

    # 数据2
    data2 = np.random.randint(0, 23, (6, 3))
    x2 = data2[:, 0]
    y2 = data2[:, 1]
    z2 = data2[:, 2]

    ax = plt.figure().add_subplot(projection='3d')

    # fig = plt.figure()
    # ax = Axes3D(fig)
    ax.scatter(x1, y1, z1, c='r', label="顺序点")
    ax.scatter(x2, y2, z2, c='g', label="随机点")

    ax.legend(loc="best")

    ax.set_zlabel("Z", fontdict={"size": 15, "color": "red"})
    ax.set_ylabel("Y", fontdict={"size": 15, "color": "red"})
    ax.set_xlabel("X", fontdict={"size": 15, "color": "red"})

    plt.show()


# 先画散点图再画折线图
def drawing_scatter_broken_line(path, name):
    colors = {"Appearance": "b", "Comfort": "g", "CP": "r", "EC": "c", "Handling": "m", "Interiors": "y", "Power": "k", "Space": "gray"}
    markers = {1: "o", 2: "*", 3: "s"}
    data = pd.read_excel(path, sheet_name=name)
    aspects = colors.keys()
    x = [2017, 2018, 2019, 2020, 2021]

    ax = plt.figure().add_subplot(projection='3d')
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        y = current_data["Performance"]
        z = current_data["Importance"]
        ax.plot(x, y, z, linewidth=1, color=colors.get(aspect), linestyle='-', label=aspect)  # 设置图例

    plt.legend(loc="upper right")

    # markers = ["o", "x", "^", "s", "d"]

    # 画散点图
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        length = len(x)
        # print("length = ", length)
        y = current_data["Performance"].tolist()
        z = current_data["Importance"].tolist()
        shape = current_data["Shape"].tolist()
        for i in range(length):
            # current_shape = shape[i]
            current_x = x[i]
            current_y = y[i]
            current_z = z[i]
            current_shape = markers.get(shape[i])
            # print(current_x, current_y, current_z, current_marker)
            ax.plot(current_x, current_y, current_z, linewidth=0.5, marker=current_shape, markeredgecolor=colors.get(aspect), markerfacecolor=colors.get(aspect), markersize=4, label=aspect)  # 设置图例
        # print("*" * 50)

    ax.dist = 9.5

    # 设置标签
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Performance", fontsize=16)
    ax.set_zlabel("Importance", fontsize=16)

    ax.set_xticklabels(["Before 2018", "2018", "2019", "2020", "2021"])
    ax.set_xticks([2017, 2018, 2019, 2020, 2021])

    plt.show()


# 二维图
def drawing_2d(path, name):
    colors = {"Appearance": "b", "Comfort": "g", "CP": "r", "EC": "c", "Handling": "m", "Interiors": "y", "Power": "k", "Space": "gray"}
    markers = {1: "o", 2: "*", 3: "s"}
    data = pd.read_excel(path, sheet_name=name)
    aspects = colors.keys()
    ax = plt.figure().add_subplot()

    # 画折线图
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        x = current_data["Performance"].tolist()
        y = current_data["Importance"].tolist()
        ax.plot(x, y, color=colors.get(aspect), linewidth=1.0)

    # 画箭头
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        x = current_data["Performance"].tolist()
        y = current_data["Importance"].tolist()
        length = len(x)
        for i in range(length):
            if i == 0:
                continue
            start_x = (x[i] + x[i - 1])/2.0
            start_y = (y[i] + y[i - 1])/2.0
            plt.annotate("", xy=(x[i - 1], y[i - 1]), xytext=(start_x, start_y), arrowprops=dict(arrowstyle="<-", color=colors.get(aspect)))

    # 画散点图
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        x = current_data["Performance"].tolist()
        y = current_data["Importance"].tolist()
        shape = current_data["Shape"].tolist()
        length = len(x)
        for i in range(length):
            current_x = x[i]
            current_y = y[i]
            current_shape = markers.get(shape[i])
            ax.scatter(current_x, current_y, color=colors.get(aspect), s=17, marker=current_shape)

    ax.set_xlabel("Performance", fontsize=16)
    ax.set_ylabel("Importance", fontsize=16)

    # 画中位线
    min_x = min(data["Performance"].tolist()) - 0.01
    max_x = max(data["Performance"].tolist()) + 0.01
    median_x = median(data["Performance"].tolist())
    min_y = min(data["Importance"].tolist()) - 0.1
    # max_y = (max_x - min_x + min_y)
    max_y = max(data["Importance"].tolist()) + 0.1
    median_y = median(data["Importance"].tolist())
    ax.plot([median_x, median_x], [min_y, max_y], color="black", linewidth=1.0)
    ax.plot([min_x, max_x], [median_y, median_y], color="black", linewidth=1.0)

    # 去掉边缘空白
    plt.margins(0, 0)
    plt.xlim(xmin=min_x, xmax=max_x)
    plt.ylim(ymin=min_y, ymax=max_y)

    plt.show()


# 计算中位数
def median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2


# 二维三维图一起画
def drawing_together(path, name):
    colors = {"Appearance": "b", "Comfort": "g", "CP": "r", "EC": "c", "Handling": "m", "Interiors": "y", "Power": "k", "Space": "gray"}
    markers = {1: "o", 2: "^", 3: "s"}
    data = pd.read_excel(path, sheet_name=name)
    aspects = colors.keys()
    figure = plt.figure(1)
    ax = figure.add_subplot(121)
    # 画折线图
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        x = current_data["Performance"].tolist()
        y = current_data["Importance"].tolist()
        ax.plot(x, y, color=colors.get(aspect), linewidth=1.0)

    # 画箭头
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        x = current_data["Performance"].tolist()
        y = current_data["Importance"].tolist()
        length = len(x)
        for i in range(length):
            if i == 0:
                continue
            start_x = (x[i] + x[i - 1])/2.0
            start_y = (y[i] + y[i - 1])/2.0
            plt.annotate("", xy=(x[i - 1], y[i - 1]), xytext=(start_x, start_y), arrowprops=dict(arrowstyle="<-", color=colors.get(aspect)))

    # 画散点图
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        x = current_data["Performance"].tolist()
        y = current_data["Importance"].tolist()
        shape = current_data["Shape"].tolist()
        length = len(x)
        for i in range(length):
            current_x = x[i]
            current_y = y[i]
            current_shape = markers.get(shape[i])
            ax.scatter(current_x, current_y, color=colors.get(aspect), s=10, marker=current_shape)

    ax.set_xlabel("Performance")
    ax.set_ylabel("Importance")

    # 画中位线
    min_x = min(data["Performance"].tolist()) - 0.01
    max_x = max(data["Performance"].tolist()) + 0.01
    median_x = median(data["Performance"].tolist())
    min_y = min(data["Importance"].tolist()) - 0.1
    max_y = max(data["Importance"].tolist()) + 0.1
    median_y = median(data["Importance"].tolist())
    ax.plot([median_x, median_x], [min_y, max_y], color="black", linewidth=1.0)
    ax.plot([min_x, max_x], [median_y, median_y], color="black", linewidth=1.0)

    # 去掉边缘空白
    plt.margins(0, 0)
    plt.xlim(xmin=min_x, xmax=max_x)
    plt.ylim(ymin=min_y, ymax=max_y)

    # 三维图
    ax2 = figure.add_subplot(122, projection='3d')
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        y = current_data["Performance"]
        z = current_data["Importance"]
        ax2.plot(x, y, z, linewidth=1, color=colors.get(aspect), linestyle='-', label=aspect)  # 设置图例

    # 画散点图
    for aspect in aspects:
        current_data = data.loc[data["Aspect"] == aspect]
        length = len(x)
        y = current_data["Performance"].tolist()
        z = current_data["Importance"].tolist()
        shape = current_data["Shape"].tolist()
        for i in range(length):
            # current_shape = shape[i]
            current_x = x[i]
            current_y = y[i]
            current_z = z[i]
            current_shape = markers.get(shape[i])
            ax2.plot(current_x, current_y, current_z, linewidth=0.5, marker=current_shape, markeredgecolor=colors.get(aspect), markerfacecolor=colors.get(aspect), markersize=4, label=aspect)  # 设置图例

    ax2.dist = 9.5

    # 设置标签
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Performance")
    ax2.set_zlabel("Importance")

    ax2.set_xticklabels(["Before 2018", "2018", "2019", "2020", "2021"])
    ax2.set_xticks([2017, 2018, 2019, 2020, 2021])

    plt.show()


def drawing_together_v2(path, name):
    t=np.arange(0.0,2.0,0.1)
    s=np.sin(t*np.pi)

    plt.figure(figsize=(8,8), dpi=80)
    plt.figure(1)
    ax1 = plt.subplot(121)
    ax1.plot(t,s, color="r",linestyle = "--")
    ax2 = plt.subplot(122)
    ax2.plot(t,s,color="y",linestyle = "-")
    # ax3 = plt.subplot(223)
    # ax3.plot(t,s,color="g",linestyle = "-.")
    # ax4 = plt.subplot(224)
    # ax4.plot(t,s,color="b",linestyle = ":")


# v2-价格&购车目的作图，二维&三维同时画
def drawing_2D_3D(path, name):
    font = {'family': 'Times New Roman', 'size': 12}
    font2 = {'family': 'Times New Roman', 'size': 18}
    font3 = font_manager.FontProperties(family='Times New Roman', style='normal', size=15)
    colors = {"Appearance": "b", "Comfort": "g", "CP": "r", "EC": "c", "Handling": "m", "Interiors": "y", "Power": "k", "Space": "yellowgreen"}
    markers = {1: "o", 2: "*", 3: "s"}
    data = pd.read_excel(path, sheet_name=name)
    aspects = colors.keys()
    figure = plt.figure(1)

    x = [2017, 2018, 2019, 2020]
    if name == "Purpose":
        types = ["工作", "生活", "综合"]
    else:
        # types = ["Less than 200,000RMB", "200,000 to 500,000RMB", "More than 500,000RMB"]
        types = [0, 1, 2]
    i = 1
    # 先画3D图
    for current_type in types:
        quadrant = "23" + str(i)
        ax = figure.add_subplot(quadrant, projection='3d')
        current_data_purpose = data.loc[data[name] == current_type]
        for aspect in aspects:
            current_data_aspect = current_data_purpose.loc[current_data_purpose["Aspect"] == aspect]
            y = current_data_aspect["Performance"].tolist()
            z = current_data_aspect["Importance"].tolist()
            # print("y:", y)
            # print("z:", z)
            # 3D折线图
            ax.plot(x, y, z, linewidth=1, color=colors.get(aspect), linestyle='-', label=aspect)  # 设置图例
            # 3D散点图
            shape = current_data_aspect["Shape"].tolist()
            length = len(y)
            for j in range(length):
                current_x = x[j]
                current_y = y[j]
                current_z = z[j]
                current_shape = markers.get(shape[j])
                # ax.scatter(x, y, z, linewidth=1, color=colors.get(aspect), linestyle="-", label=aspect)  # 设置图例
                ax.plot(current_x, current_y, current_z, linewidth=0.5, marker=current_shape, markeredgecolor=colors.get(aspect), markerfacecolor=colors.get(aspect), markersize=6, label=aspect)  # 设置图例

        ax.set_xlabel("Year", fontdict=font2)
        ax.set_ylabel("Performance", fontdict=font2)
        ax.set_zlabel("Importance", fontdict=font2)
        if name == "Purpose":
            title = "(a)-Work needs" if i == 1 else ("(b)-Living needs" if i == 2 else "(c)-General needs")
        else:
            title = "(a)-Less than 200,000RMB" if i == 1 else ("(b)-200,000~500,000RMB" if i == 2 else "(c)-More than 500,000RMB")
        print(i, title)
        ax.set_title(title, fontdict=font2)  # 设置标题

        ax.set_xticklabels(["Before 2018", "2018", "2019", "After 2019"], fontdict=font)
        ax.set_xticks([2017, 2018, 2019, 2020])
        ax.dist = 8.65
        i += 1

    # 再画2D图
    for current_type in types:
        quadrant = "23" + str(i)
        ax = figure.add_subplot(quadrant)
        current_data_purpose = data.loc[data[name] == current_type]
        legends = []
        for aspect in aspects:
            current_data_aspect = current_data_purpose.loc[current_data_purpose["Aspect"] == aspect]
            x = current_data_aspect["Performance"].tolist()
            y = current_data_aspect["Importance"].tolist()
            # 折线
            legends.append(ax.plot(x, y, color=colors.get(aspect), linewidth=1.0))
            # 箭头
            length = len(x)
            for j in range(length):
                if j == 0:
                    continue
                start_x = (x[j] + x[j - 1])/2.0
                start_y = (y[j] + y[j - 1])/2.0
                plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=colors.get(aspect), mutation_scale=20))
                # plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=colors.get(aspect)))
            # 散点
            shape = current_data_aspect["Shape"].tolist()
            length = len(x)
            for j in range(length):
                current_x = x[j]
                current_y = y[j]
                current_shape = markers.get(shape[j])
                ax.scatter(current_x, current_y, color=colors.get(aspect), s=45, marker=current_shape)
        ax.set_xlabel("Performance", fontdict=font2)
        ax.set_ylabel("Importance", fontdict=font2)
        if name == "Purpose":
            title = "(d)-Work needs" if i == 4 else ("(e)-Living needs" if i == 5 else "(f)-General needs")
        else:
            title = "(d)-Less than 200,000RMB" if i == 4 else ("(e)-200,000~500,000RMB" if i == 5 else "(f)-More than 500,000RMB")
        print(i, title)
        ax.set_title(title, fontdict=font2)  # 设置标题
        # ax.set_title(title, fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 12, 'style': 'italic'})  # 设置标题
        # ax.legend(legends, aspects, loc="upper left")  # 设置图例

        # 画中位线
        min_x = min(current_data_purpose["Performance"].tolist()) - 0.02
        max_x = max(current_data_purpose["Performance"].tolist()) + 0.02
        median_x = median(current_data_purpose["Performance"].tolist())
        min_y = min(current_data_purpose["Importance"].tolist()) - 0.2
        max_y = max(current_data_purpose["Importance"].tolist()) + 0.2
        median_y = median(current_data_purpose["Importance"].tolist())
        ax.plot([median_x, median_x], [min_y, max_y], color="black", linewidth=1.0)
        ax.plot([min_x, max_x], [median_y, median_y], color="black", linewidth=1.0)

        # 设置坐标轴范围
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # 去掉边缘空白
        # plt.margins(0, 0)
        # plt.xlim(xmin=min_x, xmax=max_x)
        # plt.ylim(ymin=min_y, ymax=max_y)
        i += 1
    leg = plt.legend(aspects, loc="lower center", prop=font3, bbox_to_anchor=(1.21, 1), borderaxespad=0., markerscale=5)  # 设置图例
    leg.get_frame().set_linewidth(0.0)  # 去掉图例边框
    plt.tight_layout()  # 调整子图之间距离
    # plt.legend(["o", "*", "□"], ["1", "2", "3"], prop=font3, bbox_to_anchor=(1.24, 0.55), borderaxespad=0.)
    plt.show()


import mpl_toolkits.axisartist as axisartist
# 2D 折线图，表现
def drawing_2D_perfomance(path, name, current_type):
    font = {'family': 'Times New Roman', 'size': 12}
    font2 = {'family': 'Times New Roman', 'size': 18}
    font3 = font_manager.FontProperties(family='Times New Roman', style='normal', size=15)
    colors = {"Appearance": "b", "Comfort": "g", "CP": "r", "EC": "c", "Handling": "m", "Interiors": "y", "Power": "k", "Space": "yellowgreen"}
    data = pd.read_excel(path, sheet_name=name, engine="openpyxl")
    print("data:", data)
    data = data.loc[data["Type"] == current_type]
    aspects = colors.keys()
    values = list(set(data["Value"].tolist()))
    print("values = ", values)
    figure = plt.figure(1)
    i = 1
    x = [2017, 2018, 2019, 2020]
    # values = ["Purpose"]
    for value in values:
        print("value = ", value)
        current_data_value = data.loc[data["Value"] == value]
        print("current_data_value:", current_data_value)
        quadrant = "13" + str(i)
        # ax = figure.add_subplot(quadrant)
        ax = axisartist.Subplot(figure, quadrant)
        figure.add_axes(ax)
        for aspect in aspects:
            current_data_aspect = current_data_value.loc[current_data_value["Aspect"] == aspect]
            print("current_data_aspect:", current_data_aspect)
            y = [current_data_aspect[2017].tolist()[0], current_data_aspect[2018].tolist()[0], current_data_aspect[2019].tolist()[0], current_data_aspect[2020].tolist()[0]]
            ax.plot(x, y, color=colors.get(aspect), linewidth=1.0)
        # ax.set_xlabel("Year", fontdict=font2)
        # ax.set_ylabel("Performance", fontdict=font2)
        if current_type == "Purpose":
            title = "(a)-Work needs" if value == "工作" else ("(b)-Living needs" if value == "生活" else "(c)-General needs")
        else:
            title = "(a)-Less than 200,000RMB" if value == 0 else ("(b)-200,000~500,000RMB" if value == 1 else "(c)-More than 500,000RMB")
        print(i, title)
        i += 1
        ax.set_title(title, fontdict=font2)  # 设置标题

        ax.set_xticklabels(["Before 2018", "2018", "2019", "After 2019"], fontdict=font)  # 设置坐标轴标签
        ax.set_xticks([2017, 2018, 2019, 2020])

        # 去掉边框
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # 带上箭头
        # ax.axis[:].set_visible(False)
        ax.axis["bottom"].set_axisline_style("->", size=1.0)
        ax.axis["left"].set_axisline_style("->", size=1.0)
        ax.axis["top"].set_visible(False)
        ax.axis["right"].set_visible(False)

        ax.axis["bottom"].label.set_text("Year")
        # ax.axis["bottom"].label.set_color("red")
        ax.axis["bottom"].label.set_size(18)
        ax.axis["bottom"].label.set_family("Times New Roman")
        ax.axis["left"].label.set_text("Performance")
        ax.axis["left"].label.set_size(18)
        ax.axis["left"].label.set_family("Times New Roman")

        # 设置坐标轴范围

    leg = plt.legend(aspects, loc="lower center", prop=font3, bbox_to_anchor=(1.21, 0.3), borderaxespad=0., markerscale=5)  # 设置图例
    leg.get_frame().set_linewidth(0.0)  # 去掉图例边框
    # plt.tight_layout()  # 调整子图之间距离
    plt.show()


# 2D 折线图+散点图，重要度
def drawing_2D_importance(path, name, current_type):
    font = {'family': 'Times New Roman', 'size': 12}
    font2 = {'family': 'Times New Roman', 'size': 18}
    font3 = font_manager.FontProperties(family='Times New Roman', style='normal', size=15)
    colors = {"Appearance": "b", "Comfort": "g", "CP": "r", "EC": "c", "Handling": "m", "Interiors": "y", "Power": "k", "Space": "yellowgreen"}
    data = pd.read_excel(path, sheet_name=name)
    data = data.loc[data["Type"] == current_type]
    aspects = colors.keys()
    values = list(set(data["Value"].tolist()))
    print("values = ", values)
    figure = plt.figure(1)
    i = 1
    x = [2017, 2018, 2019, 2020]
    for value in values:
        print("value = ", value)
        current_data_value = data.loc[data["Value"] == value]
        quadrant = "13" + str(i)
        # ax = figure.add_subplot(quadrant)
        ax = axisartist.Subplot(figure, quadrant)
        figure.add_axes(ax)
        for aspect in aspects:
            current_data_aspect = current_data_value.loc[current_data_value["Aspect"] == aspect]
            y = [current_data_aspect[2017].tolist()[0], current_data_aspect[2018].tolist()[0], current_data_aspect[2019].tolist()[0], current_data_aspect[2020].tolist()[0]]
            # 折线图
            ax.plot(x, y, color=colors.get(aspect), linewidth=1.0)
            # 散点图
            length = len(y)
            for j in range(length):
                current_x = x[j]
                current_y = y[j]
                # ax.plot(current_x, current_y, linewidth=0.5, marker='o', markerfacecolor=colors.get(aspect), markersize=6, label=aspect)
                ax.scatter(current_x, current_y, color="w", edgecolors=colors.get(aspect), s=45, marker='o')
        if current_type == "Purpose":
            title = "(a)-Work needs" if value == "工作" else ("(b)-Living needs" if value == "生活" else "(c)-General needs")
        else:
            title = "(a)-Less than 200,000RMB" if value == 0 else ("(b)-200,000~500,000RMB" if value == 1 else "(c)-More than 500,000RMB")
        print(i, title)
        i += 1
        ax.set_title(title, fontdict=font2)  # 设置标题

        ax.set_xticklabels(["Before 2018", "2018", "2019", "After 2019"], fontdict=font)  # 设置坐标轴标签
        ax.set_xticks([2017, 2018, 2019, 2020])

        ax.set_yticklabels(["8", "7", "6", "5", "4", "3", "2", "1"], fontdict=font)  # 设置坐标轴标签
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])

        # 去掉边框
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # 带上箭头
        # ax.axis[:].set_visible(False)
        ax.axis["bottom"].set_axisline_style("->", size=1.0)
        ax.axis["left"].set_axisline_style("->", size=1.0)
        ax.axis["top"].set_visible(False)
        ax.axis["right"].set_visible(False)

        ax.axis["bottom"].label.set_text("Year")
        # ax.axis["bottom"].label.set_color("red")
        ax.axis["bottom"].label.set_size(18)
        ax.axis["bottom"].label.set_family("Times New Roman")
        ax.axis["left"].label.set_text("Importance ranking")
        ax.axis["left"].label.set_size(18)
        ax.axis["left"].label.set_family("Times New Roman")

        # 设置坐标轴范围

    leg = plt.legend(aspects, loc="lower center", prop=font3, bbox_to_anchor=(1.21, 0.3), borderaxespad=0., markerscale=5)  # 设置图例
    leg.get_frame().set_linewidth(0.0)  # 去掉图例边框
    # plt.tight_layout()  # 调整子图之间距离
    plt.show()


# 画某款车的IPA数据，秦新能源
def drawing_single_car(path, name):
    font = {'family': 'Times New Roman', 'size': 12}
    font2 = {'family': 'Times New Roman', 'size': 18}
    font3 = font_manager.FontProperties(family='Times New Roman', style='normal', size=15)
    colors = {"Appearance": "b", "Comfort": "g", "CP": "r", "EC": "c", "Handling": "m", "Interiors": "y", "Power": "k", "Space": "yellowgreen"}
    markers = {1: "o", 2: "*", 3: "s"}
    data = pd.read_excel(path, sheet_name=name)
    aspects = colors.keys()

    figure = plt.figure(1)

    ax = figure.add_subplot(111)
    for aspect in aspects:
        current_data_aspect = data.loc[data["Aspect"] == aspect]
        x = current_data_aspect["Performance"].tolist()
        y = current_data_aspect["Importance"].tolist()
        # 折线
        ax.plot(x, y, color=colors.get(aspect), linewidth=1.0)
        # 箭头
        length = len(x)
        for j in range(length):
            if j == 0:
                continue
            start_x = (x[j] + x[j - 1])/2.0
            start_y = (y[j] + y[j - 1])/2.0
            plt.annotate("", xy=(x[j - 1], y[j - 1]), xytext=(start_x, start_y), weight="extra bold", arrowprops=dict(arrowstyle="<-", color=colors.get(aspect), mutation_scale=20))
        # 散点
        shape = current_data_aspect["Shape"].tolist()
        length = len(x)
        for j in range(length):
            current_x = x[j]
            current_y = y[j]
            current_shape = markers.get(shape[j])
            ax.scatter(current_x, current_y, color=colors.get(aspect), s=45, marker=current_shape)
        ax.set_xlabel("Performance", fontdict=font2)
        ax.set_ylabel("Importance", fontdict=font2)

    # 画中位线
    min_x = min(data["Performance"].tolist()) - 0.02
    max_x = max(data["Performance"].tolist()) + 0.02
    median_x = median(data["Performance"].tolist())
    min_y = min(data["Importance"].tolist()) - 0.2
    max_y = max(data["Importance"].tolist()) + 0.2
    median_y = median(data["Importance"].tolist())
    ax.plot([median_x, median_x], [min_y, max_y], color="black", linewidth=1.0)
    ax.plot([min_x, max_x], [median_y, median_y], color="black", linewidth=1.0)

    # 设置坐标轴范围
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    leg = plt.legend(aspects, loc="lower center", prop=font3, bbox_to_anchor=(1.21, 0.35), borderaxespad=0., markerscale=5)  # 设置图例
    leg.get_frame().set_linewidth(0.0)  # 去掉图例边框
    plt.tight_layout()  # 调整子图之间距离
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    path = "data/attribute_performance-20211018（python数据）.xlsx"
    sheet_name = "数据"
    # drawing_v3(path)
    print("current sheet:", sheet_name)
    # drawing_2d(path, sheet_name)
    current_type = "Purpose"
    drawing_2D_perfomance(path, sheet_name, current_type)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


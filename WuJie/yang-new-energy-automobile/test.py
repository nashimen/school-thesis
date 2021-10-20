import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support as score, classification_report
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 显示所有列
pd.set_option('display.max_columns', None)


def evaluationTest():
    # y = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
    # y_pred = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    y = [1, 1, 1, 0]
    y_pred = [1, 0, 0, 1]
    report = classification_report(y, y_pred, digits=4, output_dict=True)
    print(report)
    accuracy = report.get("accuracy")
    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')
    weighted_avg = report.get("weighted avg")
    weighted_precision = weighted_avg.get("precision")
    weighted_recall = weighted_avg.get("recall")
    weighted_f1 = weighted_avg.get('f1-score')
    micro_avg = report.get("micro avg")
    print(micro_avg)
    data = [weighted_precision, weighted_recall, weighted_f1, macro_precision, macro_recall, macro_f1, accuracy]
    print(data)


def dictTest():
    aaa = {"a": 1, "b": 3, "aac": 1.5}
    print(aaa)
    b = aaa.pop("a")
    print(aaa)
    print(b)


def getCurrentTime():
    current_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
    print(current_time)


if __name__ == "__main__":
    print("start...")

    import matplotlib.pyplot as plt
    import mpl_toolkits.axisartist as axisartist

    #建立画布
    fig = plt.figure()
    #使用axisartist.Subplot方法建立一个绘图区对象ax
    # ax = fig.add_subplot(111)
    ax = axisartist.Subplot(fig, 111)
    #将绘图区对象添加到画布中
    fig.add_axes(ax)
    #经过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
    #"-|>"表明实心箭头："->"表明空心箭头
    ax.axis["bottom"].set_axisline_style("-|>", size = 1.5)
    ax.axis["left"].set_axisline_style("->", size = 1.5)
    #经过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    x=[1,4]
    y=[1,8]
    plt.plot(x,y)
    plt.show()

    print("end...")


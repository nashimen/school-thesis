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


if __name__ == "__main__":
    print("start...")

    dictTest()

    print("end...")


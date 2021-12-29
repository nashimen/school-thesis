import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub
import xiangshi as xs
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

debug = True


# 把结果保存到csv
# report是classification_report生成的字典结果
def save_result_to_csv_cv(rows):
    path = 'result/evaluation-debug.csv' if debug else 'result/evaluation.csv'
    with codecs.open(path, "a", encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        f.close()


if __name__ == "__main__":
    print("开始执行main函数咯。。。")

    # 读取数据:真实标签+预测标签
    path_true = "test/test.csv" if debug else "result/sentiment_analysis_training_set.csv"
    path_pred = "test/predicted-test.xlsx" if debug else "result/predicted.xlsx"
    labels_true = pd.read_csv(path_true, encoding="utf-8")
    labels_pred = pd.read_excel(path_pred, engine="openpyxl")

    evaluation_global = pd.DataFrame()
    columns = labels_pred.columns
    rows = []
    for col in columns:
        Y_validation = labels_true[col]
        y_val_pred = labels_pred[col]

        # 计算MSE MAE R2 report
        mse = mean_squared_error(Y_validation, y_val_pred)
        mae = mean_absolute_error(Y_validation, y_val_pred)
        r2 = r2_score(Y_validation, y_val_pred)

        report = classification_report(Y_validation, y_val_pred, digits=4, output_dict=True)
        accuracy = report.get("accuracy")
        weighted_avg = report.get("weighted avg")
        weighted_precision = weighted_avg.get("weighted precision")
        weighted_recall = weighted_avg.get("weighted recall")
        weighted_f1 = weighted_avg.get('weighted f1-score')
        row = [weighted_precision, weighted_recall, weighted_f1, accuracy, mse, mae, r2]
        rows.append(row)

    save_result_to_csv_cv(rows)

    print("main函数执行结束咯。。。")


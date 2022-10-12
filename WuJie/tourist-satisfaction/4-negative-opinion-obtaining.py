import time, codecs, csv, math, numpy as np, random, datetime, os, gc, jieba, re, sys, pandas as pd, math
import jieba.posseg as pseg

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 20

# 初始化几个list
Entity_list = {"Traffic_convenience": [], "Distance_from_business_district": [], "Easy_to_find": [], "Wait_time": [],
               "Waiters_attitude": [], "Parking_convenience": [], "Serving_speed": [],  "Price_level": [],
               "Cost_effective": [], "Discount": [], "Decoration": [], "Noise": [], "Repast_space": [],
               "Environment_cleanliness": [], "Dish_portion": [], "Dish_taste": [], "Dish_look": []}
Opinion_list = {"Traffic_convenience": [], "Distance_from_business_district": [], "Easy_to_find": [], "Wait_time": [],
                "Waiters_attitude": [], "Parking_convenience": [], "Serving_speed": [],  "Price_level": [],
                "Cost_effective": [], "Discount": [], "Decoration": [], "Noise": [], "Repast_space": [],
                "Environment_cleanliness": [], "Dish_portion": [], "Dish_taste": [], "Dish_look": []}


def extract_negative_opinion(current_attribte, sentiment_set, opinion_set):
    length = len(sentiment_set)
    for i in range(length):
        if sentiment_set[i] in [-1]:
            # 如果有的话，保存当前观点
            current_opinion_set = opinion_set[i]
            if isinstance(current_opinion_set, float):
                continue
            # print("current opinion:", current_opinion_set, sentiment_set[i])
            pairs = current_opinion_set.split(",")
            for pair in pairs:
                temp = pair.split(" ")
                if len(temp) < 2:
                    continue
                Entity_list[current_attribte].append(temp[0])
                Opinion_list[current_attribte].append(temp[1])


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 1. 读取属性-观点文件
    path = "test/Entity_opinion-test.xlsx" if debug else "result/Entity_opinion.xlsx"
    data_global = pd.read_excel(path, engine="openpyxl", nrows=debugLength if debug else None)
    # print(data_global.head())
    print("data_global's length = ", len(data_global))

    # 2. 读取属性级情感文件
    sentiment_path = "data/2 Attribute performance.xlsx"
    data_sentiment = pd.read_excel(sentiment_path, engine="openpyxl", nrows=debugLength if debug else None)
    # print(data_global.head())

    # 3. 依次遍历每个属性，判断当前属性下&评论的情感是否为负
    attributes = Entity_list.keys()
    for attribute in attributes:
        print("正在处理属性:", attribute)
        extract_negative_opinion(attribute, data_sentiment[attribute], data_global[attribute])

        df = pd.DataFrame()
        df[str(attribute + "_entity")] = pd.Series(Entity_list.get(attribute))
        df[str(attribute + "_opinion")] = pd.Series(Opinion_list.get(attribute))

        s_path = "test/Entity_opinion_negative/Entity_opinion_" + attribute + "-test.xlsx" if debug \
            else "result/Entity_opinion_only_negative/Entity_opinion_" + attribute + ".xlsx"
        df.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


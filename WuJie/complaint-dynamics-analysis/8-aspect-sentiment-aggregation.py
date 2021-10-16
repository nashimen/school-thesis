import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
import paddlehub as hub
from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


# 将日期格式修改为 2018S1
def change_time_format(day):
    year = day.strftime("%Y")
    month = day.strftime("%m")

    if month in ["01", "02", "03"]:
        season = "S1"
    elif month in ["04", "05", "06"]:
        season = "S2"
    elif month in ["07", "08", "09"]:
        season = "S3"
    else:
        season = "S4"

    return year + season


# 返回指定元素下标
def remove_targe(mylist, target):
    index = []
    for i in range(len(mylist)):
        if mylist[i] == target:
            index.append(i)
    return index


# 计算当前酒店&当前季度的方面平均分
def calculate_average_score(hotel_data):
    aspects = {"LOCATION_label": [], "SERVICE_label": [], "ROOM_label": [], "SLEEP_QUALITY_label": [], "VALUE_label": []}
    for index, row in hotel_data.iterrows():
        for aspect in aspects.keys():
            aspects[aspect].append(row[aspect])
    # 移除-1
    for aspect, values in aspects.items():
        # 计算平均分
        temp = []
        for value in values:
            if value != -1:
                temp.append(value)
        if len(temp) == 0:  # 说明当前属性下没有分数
            score = -1
        else:
            score = np.mean(temp)
        aspects[aspect] = score

    return aspects["LOCATION_label"], aspects["SERVICE_label"], aspects["ROOM_label"], aspects["SLEEP_QUALITY_label"], aspects["VALUE_label"]


# 按照季度计算平均分
def calculate_score_season(data):
    # 修复时间问题（之前的方面级情感分类结果没有季度）
    data["季度"] = data.apply(lambda row: change_time_format(row["入住日期"]), axis=1)

    result = pd.DataFrame(columns=["city", "star", "hotel", "season", "location", "service", "room", "sleep_quality", "value"])
    # 按照酒店名称×季度进行汇总→保存至文件：地区，星级酒店名称，季度，aspect1，aspect2。。。
    hotels = list(set(data["名称"]))
    for hotel in hotels:
        current_data = data.loc[data["名称"] == hotel]
        city = current_data["地区"].tolist()[0]
        star = current_data["星级"].tolist()[0]
        seasons = list(set(current_data["季度"]))
        print(hotel, seasons)
        for season in seasons:
            location, service, room, sleep_quality, value = calculate_average_score(current_data.loc[current_data["季度"] == season])
            row = [city, star, hotel, season, location, service, room, sleep_quality, value]
            result = result.append([row])

    # 保存
    s_path = "data/test/aspect_sentiment_aggregation-test.xlsx" if debug else "result/aspect_sentiment_aggregation_v2.xlsx"
    result.to_excel(s_path, header=None, index=False)


# 按照月份计算平均分
def calculate_score_month(data):
    data["month"] = data.apply(lambda row: change_time_format_month(row["入住日期"]), axis=1)
    result = pd.DataFrame(columns=["city", "star", "hotel", "month", "location", "service", "room", "sleep_quality", "value"])
    # 按照酒店名称×month进行汇总→保存至文件：地区，星级酒店名称，month，aspect1，aspect2。。。
    hotels = list(set(data["名称"]))
    for hotel in hotels:
        current_data = data.loc[data["名称"] == hotel]
        city = current_data["地区"].tolist()[0]
        star = current_data["星级"].tolist()[0]
        months = list(set(current_data["month"]))
        print(hotel, months)
        for month in months:
            location, service, room, sleep_quality, value = calculate_average_score(current_data.loc[current_data["month"] == month])
            row = [city, star, hotel, month, location, service, room, sleep_quality, value]
            result = result.append([row])

    # 保存
    s_path = "data/test/aspect_sentiment_aggregationByMonth-test.xlsx" if debug else "result/aspect_sentiment_aggregationByMonth.xlsx"
    result.to_excel(s_path, header=None, index=False)


def change_time_format_month(day):
    return day.strftime("%Y%m")


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    path = "data/test/aspect_sentiment_result-test.xlsx" if debug else "result/aspect_sentiment_result.xlsx"
    data = pd.read_excel(path)

    calculate_score_month(data)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


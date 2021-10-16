import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 30


# 合并文件
def merge_files():
    # 获取路径下所有文件名
    pre_path = "data/pre_merge/"
    for root, dirs, files in os.walk(pre_path):
        print(files)
        print(type(files))

    # 依次读取每个文件内容
    print("正在读取文件内容。。。")
    length = len(files)
    for i in range(length):
        file_path = pre_path + files[i]
        if i == 0:
            df = pd.read_excel(file_path, nrows=debugLength if debug else None)
            print(df.head())
        else:
            current_df = pd.read_excel(file_path, nrows=debugLength if debug else None)
            df = df.append(current_df, ignore_index=True)
    print(df[:15])
    # 按照编号排序
    df = df.sort_values(by="编号", ascending=True)

    # 存入文件
    s_path = "test/merged.xlsx" if debug else "data/merged.xlsx"
    df.to_excel(s_path, index=False)


# 计算不同属性的感知质量分数
def calculate_aspect_score(read_path):
    data = pd.read_excel(read_path, engine="openpyxl", usecols=[9, 14, 16, 18, 20, 22, 24, 26, 28, 36])
    # 将 购买年份 列2017年之前数据换为2017
    data["购买年份_fix"] = data.apply(lambda r: time_fix(r["购买年份"]), axis=1)

    all_columns = data.columns
    print("all_columns:", all_columns)

    # 计算不同类型的分数
    types = set(data["类型"].tolist())


    for current_type in types:
        print("正在处理:", current_type)
        current_data = data.loc[data["类型"] == current_type]
        for aspect in all_columns:
            if aspect not in ['空间评分', '动力评分', '操控评分', '能耗评分', '舒适性评分', '外观评分', '内饰评分', '性价比评分']:
                continue
            years = list(set(current_data["购买年份_fix"]))
            for year in years:
                current_data_aspect = current_data.loc[current_data["购买年份_fix"] == year][aspect]
                score = np.mean(current_data_aspect)
                row = [current_type, aspect, year, score]
                result = result.append([row])
    return result


# 根据word的tfidf值汇总重要度
# 类型×年份×属性
def sum_tfidf_1(input_path):
    df = pd.DataFrame(columns=["type", "year", "aspect", "importance"])
    data = pd.read_csv(input_path)
    types = list(set(data["type"]))
    for current_type in types:
        current_data = data.loc[data["type"] == current_type]
        years = list(set(current_data["year"]))
        for year in years:
            current_data_year = current_data.loc[current_data["year"] == year]
            aspects = list(set(current_data_year["aspect"]))
            for aspect in aspects:
                importance = np.sum(current_data_year.loc[current_data_year["aspect"] == aspect]["tfidf"])
                df = df.append([{"type": current_type, "year": year, "aspect": aspect, "importance": importance}])

    return df


# 数据表中的数字后有空格，删掉
def delete_space(path, s_path):
    data = pd.read_excel(path, sheet_name="原始数据")
    data["价格-处理后"] = data.apply(lambda row: str(row["价格"]).strip(), axis=1)
    data.to_excel(s_path, index=False, header=None)


def year_fix(day):
    # day_time = datetime.datetime.strptime(day, "%Y/%m/%d")
    year = day.strftime("%Y")

    return year


# 数据处理，将时间改为年份
def change_time_format(path, s_path):
    data = pd.read_excel(path)
    print("数据读取完毕。。。")

    data["购买年份"] = data.apply(lambda row: year_fix(row["购买时间"]), axis=1)
    print("时间格式处理结束。。。")

    data.to_excel(s_path, index=False)


def time_fix(year):
    if int(year) < 2017:
        return 2017
    else:
        return year


# 购买年份处理
def buy_time_fix(path, s_path):
    data = pd.read_excel(path)
    data["购买年份"] = data.apply(lambda row: year_fix(row["购买时间"]), axis=1)
    # 将 购买年份 列2017年之前数据换为2017
    data["购买年份-fix"] = data.apply(lambda r: time_fix(r["购买年份"]), axis=1)
    data.to_excel(s_path, index=False)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    '''
    path = "test/新能源感知(2 types)-test_v2.xlsx" if debug else "data/新能源感知(2 types)_v2_temp.xlsx"
    s_path = "test/属性级感知质量分数-test.xlsx" if debug else "result/属性级感知质量分数.xlsx"
    sentiment_score = calculate_aspect_score(path)
    sentiment_score.to_excel(s_path, index=False, header=None)

    path = "test/hot_words-v2-test.csv" if debug else "result/hot_words-v2.csv"
    s_path = "test/hot_words-v2-sum-test.xlsx" if debug else "result/hot_words-v2-sum.xlsx"
    result = sum_tfidf_1(path)
    result.to_excel(s_path, index=False, header=None)

    path = "data/描述性统计.xlsx"
    s_path = "result/描述性统计-处理后.xlsx"
    delete_space(path, s_path)
    '''

    path = "test/新能源感知.xlsx" if debug else "data/新能源感知-20211006.xlsx"
    s_path = "test/新能源感知-20211006.xlsx" if debug else "data/新能源感知-20211006_v2.xlsx"
    buy_time_fix(path, s_path)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")



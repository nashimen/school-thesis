import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub
import Levenshtein as lev

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


# 修正日期
def time_fix(day):
    month = day.strftime("%Y%m")
    return month


# 评论预处理：去标点符号→去数字→分词→去停用词→合并
stoplist = pd.read_csv('../stopwords.txt').values
def data_processing(content):
    content = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？、~@#￥%……&*（）．；：】【|]+", "", str(content))
    content = list(jieba.cut(content))

    line = "".join(word.strip() for word in content if word not in stoplist and not word.isdigit())

    return line


# 计算传入评论的相似度
def calculate_similarity(texts):
    # print("texts:", texts)
    similarities = []
    length = len(texts)
    for i in range(length):
        for j in range(i, length):
            if i == j:
                continue
            dist = lev.distance(texts[i], texts[j])
            similarity = 1 - (dist / max(len(texts[i]), len(texts[j])))
            similarities.append(similarity)
            # print(similarity, texts[i], texts[j])

    return np.mean(similarities)


# 处理当前酒店的数据
def process_current_hotel(hotel_name, data, s_path, data_done):

    star = data["星级"].tolist()[0]
    current_result = pd.DataFrame()
    months = list(set(data["日期-fix"]))
    for month in months:
        if hotel_name in set(data_done["hotel"]):
            if month in set(data_done.loc[data_done["hotel"] == hotel_name]["month"]):
                print("该酒店在当前月份已完成:", hotel_name, month)
                continue
        data_month = data.loc[data["日期-fix"] == month]
        # print("2", data_month["酒店回复-fix"].tolist())
        if len(data_month["酒店回复-fix"].tolist()) <= 1:
            print("该酒店当前月份回复数量小于2，无法计算:", hotel_name, month)
            continue
        similarity = calculate_similarity(data_month["酒店回复-fix"].tolist())
        row = [star, hotel_name, month, similarity]
        # row = [{"star": star, "hotel": hotel_name, "month": month, "similarity": similarity}]
        current_result = current_result.append([row], ignore_index=True)

    # 保存当前酒店的数据
    current_result.to_csv(s_path, mode="a", encoding='utf_8_sig', index=False, header=None)

    return current_result


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    path_global = "test/北京2星-test.xlsx" if debug else "data/merged_file_month.xlsx"
    # 读取数据
    data_global = pd.read_excel(path_global)

    # 增加日期，例如 202101、202010
    data_global["日期-fix"] = data_global.apply(lambda row: time_fix(row["入住日期"]), axis=1)
    # print(data_global["日期-fix"])

    # 数据预处理：去标点符号→去数字→分词→去停用词→合并
    data_global["酒店回复-fix"] = data_global.apply(lambda row: data_processing(row["酒店回复"]), axis=1)

    # 读取已完成的酒店&月份
    data_done_global = pd.read_csv("result/similarity_calculating.csv")
    data_done_global.columns = ["star", "hotel", "month", "similarity"]
    # print(data_done_global)

    # 分别处理每家酒店的数据并保存
    s_path_global = "test/similarity_calculating-test.csv" if debug else "result/similarity_calculating.csv"
    hotels = list(set(data_global["名称"]))
    for hotel in hotels:
        current_data_hotel = data_global.loc[data_global["名称"] == hotel]
        # print("1", current_data_hotel["酒店回复"].tolist())
        if len(current_data_hotel["酒店回复"].tolist()) == 0:
            print("当前酒店没有回复，无法计算：", hotel)
            continue
        # 处理当前酒店
        process_current_hotel(hotel, current_data_hotel, s_path_global, data_done_global)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


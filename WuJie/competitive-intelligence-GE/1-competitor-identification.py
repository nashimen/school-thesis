import pandas as pd, numpy as np, time, re, os, jieba, json, requests, traceback, ast, paddlehub as hub, sys, math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 显示所有列
pd.set_option('display.max_columns', None)


debug = True
debugLength = 110


# 处理得到每家酒店的所有简称
def get_brief_name(chain_hotel_path):
    result = {}
    chain_hotel_data = pd.read_excel(chain_hotel_path)
    # chain_hotel_data = pd.read_excel(chain_hotel_path, nrows=debugLength if debug else None)
    # print(chain_hotel_data.head())
    # 酒店名称 col col1 col2 col3 col4 col5 col6 col7
    for indexs in chain_hotel_data.index:
        row = chain_hotel_data.loc[indexs].values[0: -1]
        for i in range(len(row)):
            if row[i] != "-1" and row[i] != -1 and row[i] not in result.keys():
                    result[row[i]] = row[0]

    return result


# 遍历当前酒店的评论，检索是否出现了连锁酒店名称，打印出来
def search_chain_hotel_name_v2(current_data, chbn, name):
    record = {}
    counter = 0
    for line in current_data["评论文本"].tolist():
        # 判断当前评论是否有连锁酒店名称
        for key in chbn.keys():
            # if counter % 100000 == 0:
            #     print("正在检索中，已检索", str(counter), "次")
            counter += 1
            if str(line).find(key) != -1:
                if str(name).find(key) != -1:
                    continue
                # print("counter = ", counter)
                temp = record.get(key, 0)
                record[key] = temp + 1

    # 把record转为pd格式
    final = pd.DataFrame(columns=("hotels", "counts"))
    for key, value in record.items():
        final = final.append([{"hotels": key, "counts": value}])
    # 索引重置
    final = final.reset_index(drop=True)
    # 排序
    final = final.sort_values(by="counts", ascending=False)
    # 转为str
    final["counts"] = final["counts"].apply(str)

    return ','.join(final["hotels"].tolist()), ','.join(final["counts"].tolist())


# 遍历当前酒店的评论，检索是否出现了连锁酒店名称，打印出来
def search_chain_hotel_name(current_data, chbn, name):
    record = {}
    counter = 0
    for line in current_data["评论文本"].tolist():
        # 判断当前评论是否有连锁酒店名称
        for key in chbn.keys():
            # if counter % 100000 == 0:
            #     print("正在检索中，已检索", str(counter), "次")
            counter += 1
            if str(line).find(key) != -1:
                if str(name).find(key) != -1:
                    continue
                # print("counter = ", counter)
                chain_hotel_name = chbn.get(key)
                print(name, chain_hotel_name, key)
                temp = record.get(chain_hotel_name, 0)
                record[chain_hotel_name] = temp + 1

    # 把record转为pd格式
    final = pd.DataFrame(columns=("hotels", "counts"))
    for key, value in record.items():
        final = final.append([{"hotels": key, "counts": value}])
    # 索引重置
    final = final.reset_index(drop=True)
    # 排序
    final = final.sort_values(by="counts", ascending=False)
    # 转为str
    final["counts"] = final["counts"].apply(str)

    return ','.join(final["hotels"].tolist()), ','.join(final["counts"].tolist())


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in 1-competitor-identification.py ...")

    # 读取所有连锁酒店名称及其简称
    path = "data/酒店信息-processing.xlsx"
    # 处理得到每家连锁酒店的所有简称
    chain_hotel_brief_names = get_brief_name(path)
    print("chain_hotel_brief_name's length = ", len(chain_hotel_brief_names))
    # print(chain_hotel_brief_names)

    # 结果形式 row:评论酒店，名称酒店，提及频率（降序）
    # {评论酒店: {名称酒店1：次数，名称酒店2：次数}，评论酒店2：{名称酒店1：次数，名称酒店2：次数}}

    # 读取所有评论酒店的评论数据
    path2 = "data/merged_file+补充数据.xlsx"
    print("正在读取评论酒店数据", "*" * 50)
    hotel_data = pd.read_excel(path2, nrows=debugLength if debug else None)
    # 只要北京数据
    hotel_data = hotel_data.loc[hotel_data["地区"] == "北京"]
    # 依次遍历每家酒店的评论，判断是否出现了连锁酒店名称
    hotel_names = list(set(hotel_data["名称"]))
    result = pd.DataFrame(columns=["location", "star", "name", "competitors", "counts"])
    for name in hotel_names:
        print("current hotel is", name)
        competitors, counts = search_chain_hotel_name(hotel_data.loc[hotel_data["名称"] == name], chain_hotel_brief_names, name)
        location = hotel_data.loc[hotel_data["名称"] == name]["地区"].tolist()[0]
        star = hotel_data.loc[hotel_data["名称"] == name]["星级"].tolist()[0]
        row = [location, star, name, competitors, counts]
        print("row:", row)
        result = result.append([{"location": location, "star": star, "name": name, "competitors": competitors, "counts": counts}])
    s_path = "result/competitors_v3.csv"
    # print(result.head())
    result.to_csv(s_path, index=False, encoding='utf_8_sig', mode="w")

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of 1-competitor-identification.py...")


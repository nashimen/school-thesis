import time, codecs, csv, math, numpy as np, random, datetime, os, gc, jieba, re, sys, pandas as pd
import jieba.posseg as pseg

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True
debugLength = 50

origin_dictionary = {
    "Traffic_convenience": "交通便捷，交通便利，交通，方便，周边，开车",
    "Distance_from_business_district": "商圈，商业区，中心，中心地带，距离",
    "Easy_to_find": "找不到，容易找到，位置，位置好找，好找，定位，地点，地址，导航，招牌，临街",
    "Wait_time": "排队，等待，等候，叫号，等位，人气，生意，流量，时间，客流量",
    "Waiters_attitude": "服务员，服务生，态度，热情，冷漠，老板，服务大姐，阿姨，大妈，店员，服务，服务态度，大姐，小姐姐，管理，发票，老板娘，小哥哥，人，营业时间，员工，姐姐，前台，店家，商家，店里",
    "Parking_convenience": "停车，停车场，停车费，停车位，车位",
    "Serving_speed": "速度，上菜速度，服务速度，上菜，点菜",
    "Price_level": "价格，贵，便宜，价位，人均，菜价，价钱，定价",
    "Cost_effective": "性价比，划算，不划算，超值，不值，实惠，经济，经济实惠，优惠，原材料，套餐，用料",
    "Discount": "打折，折扣，活动，免费，团购",
    "Decoration": "装饰，装修，装潢，店面，环境，店铺，馆子，门脸，装修风格，店里，空调，空气流通，排风，气氛，灯光，格局，过道，设施，坏境，区域，冷气，档次，大厅，桌位，私密性，布局，布置，室内，细节，视野，氛围，包间，桌子，店门，设计，门面",
    "Noise": "噪音，吵闹，嘈杂，闹腾，闹哄哄，包厢，隔音，声音，音乐，嗓音",
    "Repast_space": "空间，就餐空间，拥挤，面积，小店，店面，座位，座，地方",
    "Environment_cleanliness": "整洁，干净，脏，卫生，卫生间",
    "Dish_portion": "分量，菜品分量，量大，量很足，品种，面料，量，个头，配菜，份量，料，饭量，块头，食材量，重量，质量，食量，小料量，菜量，餐量，体量，个儿，种类，肉量，数量，菜量",
    "Dish_taste": "味道，好吃，难吃，口味，咸淡，咸，重口味，辣，口感，脆，不脆，新鲜，嫩，牛蛙，饮料，果汁，酸菜鱼，大麦茶，汤，糖醋里脊，米饭，疙瘩汤，美味，正宗，爱吃，川菜，油腻，地道，过瘾，食材，新鲜，配料，搭配，肉质，食材，牛蛙，茄子，肥肠，莴笋，花生，鸡块，辣度，馋嘴蛙，辣味，肉肉，烧饼，鸡丁，毛血旺，牛肉饼，麻辣，芹菜，黄瓜，小龙虾，川菜，米饭，菜品，辣子鸡，鱼刺，烧饼，材料，烤鸭，酸奶，青菜，炸酱面，香味，蔬菜，肉質，调味，手艺，品质，内容，手工制作，鸡翅，菜，锅底，菜味，花生米，鱼头，手撕包菜，藕片，包菜，青笋，菠菜，耗儿鱼，油，丝瓜，鸡肉，辣椒",
    "Dish_look": "菜品外观，颜色，装盘，摆盘，餐具，汤色，色泽，外形，颜值，色香味，品相，卖相"
}
# 初始化几个list
Entity_list = {"Traffic_convenience": [], "Distance_from_business_district": [], "Easy_to_find": [], "Wait_time": [],
               "Waiters_attitude": [], "Parking_convenience": [], "Serving_speed": [],  "Price_level": [],
               "Cost_effective": [], "Discount": [], "Decoration": [], "Noise": [], "Repast_space": [],
               "Environment_cleanliness": [], "Dish_portion": [], "Dish_taste": [], "Dish_look": []}
Opinion_list = {"Traffic_convenience": [], "Distance_from_business_district": [], "Easy_to_find": [], "Wait_time": [],
                "Waiters_attitude": [], "Parking_convenience": [], "Serving_speed": [],  "Price_level": [],
                "Cost_effective": [], "Discount": [], "Decoration": [], "Noise": [], "Repast_space": [],
                "Environment_cleanliness": [], "Dish_portion": [], "Dish_taste": [], "Dish_look": []}

def initDictionary():
    dictionary = {}
    # 去重+统计个数
    count = 0
    for attribute, words in origin_dictionary.items():
        words = words.split("，")
        # print(attribute, "原始长度为", len(words))
        words = list(set(words))
        print(attribute, "长度为:", len(words))
        count += len(words)
        dictionary[attribute] = words
    # 统计个数
    print("词典总长度为", count)
    return dictionary

dictionary = initDictionary()

# 匹配原则：
# 1.只保留同时包括评价对象和观点的数据


import xiangshi as xs
# 根据相似度匹配属性
def judgeTopicBySimilarity(entity):
    # 计算entity与字典中所有key的相似度，取最大值为最终结果
    max_similarity = 0
    topic = "EMPTY"
    for key, value in dictionary.items():
        sim = 0
        for v in value:
            # print("text:", text, ", v:", v)
            sim = max(sim, xs.cossim([entity, v]))  # 找到当前key下最大的相似度
        if max_similarity < sim:
            max_similarity = sim
            topic = key

    return topic


def matching_line(entity, opinion):
    # print(entity, ":", opinion)
    # 开始判断属性
    attribute_result = "Traffic_convenience" if entity in dictionary.get("Traffic_convenience") \
        else "Distance_from_business_district" if entity in dictionary.get("Distance_from_business_district") \
        else "Easy_to_find" if entity in dictionary.get("Easy_to_find") else "Wait_time" if entity in dictionary.get("Wait_time") \
        else "Waiters_attitude" if entity in dictionary.get("Waiters_attitude") \
        else "Parking_convenience" if entity in dictionary.get("Parking_convenience") \
        else "Serving_speed" if entity in dictionary.get("Serving_speed") else "Price_level" if entity in dictionary.get("Price_level") \
        else "Cost_effective" if entity in dictionary.get("Cost_effective") else "Discount" if entity in dictionary.get("Discount") \
        else "Decoration" if entity in dictionary.get("Decoration") else "Noise" if entity in dictionary.get("Noise") \
        else "Repast_space" if entity in dictionary.get("Repast_space") \
        else "Environment_cleanliness" if entity in dictionary.get("Environment_cleanliness") \
        else "Dish_portion" if entity in dictionary.get("Dish_portion") else "Dish_taste" if entity in dictionary.get("Dish_taste") \
        else "Dish_taste" if entity in dictionary.get("Dish_taste") else "Dish_look" if entity in dictionary.get("Dish_look") else "EMPTY"

    if attribute_result == "EMPTY":
        attribute_result = judgeTopicBySimilarity(entity)

    if attribute_result == "EMPTY":
        print("最终仍未匹配到属性：", entity, opinion)
        return
    Entity_list[attribute_result].append(entity)
    Opinion_list[attribute_result].append(opinion)


def matching(data_list):
    # 依次处理每行
    for line in data_list:
        # print("正在匹配:", line)
        pairs = line.split(",")
        # print("pairs:", pairs)
        for current_pair in pairs:
            # print("正在查找属性:", current_pair)
            temp = current_pair.split(" ")
            if len(temp) < 2:
                # if len(temp) == 1:
                #     print("没有观点词：", temp)
                continue
            matching_line(temp[0], temp[1])

    print("》》》结果检验：")
    for key, value in Entity_list.items():
        length_entity = len(value)
        length_opinion = len(Opinion_list.get(key))
        if length_entity != length_opinion:
            print("检验不通过，实体列表和观点列表长度不一致！！！")


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 1. 读取观点文件
    path = "data/1 opinion extraction results.xlsx"
    data_global = pd.read_excel(path, engine="openpyxl", nrows=debugLength if debug else None)
    print(data_global.head())
    print("data_global's length = ", len(data_global))

    # 2. 依次处理每行，判断其中观点所属的属性→将每个属性下的观点对保存为列表→新建一个DF保存五列的结果
    matching(data_global["Opinion"])

    # print("Entity_list:", Entity_list)
    # print("Opinion_list:", Opinion_list)

    # 3. 保存到文件
    for k, v in Entity_list.items():
        df = pd.DataFrame()
        # print("k:", k)
        # print("v:", v)
        df[str(k + "_entity")] = pd.Series(v)
        df[str(k + "_opinion")] = pd.Series(Opinion_list.get(k))
        s_path = "test/17-Entity_opinion_" + k + "-test.xlsx" if debug else "result/17-Entity_opinion_" + k + ".xlsx"
        df.to_excel(s_path, index=False)

    # print(df.head())

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


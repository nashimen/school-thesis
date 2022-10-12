import time, codecs, csv, math, numpy as np, random, datetime, os, gc, jieba, re, sys, pandas as pd
import jieba.posseg as pseg

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True
debugLength = 20

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


def matching_line(current_attribute, entity):
    # print(entity, ":", opinion)
    if entity in dictionary.get(current_attribute):
        return True

    attribute_result = judgeTopicBySimilarity(entity)

    if attribute_result == "EMPTY":
        print("未匹配到属性：", entity)

    return attribute_result == current_attribute


# 查找属于当前属性的观点，返回字符串，例如“味道 不错, 味道 好吃”
def searching_opinions(current_attribute, line):
    result = []
    pairs = line.split(",")
    for current_pair in pairs:
        # print("正在查找属性:", current_pair)
        temp = current_pair.split(" ")
        if len(temp) < 2:
            continue
        if matching_line(current_attribute, temp[0]):
            result.append(current_pair)

    return ",".join(result)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 1. 读取观点文件
    path = "data/1 opinion extraction results.xlsx"
    data_global = pd.read_excel(path, engine="openpyxl", nrows=debugLength if debug else None)
    print(data_global.head())
    print("data_global's length = ", len(data_global))

    # 2. 遍历17个属性，依次处理每行，查找属于当前属性的观点
    # attributes = origin_dictionary.keys()
    # print("1:", attributes)
    attributes = [*origin_dictionary]
    attributes.reverse()
    print(attributes)
    for attribute in attributes:
        print("正在处理属性：", attribute)
        data_global[attribute] = data_global.apply(lambda row_global: searching_opinions(attribute, row_global["Opinion"]), axis=1)
        s_path = "test/Entity_opinion-reverse-test.xlsx" if debug else "result/Entity_opinion-reverse.xlsx"
        data_global.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


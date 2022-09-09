import time, codecs, csv, math, numpy as np, random, datetime, os, gc, jieba, re, sys, pandas as pd

origin_dictionary = {
    "Location": "交通便捷，交通便利，交通，方便，周边，商圈，商业区，中心，中心地带，找不到，容易找到，位置，位置好找，好找",
    "Service": "排队，等待，等候，叫号，等位，人气，生意，流量，服务员，服务生，态度，热情，冷漠，老板，服务大姐，阿姨，大妈，店员，服务，服务态度，大姐，小姐姐，管理，停车，停车场，停车费，停车位，速度，上菜速度，服务速度，上菜，点菜",
    "Price": "价格，贵，便宜，价位，性价比，划算，不划算，超值，不值，实惠，经济，经济实惠，优惠，打折，折扣，活动，免费，团购",
    "Environment": "卫生间，室内，嗓音，装饰，装修，装潢，店面，环境，店铺，馆子，门脸，装修风格，噪音，吵闹，嘈杂，闹腾，闹哄哄，空间，就餐空间，拥挤，面积，小店，店面，座位，座，整洁，干净，脏，卫生",
    "Dish": "用料，油，丝瓜，鸡肉，辣椒，，分量，菜品分量，量大，量很足，品种，面料，量，个头，配菜，份量，料，饭量，块头，味道，好吃，难吃，口味，咸淡，咸，重口味，辣，口感，脆，不脆，新鲜，嫩，牛蛙，饮料，果汁，酸菜鱼，大麦茶，汤，糖醋里脊，米饭，疙瘩汤，美味，正宗，爱吃，川菜，油腻，地道，过瘾，食材，新鲜，配料，搭配，肉质，食材，牛蛙，茄子，肥肠，莴笋，花生，鸡块，辣度，馋嘴蛙，辣味，肉肉，烧饼，鸡丁，毛血旺，牛肉饼，麻辣，芹菜，黄瓜，小龙虾，川菜，米饭，菜品，辣子鸡，鱼刺，烧饼，材料，汤色，烤鸭，酸奶，青菜，炸酱面，菜品外观，颜色，装盘，摆盘，餐具"
}


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
            print("sim = ", sim)
        if max_similarity < sim:
            max_similarity = sim
            topic = key

    return topic


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    term = "菜"
    print(judgeTopicBySimilarity(term))

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


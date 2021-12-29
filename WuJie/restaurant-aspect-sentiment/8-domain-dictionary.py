import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False

# 种子词
seed_dictionary = {
    "location_traffic_convenience": "交通，交通便利，公交车，公交站，地铁，地铁站，打车",
    "location_distance_from_business_district": "偏僻，市区，闹市，中心，城中心，闹市区，商圈",
    "location_easy_to_find": "不好找，位置，角落，定位，定位不准，好找，地标，卡角",
    "service_wait_time": "预定，排队，等位，提前，等候，排号，位子，人满，满座，生意火，提前，订餐，定位等候时间，排队时间，人气，人气很旺",
    "service_waiters_attitude": "服务员，服务，态度，管理，迎宾，前台，热情，小哥，小姐姐，阿姨，到位，忙，冷漠，态度差，态度好，热情似火，清洁人员，服务生，吧台，不热情，很热情，服务到位，服务态度差，大堂经理，店家，服务周到，周到，怠慢",
    "service_parking_convenience": "停车，车库，停车场，停车位，车位，车位费，免费停车，不好停车，开车，停车方便，空位置，地下停车场",
    "service_serving_speed": "上菜，上菜快，上菜慢，上菜速度，速度，做菜时间，上菜时间，磨叽",
    "price_level": "价格，价钱，贵，离谱，高，低，便宜，有点贵，有点便宜，不贵，不便宜，价格适中，价格适合，价格合理，不合理，小资，价位，价格贵，价位高，价位低，价格便宜",
    "price_cost_effective": "性价比，不值，值得，性价比低，性价比高，不划算，划算",
    "price_discount": "折扣，有折扣，无折扣，折扣力度，力度，优惠券，团购，代金券",
    "environment_decoration": "装修，精装，精装修，简装，装修一般，硬件，设施，装潢，装潢大气，装潢一般，装饰，布置，光线",
    "environment_noise": "很吵，安静，静，吵闹，大声，喧哗，大声喧哗，烦躁，人多，热闹，嘈杂，静谧",
    "environment_space": "空间大，空间，拥挤，就餐空间，挤，压迫感，空间小，店面",
    "environment_cleaness": "卫生，干净，不干净，抽烟，吸烟，烟味儿，烟味，脏，乱，不卫生，很脏，不太干净，很干净，非常干净",
    "dish_portion": "满满的，量，足，不够，少，大，足够，小，撑，剩下，多，丰富，种类齐全，菜品，菜量",
    "dish_taste": "口感，味道，好吃，美味，不好吃，难吃，太硬，硬，味道一般，味道差，口味，中规中矩，甜，酸，咸，辣，油腻，油，鲜美，传统，可口，闻，香，吃不惯，味儿，味，嫩，巨好吃，很好吃，很难吃，酸酸甜甜，酸甜，酸酸，甜甜，惊艳，味道普通，普通，太咸，外焦里嫩，肉质，浓郁，浓，脆脆的，脆，品质，清爽",
    "dish_look": "精致，好看，拍照，外观，照片"
}
dictionary = {}  # 最终的领域词典
all_words = set()  # 已存在的词汇
counter = 0
for key, value in seed_dictionary.items():
    value = value.split("，")
    all_words.update(set(value))
    # print("value:", value)
    dictionary[key] = set(value)
    counter += len(set(value))
    print(key, len(set(value)))
print("counter = ", counter)
print("dictionary:", dictionary)
print("*" * 100)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    version = time.strftime('%Y%m%d%H%M%S', time.localtime(start_time))

    # 保存dictionary
    s_path_global = "test/domain_dictionary_" + str(version) + "-test.npy" if debug else "result/domain_dictionary_" + str(version) + ".npy"
    np.save(s_path_global, dictionary)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


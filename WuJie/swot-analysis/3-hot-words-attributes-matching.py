import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
import paddlehub as hub
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 30

origin_dictionary = {
    "Location": "王府井，天安门广场，天安门，长安街，位置，地区，城市，距离，方位，路边，马路边，偏僻，偏，偏僻，僻静，方位，位置，方位，角落，地段，距离，东长安街，北门，宅急送，五环，北四环，北三环，大兴区，北二环，五环，中医院，肿瘤医院，科技馆，体检中心，工体，工人体育馆，人工湖，阜外医院，马桥，东单，西单，团结湖，潘家园，北京宾馆，毛主席纪念堂，平安里，后花园，奥林匹克公园，四环路，建国路，朝阳医院，国家电网，北京大学，清华大学，北大，清华，北京天文馆，使馆区，北大第一医院，东南西北，地坛公园，东安市场，南楼，北京动物园，西二环，首都医科大学，西三环，内环，北京医院，镇政府，北影，产业园，同仁医院，西路，昆仑饭店，西直门，国医学科学院肿瘤医院，故宫，虎坊桥，地图，地理位置，玉渊潭，卡地亚，三庆园，不好找，好找，难找，偏僻，偏，王府井，很近，王府井大街，建国门，不远，地理，位于，处于，就近，故宮，地点，二环，国贸，北京站，国家博物馆，故宫博物院，国博，人民大会堂，秀水街，东二环，黄金地段，博物馆，",
    "Value": "价格，价值，放假，贵，便宜，花费，费用，质量，性价比，星级，昂贵，金钱，钱，不值，乱收费，服务费，停车费，价位，不便宜，坑人，太坑，盈利，毛钱，一分货，坑，大坑，不值，性价比，乱收费，价格，贵，太值，服务费，价钱，费用，收费，停车费，价位，性价，分值，太坑，坑，大坑，坑人，不便宜，一分货，同价，价格优势，太贵，一分钱，价，一毛钱，不太值，很贵，快递费，昂贵，非常规，盈利，毛钱，性價，优惠，小贵，超值，",
    "Room": "房间，地板，大小，房型，空间，窗户，卧室，卫生间，洗漱台，洗漱间，厕所，洗手间，双人房，标准间，套房，精品房总统套房，家庭房，玻璃，太冷，房间，房型，房間，大小，标间，双人房，老房子，地方，双人间，房间内，标准间，套房，商务房，精品房，总统套房，四间房，旅社，旅舍，屋子，屋，标房，居室，民宅，豪庭，贵宾房，换衣间，窗房，装修，舒服，舒适，客房，地毯，阳台，太小，枕头，面积，挺大，宽敞，宽敞明亮，明亮，家具，通风，沙发，床垫，床单，陈设，偏小，床铺，床头柜，舒适度，简陋，",
    "Restaurant": "食物，食品，早餐，餐食，白酒，咖啡，饮料，面条，包子，油条，煎饼，食堂，堂食，炒饭，炒面，茶点，点心，午餐，晚餐，早饭，午饭，晚饭，面包，美食，饺子，饮品，矿泉水，微波炉，杂粮，菜单，水果，食材，茶叶，胃口，瓶装水，家常菜，啤酒，果汁，馄饨，炒菜，西餐厅，西餐，口味，品种，品类，早餐券，鸡蛋，零食，番茄酱，西瓜，果盘，苹果汁，咖啡杯，牛奶，中餐厅，葡萄，餐具，鸡蛋，早餐券，食物，品种，咖啡机，套餐，口味，农夫山泉，热饮，咖啡，西餐厅，绿茶，汽水，炒菜，肉包，茶包，混沌，馄饨，小吃，果汁，面条，家常菜，瓶装水，啤酒，微波炉，饮料机，休闲吧，烤面包机，面包机，盘子，冰箱，餐，菜量，胃口，茶叶，食材，猕猴桃，晚餐，菜品，餐饮，下午茶，蒸点，凯撒，面包干，芝麻，菜，吸油烟机，热菜，凉菜，早餐，海鲜，苹果，餐食，厨房，餐台，油条，欧式，面包，包子，电磁炉，美食，黄油，零食，午餐，饺子，饮品，番茄酱，菜单，杂粮，矿泉水，微波炉，水果，西瓜，果盘，用餐，苹果汁，咖啡杯，牛奶，中餐厅，餐厅，葡萄，煎饼果子，自助餐厅，餐具，品种，吃饭，自助餐，",
    "Transport": "交通，公交车，出租车，站点，车站，地铁，路线，马路，公路，机场，线路，公共交通，车流，走路，地鐵站，公交车，公交车站，车流，四惠东，线站，车辆，公共交通，换乘，北京地铁，线路，主路，二号线，地铁，机场，地铁站，高铁，公交，路线，火车站，火车，地铁口，车站，便利，分钟，时间，打车，叫车，出租，打出租，滴滴，不到，十分钟，便捷，号线，小时，几分钟，一号线，五分钟，公交站，打的，打滴，",
    "Cleanliness": "霉味，卫生状况，卫生，印渍，臭味，一股味，一股味儿，味儿，环境恶劣，潮味，甲醛，烟味，抽烟，卫生条件，空气流通，脏，爬虫，臭虫，积水，烟雾，排泄物，血渍，除味剂，霉味，卫生状况，烟味，印渍，灰尘，排泄物，烟雾，墙角，潮虫，蜈蚣，条腿，有脏，二手烟，酸味，香烟，臭味，除味剂，空气净化，鼻涕，臭气，股子，一股子，一股，味儿，环境恶劣，潮味，甲醛，不脏，爬虫，脏，湿气，空气流通，积水，吸烟区，油烟味，有烟味，环境，卫生条件，霉味，卫生状况，烟味，印渍，灰尘，排泄物，烟雾，墙角，潮虫，蜈蚣，条腿，有脏，二手烟，酸味，香烟，臭味，除味剂，湿疹，空气净化，臭气，股子，一股子，一股，味儿，潮味，甲醛，不脏，爬虫，脏，湿气，空气流通，积水，卫生条件，床太潮，湿疹，皮肤过敏，过敏，枕头，褥子，床太潮，皮肤过敏，鼻涕，环境恶劣，吸烟区，油烟味，有烟味，环境，干净，味道，打扫，气味，难闻，恶心，消毒，防疫，疫情，防控，疫情防控，酒精消毒，酒精消毒液，消毒液，洗手液，",
    "Parking": "停车场，停车，停车点，停车位，车位，停靠，车库，停车库，停车场，停车费，方便停车，泊车，代停车，代泊车，停车服务，",
    "SleepQuality": "透光度，隔音，噪音，异响，隔音性，耳塞，咳嗽声，咳嗽，嗓音，吵闹，临街，动静，噪声，杂音，车声，猫猫，猫，撸猫，声音，闹腾，隔间，空气，气氛，床，隔音，邻居，隔壁，晚上，灯光，噪音，施工，枕头，睡觉，睡眠，睡眠质量，周边，周围，被子，床单，床板，异响，吵闹，吵，动静，临街，声音，耳塞，咳嗽声，嗓音，闹腾，杂音，噪声，门铃，隔音性，窗帘，",
    "Facility": "体育馆，羽毛球馆，电梯，直梯，健身房，游泳池，音响，音箱，点石，TV，投影仪，高尔夫，空调，设施，电视，TV，频道，空调，镜子，下水道，电热器，马桶，花洒，卫浴，水龙头，淋浴房，下水道，景观灯，下水道，卫浴，花洒，汤池，卫生设施，拖鞋，衣柜，景观灯，便器，马桶，干衣机，排气扇，小便池，卫生洁具，化妆镜，电热器，置物架，洗手盆，洗脸池，泡池，洗手盆，装修，陈旧，老旧，酒店设施，设备，泳池，齐全，硬件，新装修，气派，設施，年代，老舊，太老，配套，太旧，豪华，老化，太老旧，有用，翻新，洗浴，破旧，插头，电脑，",
    "Internet": "Wi-Fi，WiFi，wifi，wi-fi，互联网，网络，无线网络，无线网，宽带，网速，断网，路由器，信号，无线，路由，无线路由，有线网，有线，有线宽带，无线宽带，网络服务，拨号连接，Wifi",
    "Service": "服务，微笑，前台，接待，人员，态度，冷漠，不耐烦，会员，经理，管理，热情，入住，退房，阿姨，清洁工，管家，行政，保安，清洁员，房嫂，吧台，客户，投诉，服务设施，阿姨，退房，管理，服務員，管家，行政，保安，服务，经理，换房，前台，态度，态度恶劣，服务员，人员，服务态度，解决问题，极差，大堂，經理，经理，清洁员，中层干部，干部，服务区，房嫂，男士，吧台，老客户，办理，员工，客户，水平，投诉，清洁工，投诉无门，业务素质，小姐，服务体系，工作人员，小姐姐，小姑娘，女服务员，男服务员，素质，电话，专业，极差，客服，服务水平，联系，业务，办事，迎宾，小伙子，女士，美女，总台，职业，客房部，办手续，人手，热情，帅哥，押金，师傅，素养，客户经理，服务中心，服务行业，管理人员，贴心，热情周到，礼貌，热心，预定，服务质量，服务生，笑脸相迎，热情服务，友好，亲和力，友善，优质服务，开发票，发票，结账，",
    "Bathroom": "毛巾，温度，太冷，浴袍，牙膏，牙刷，洗面奶，沐浴露，浴缸，冷热水，热水，浴帽，马桶刷，溢水，洗澡水，梳子，镜子，洗漱间，水渍，浴袍，牙刷毛，牙刷，牙膏，洗面奶，洗发液，沐浴露，便池，浴缸，蓬头，排水口，洗发液，冷热水，浴液，洗手台，欧舒丹，浴帽，浴房，洗澡时，盆浴，浴盆，淋浴房，隔柜，排气，肥皂，洗发水，牙具，洗发水，淋浴，浴帘，帘子，帘，沐浴露，热水，厕所，热水澡，卫生间，洗手间，水盆，水龙头，热水，水温，吹风机，浴巾，浴袍，淋浴间，",
    "Surrounding": "街道，街上，公园，商铺，周围，商业街，周边，周围，基站，庆丰，周边，周边环境，工地，路边，马路边，对面，商铺，公园，雍和宫，维景，維景，商场，妇产医院天安门城楼，亮马桥，外景，度假村，中餐馆，滑雪场，大型超市，娱乐区，百货店，环内，小吊梨，小吊梨汤，美食城，夜景，医院，周围，附近，快餐店，麦当劳，肯德基，餐饮店，水果店，商业街，经典，茶馆，风景，風景，旁边，安静，步行街，安全，边上，环境优美，超市，便利店，周围环境，毗邻，"
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


# 加载停用词
def getStopList():
    stoplist = pd.read_csv(filepath_or_buffer='../stopwords.txt').values
    return stoplist


# 判断是否包含中文字符
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


# 保留中文字符，删除非中文字符
def find_Chinese(doc):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    Chinese = re.sub(pattern, "", doc)
    # print(Chinese)
    return Chinese


dictionary = initDictionary()
# 关键词-属性匹配
def word_attribute_match(wordlist):
    length = len(wordlist)
    # 存在每条评论的短文本（属性匹配后），所有value的长度都一样。如果同一属性下有多条，则合并
    texts_dictionary = {"Location": [], "Value": [], "Room": [], "Restaurant": [], "Transport": [], "Cleanliness": [], "Parking": [], "SleepQuality": [], "Surrounding": [], "Internet": [], "Service": [], "Bathroom": [], "Facility": []}
    # 依次遍历每条评论&及其短文本
    for i in range(length):
        current_word = wordlist[i]

        topic = judgeTopic(current_word)
        # 如果根据先验词典未匹配到属性，则计算相似度，选择相似度最大的属性作为当前属性
        if topic == "EMPTY":
            topic = judgeTopicBySimilarity(current_word)
        if topic == "EMPTY":
            print("最终仍未匹配到属性：", current_word)
            continue

        texts_dictionary[topic].append(current_word)

    return texts_dictionary


# 判断短文本属于哪个主题，根据dictionary
def judgeTopic(text):
    words = text.split(' ')
    topic_result = ''
    for word in words:
        # 遍历先验词库
        topic_result = "Location" if word in dictionary.get("Location") else "Value" if word in dictionary.get("Value") \
            else "Room" if word in dictionary.get("Room") else "Internet" if word in dictionary.get("Internet") \
            else "Restaurant" if word in dictionary.get("Restaurant") else "Transport" if word in dictionary.get("Transport") \
            else "Cleanliness" if word in dictionary.get("Cleanliness") else "Parking" if word in dictionary.get("Parking") \
            else "SleepQuality" if word in dictionary.get("SleepQuality") else "Surrounding" if word in dictionary.get("Surrounding") \
            else "Service" if word in dictionary.get("Service") else "Bathroom" if word in dictionary.get("Bathroom") \
            else "Facility" if word in dictionary.get("Facility") else "EMPTY"

    return topic_result


import xiangshi as xs
# 根据相似度匹配属性
def judgeTopicBySimilarity(word):
    # 计算text与字典中所有key的相似度，取最大值为最终结果
    max_similarity = 0
    topic = "EMPTY"
    for key, value in dictionary.items():
        sim = 0
        for v in value:
            # print("text:", text, ", v:", v)
            sim = max(sim, xs.cossim([word], [v]))  # 找到当前key下最大的相似度
        if max_similarity < sim:
            max_similarity = sim
            topic = key

    return topic


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    print(">>>正在读取数据。。。")
    path_global = "data/test/hot-words-test.xlsx" if debug else "result/hot-words.xlsx"
    data_global = pd.read_excel(path_global, engine="openpyxl")

    s_path_global = "data/test/hot-words-matching-test.xlsx" if debug else "result/hot-words-matching.xlsx"

    # 将热词与属性进行匹配
    result = word_attribute_match(data_global["word"].tolist())

    attributes = ["Location", "Value", "Room", "Restaurant", "Bathroom", "Cleanliness", "Parking", "SleepQuality", "Surrounding", "Internet", "Service", "Transport", "Facility"]
    # 依次处理每个属性
    for attribute in attributes:
        df = pd.concat([pd.DataFrame({attribute: result[attribute]})], axis=1)
    df.to_excel(s_path_global, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


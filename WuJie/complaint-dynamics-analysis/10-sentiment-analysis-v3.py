import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
import paddlehub as hub
from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 30


origin_dictionary = {
    # 包括交通Transportation
    "Location": "位置，地区，城市，距离，方位，路边，马路边，偏僻，偏，偏僻，僻静，方位，位置，方位，角落，地段，距离，东长安街，北门，宅急送，五环，北四环，北三环，大兴区，北二环，五环，中医院，肿瘤医院，科技馆，体检中心，工体，工人体育馆，人工湖，阜外医院，马桥，东单，西单，团结湖，潘家园，北京宾馆，毛主席纪念堂，平安里，后花园，奥林匹克公园，四环路，建国路，朝阳医院，国家电网，北京大学，清华大学，北大，清华，北京天文馆，使馆区，北大第一医院，东南西北，地坛公园，东安市场，南楼，北京动物园，西二环，首都医科大学，西三环，内环，北京医院，镇政府，北影，产业园，同仁医院，西路，昆仑饭店，西直门，国医学科学院肿瘤医院，故宫，虎坊桥，地图，地理位置，玉渊潭，卡地亚，三庆园，不好找，好找，难找，偏僻，偏，",
    "Transport": "交通，公交车，出租车，站点，车站，地铁，路线，马路，公路，机场，线路，公共交通，车流，走路，地鐵站，公交车，公交车站，车流，四惠东，线站，车辆，公共交通，换乘，北京地铁，线路，主路，二号线，地铁，机场，地铁站，高铁，公交，路线，火车站，火车，地铁口，车站",
    "Surroundings": "街道，街上，公园，商铺，周围，商业街，周边，周围，基站，庆丰，周边，周边环境，工地，路边，马路边，对面，商铺，公园，雍和宫，维景，維景，商场，妇产医院天安门城楼，亮马桥，外景，度假村，中餐馆，滑雪场，大型超市，娱乐区，百货店，环内，小吊梨，小吊梨汤，美食城，夜景，医院，周围，附近，快餐店，麦当劳，肯德基，餐饮店，水果店，商业街，经典，茶馆，风景，風景，",
    "Service": "服务，微笑，前台，接待，人员，态度，冷漠，不耐烦，会员，经理，管理，热情，入住，退房，阿姨，清洁工，管家，行政，保安，清洁员，房嫂，吧台，客户，投诉，服务设施，阿姨，退房，管理，服務員，管家，行政，保安，服务，经理，换房，前台，态度，态度恶劣，服务员，人员，服务态度，解决问题，极差，大堂，經理，经理，清洁员，中层干部，干部，服务区，房嫂，男士，吧台，老客户，办理，员工，客户，水平，投诉，清洁工，投诉无门，业务素质，小姐，服务体系，工作人员，小姐姐，小姑娘，女服务员，男服务员，素质，电话，专业，极差，客服，服务水平，联系，业务，办事，迎宾，小伙子，女士，美女，总台，职业，客房部，办手续，人手，热情，帅哥，押金，师傅，素养，客户经理，服务中心，服务行业，管理人员，",
    "Wifi": "Wi-Fi，WiFi，wifi，wi-fi，互联网，网络，无线网络，无线网，宽带，网速，断网，路由器，信号，",
    "Restaurant": "食物，食品，早餐，餐食，白酒，咖啡，饮料，面条，包子，油条，煎饼，食堂，堂食，炒饭，炒面，茶点，点心，午餐，晚餐，早饭，午饭，晚饭，面包，美食，饺子，饮品，矿泉水，微波炉，杂粮，菜单，水果，食材，茶叶，胃口，瓶装水，家常菜，啤酒，果汁，馄饨，炒菜，西餐厅，西餐，口味，品种，品类，早餐券，鸡蛋，零食，番茄酱，西瓜，果盘，苹果汁，咖啡杯，牛奶，中餐厅，葡萄，餐具，鸡蛋，早餐券，食物，品种，咖啡机，套餐，口味，农夫山泉，热饮，咖啡，西餐厅，绿茶，汽水，炒菜，肉包，茶包，混沌，馄饨，小吃，果汁，面条，家常菜，瓶装水，啤酒，微波炉，饮料机，休闲吧，烤面包机，面包机，盘子，冰箱，餐，菜量，胃口，茶叶，食材，猕猴桃，晚餐，菜品，餐饮，下午茶，蒸点，凯撒，面包干，芝麻，菜，吸油烟机，热菜，凉菜，早餐，海鲜，苹果，餐食，厨房，餐台，油条，欧式，面包，包子，电磁炉，美食，黄油，零食，午餐，饺子，饮品，番茄酱，菜单，杂粮，矿泉水，微波炉，水果，西瓜，果盘，用餐，苹果汁，咖啡杯，牛奶，中餐厅，餐厅，葡萄，煎饼果子，自助餐厅，餐具，品种",
    "Parking": "停车场，停车，停车点，停车位，车位，停靠，车库，停车库，停车场，",
    "Facility": "体育馆，羽毛球馆，电梯，直梯，健身房，游泳池，音响，音箱，点石，TV，投影仪，高尔夫，空调，设施，电视，TV，频道，空调，镜子，下水道，电热器，马桶，花洒，卫浴，水龙头，淋浴房，下水道，景观灯，下水道，卫浴，花洒，汤池，卫生设施，拖鞋，衣柜，景观灯，便器，马桶，干衣机，排气扇，小便池，卫生洁具，化妆镜，电热器，置物架，洗手盆，洗脸池，泡池，洗手盆，",
    "Room": "房间，地板，大小，房型，空间，窗户，卧室，卫生间，洗漱台，洗漱间，厕所，洗手间，双人房，标准间，套房，精品房总统套房，家庭房，玻璃，太冷，房间，房型，房間，大小，标间，双人房，老房子，地方，双人间，房间内，标准间，套房，商务房，精品房，总统套房，四间房，旅社，旅舍，屋子，屋，标房，居室，民宅，豪庭，贵宾房，换衣间，窗房，装修",
    "Bathroom": "毛巾，温度，太冷，浴袍，牙膏，牙刷，洗面奶，沐浴露，浴缸，冷热水，热水，浴帽，马桶刷，溢水，洗澡水，梳子，镜子，洗漱间，水渍，浴袍，牙刷毛，牙刷，牙膏，洗面奶，洗发液，沐浴露，便池，浴缸，蓬头，排水口，洗发液，冷热水，浴液，洗手台，欧舒丹，浴帽，浴房，洗澡时，盆浴，浴盆，淋浴房，隔柜，排气，肥皂，洗发水，牙具，洗发水，淋浴，浴帘，帘子，帘，沐浴露，热水，厕所，热水澡，卫生间，洗手间，水盆，水龙头，热水，",
    "Sleep": "透光度，隔音，噪音，异响，隔音性，耳塞，咳嗽声，咳嗽，嗓音，吵闹，临街，动静，噪声，杂音，车声，猫猫，猫，撸猫，声音，闹腾，隔间，空气，气氛，床，隔音，邻居，隔壁，晚上，灯光，噪音，施工，枕头，睡觉，睡眠，睡眠质量，周边，周围，被子，床单，床板，异响，吵闹，吵，动静，临街，声音，耳塞，咳嗽声，嗓音，闹腾，杂音，噪声，门铃，隔音性，",
    "Value": "价格，价值，放假，贵，便宜，花费，费用，质量，性价比，星级，昂贵，金钱，钱，不值，乱收费，服务费，停车费，价位，不便宜，坑人，太坑，盈利，毛钱，一分货，坑，大坑，不值，性价比，乱收费，价格，贵，太值，服务费，价钱，费用，收费，停车费，价位，性价，分值，太坑，坑，大坑，坑人，不便宜，一分货，同价，价格优势，太贵，一分钱，价，一毛钱，不太值，很贵，快递费，昂贵，非常规，盈利，毛钱",
    "Cleanliness": "霉味，卫生状况，卫生，印渍，臭味，一股味，一股味儿，味儿，环境恶劣，潮味，甲醛，烟味，抽烟，卫生条件，空气流通，脏，爬虫，臭虫，积水，烟雾，排泄物，血渍，除味剂，霉味，卫生状况，烟味，印渍，灰尘，排泄物，烟雾，墙角，潮虫，蜈蚣，条腿，有脏，二手烟，酸味，香烟，臭味，除味剂，空气净化，鼻涕，臭气，股子，一股子，一股，味儿，环境恶劣，潮味，甲醛，不脏，爬虫，脏，湿气，空气流通，积水，吸烟区，油烟味，有烟味，环境，卫生条件，霉味，卫生状况，烟味，印渍，灰尘，排泄物，烟雾，墙角，潮虫，蜈蚣，条腿，有脏，二手烟，酸味，香烟，臭味，除味剂，湿疹，空气净化，臭气，股子，一股子，一股，味儿，潮味，甲醛，不脏，爬虫，脏，湿气，空气流通，积水，卫生条件，床太潮，湿疹，皮肤过敏，过敏，枕头，褥子，床太潮，皮肤过敏，鼻涕，环境恶劣，吸烟区，油烟味，有烟味，环境，",
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


# 保留中文字符，删除非中文字符
def find_Chinese(doc):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    Chinese = re.sub(pattern, "", doc)
    # print(Chinese)
    return Chinese


# 加载停用词
def getStopList():
    stoplist = pd.read_csv(filepath_or_buffer='../stopwords.txt').values
    return stoplist


digits = '0123456789'
# 文本预处理,shortTexts-list type
def text_processing(shortTexts, scores):
    result = []
    result_scores = []
    result_origin = []
    for current_scores in scores:
        current_scores = current_scores.strip('[')
        current_scores = current_scores.strip(']')
        current_scores = current_scores.replace(" ", "")
        current_scores = current_scores.split(',')
        # print("current_scores:", current_scores)
        # print("current_scores' type:", type(current_scores))
        result_scores.append(list(map(float, current_scores)))

    for texts in shortTexts:
        texts = texts.strip('[')
        texts = texts.strip(']')
        texts = texts.replace("'", "")
        texts = texts.replace(" ", "")
        texts = texts.split(',')
        # print("texts:", texts)
        result_origin.append(texts)

        current_result = []
        for line in texts:
            origin_line = line
            # 去标点符号
            line = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(line))
            # 去掉数字
            line = line.translate(str.maketrans('', '', digits))
            # 去掉非中文字符
            line = find_Chinese(line)
            # 分词
            line = jieba.cut(line)
            # 去停用词
            stoplist = getStopList()
            line = [word.strip() for word in line if word not in stoplist]
            line = ' '.join(line)
            if len(line.strip()) == 0:
                print("当前短文本预处理后为空：", origin_line)
            current_result.append(line)
        result.append(current_result)
    # print("result_origin:", result_origin)
    return result, result_scores, result_origin


import xiangshi as xs
# 根据相似度匹配属性
def judgeTopicBySimilarity(text):
    # 计算text与字典中所有key的相似度，取最大值为最终结果
    max_similarity = 0
    topic = "EMPTY"
    for key, value in dictionary.items():
        sim = 0
        for v in value:
            sim = max(sim, xs.cossim([text], [v]))  # 找到当前key下最大的相似度
        if max_similarity < sim:
            max_similarity = sim
            topic = key

    return topic


# 判断短文本属于哪个主题，根据dictionary
def judgeTopic(text):
    words = text.split(' ')
    topic_result = ''
    for word in words:
        # 遍历先验词库
        topic_result = "Location" if word in dictionary.get("Location") else "Transport" if word in dictionary.get("Transport") \
            else "Surroundings" if word in dictionary.get("Surroundings") else "Service" if word in dictionary.get("Service") \
            else "Facility" if word in dictionary.get("Facility") else "Room" if word in dictionary.get("Room") \
            else "Bathroom" if word in dictionary.get("Bathroom") else "Sleep" if word in dictionary.get("Sleep") \
            else "Value" if word in dictionary.get("Value") else "Cleanliness" if word in dictionary.get("Cleanliness") \
            else "Wifi" if word in dictionary.get("Wifi") else "Parking" if word in dictionary.get("Parking") \
            else "Restaurant" if word in dictionary.get("Restaurant") else "EMPTY"

    return topic_result


dictionary = initDictionary()
# 短文本-属性匹配
def shortText_attribute_match(shortTexts, scores):
    # 对shortTexts进行预处理：去标点符号→分词→去停用词
    shortTexts, scores, origin_shortTexts = text_processing(shortTexts, scores)
    if len(shortTexts) != len(scores):
        print("shortTexts和scores长度不一致。。。")
        sys.exit(-2)
    length = len(shortTexts)
    # 存在每条评论的短文本（属性匹配后），所有value的长度都一样。如果同一属性下有多条，则合并
    texts_dictionary = {"Location": [], "Transport": [], "Surroundings": [], "Service": [], "Wifi": [], "Restaurant": [], "Parking": [], "Facility": [], "Room": [], "Bathroom": [], "Sleep": [], "Value": [], "Cleanliness": []}
    # 存在每条评论短文本（属性匹配后）对应的正向情感分，所有value的长度都一样。如果同一属性下有多条，则取平均值
    scores_dictionary = {"Location": [], "Transport": [], "Surroundings": [], "Service": [], "Wifi": [], "Restaurant": [], "Parking": [], "Facility": [], "Room": [], "Bathroom": [], "Sleep": [], "Value": [], "Cleanliness": []}  # 未提及-1
    # 依次遍历每条评论&及其短文本
    for i in range(length):
        texts_dictionary_temp = {"Location": [], "Transport": [], "Surroundings": [], "Service": [], "Wifi": [], "Restaurant": [], "Parking": [], "Facility": [], "Room": [], "Bathroom": [], "Sleep": [], "Value": [], "Cleanliness": []}  # 当前评论的变量
        scores_dictionary_temp = {"Location": [], "Transport": [], "Surroundings": [], "Service": [], "Wifi": [], "Restaurant": [], "Parking": [], "Facility": [], "Room": [], "Bathroom": [], "Sleep": [], "Value": [], "Cleanliness": []}  # 当前评论的
        current_texts_origin = origin_shortTexts[i]  # 原始的评论文本
        current_texts = shortTexts[i]
        current_scores = scores[i]
        current_length = len(current_texts)
        if current_length != len(current_scores):
            print("current_texts:", current_texts)
            print("current_scores:", current_scores)

            print("短文本条数和scores个数不一致。。。")
            sys.exit(-3)
        for j in range(current_length):
            text_origin = current_texts_origin[j]  # 原始的短文本
            text = current_texts[j]
            # 如果短文本预处理完之后为空，则continue
            if len(text.strip()) == 0:
                continue
            score = current_scores[j]
            topic = judgeTopic(text)

            # 如果根据先验词典未匹配到属性，则计算相似度，选择相似度最大的属性作为当前属性
            if topic == "EMPTY":
                topic = judgeTopicBySimilarity(text_origin)
            if topic == "EMPTY":
                print("最终仍未匹配到属性：", text_origin)
                continue

            texts_dictionary_temp[topic].append(text_origin)
            scores_dictionary_temp[topic].append(score)

        # 每处理完一条评论（可能包含多个短文本），进行以此合并和平均汇总，存放的应该是原始的评论文本（短文本）
        for key, value in scores_dictionary_temp.items():
            value_texts = ",".join(texts_dictionary_temp.get(key))
            # 如果value有多个，则求平均;同时合并短文本
            if len(value) > 0:
                mean = np.mean(value)
            else:
                mean = -1  # 如果不存在当前属性，则标记为-1
            texts_dictionary[key].append(value_texts)
            scores_dictionary[key].append(mean)

    # 判断长度是否一致
    lengths = []
    for key, value in texts_dictionary.items():
        # print(key, "'s length = ", len(value))
        lengths.append(len(value))
    for key, value in scores_dictionary.items():
        # print(key, "'s length = ", len(value))
        lengths.append(len(value))
    if len(set(lengths)) > 1:
        print("匹配结果有问题，长度不一致")
        sys.exit(-4)
    else:
        print("匹配结束。。。")
    return texts_dictionary, scores_dictionary


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件+短文本提取
    print(">>>正在读取数据。。。")
    path = "data/test/merged_file+score.xlsx" if debug else "data/original/merged_file+score.xlsx"
    current_data = pd.read_excel(path, nrows=debugLength if debug else None)

    # 短文本-属性匹配
    texts_result, scores_result = shortText_attribute_match(current_data["shortTexts"].tolist(), current_data["score"].tolist())
    # 将以上结果进行保存
    for key, value in texts_result.items():
        current_data[key] = pd.Series(value)
        current_data[(key + "_label")] = pd.Series(scores_result.get(key))

    # print(current_data.head())
    s_path = "data/test/aspect_sentiment_result_13_aspects.xlsx" if debug else "result/aspect_sentiment_result_13_aspects.xlsx"
    current_data.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")
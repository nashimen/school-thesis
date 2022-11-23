import time, codecs, csv, math, numpy as np, random, datetime, os, pandas as pd, jieba, re, sys, string, lexical_diversity
# import paddlehub as hub
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 150

origin_dictionary = {
    "Facility": "区域，健身房，恒温，水温，温水，池子，休息区，游泳池，泳池，自带，露天，篮球场，运动，打篮球，更衣室，水质，游泳馆，温泉池，温泉，泡温泉，泡池，公共，spa，私汤，汤池，半山，設施，窗帘，电视，灯光，开关，网速，电影，机器人，投屏，系统，智能，网络，wifi，信号，科技感，电视机，马桶盖，高科技，小机器人，空气净化器，新风，模式，音乐，机器，频道，家居，无线，蓝牙，音箱，音响，小度，语音，插卡，加湿器，新风机，智能化，感应，有意思，摆设",
    "Room": "走廊，房子，房门，房间，窗户，过道，客房，楼道，房間，一人，双人，一个人，大床，公寓，窗房，内窗，房型，楼层，两人，两个人，大床房，早餐券，标间，标准间，双人床，朝向，榻榻米，特价，行政，双床，家庭房，酒廊，单人，商务房，三个人，套房，行政房，门票，平米，套间，高楼层，客厅，主楼，一厅，贵宾，loft，床品，浴巾，毛巾，床垫，枕头，被子，床铺，床单，被套，被褥，软硬，餐具，耳塞，衣服，电源，插座，手机，遥控器，床头，水壶，吹风机，充电器，电脑，床头柜，插头，usb，床边，台灯，穿着，一边，声音，私密性，睡眠，安全感，大声，蚊子，屋子，门锁，睡觉",
    "Cleanliness": "异味，烟味，季节，味儿，空调，冷风，隔壁，中央空调，卫生，臭味，空气，暖气，冷气，天气，温度，光线，霉味，气味，感冒，透气，通风，头发，垃圾，状况，污渍，不到位，下水道，地面，水池，蟑螂，垃圾桶，镜子，洗手台，地板，灰尘，厨房，地漏，地毯，墙壁，衣柜，桌子，沙发，角落，椅子，颜色，柜子，黑色，洗手，两边"
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


# 判断词语出现频率
def count_frequency(sentence_set, word_set):
    local_words = []
    local_counters = []
    for current_word in word_set:
        counter = 0
        for current_sentence in sentence_set:
            if current_word in str(current_sentence):
                counter += 1
        if counter > 0:
            local_words.append(current_word)
            local_counters.append(counter)
            if counter > 10:
                print(current_word, counter)

    return local_words, local_counters


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    print(">>>正在读取数据。。。")

    attributes = dictionary.keys()
    # 读取sentence文件
    sentences_path = "data/6 aspect_sentences.xlsx"
    sentences = pd.read_excel(sentences_path, engine="openpyxl", nrows=debugLength if debug else None)

    for attribute in attributes:
        # 判断当前方面下各属性词出现的频率
        current_attribute_sentences = sentences[attribute]
        words, counters = count_frequency(current_attribute_sentences, dictionary.get(attribute))
        if debug:
            print(words)
            print(counters)

        # 保存words和counters
        result = {"Words": words, "Counters": counters}
        result = pd.DataFrame(result)
        save_path = "test/6 word counter-" + attribute + ".xlsx" if debug else "result/6 word counter-" + attribute + ".xlsx"
        result.to_excel(save_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    sys.exit(10086)


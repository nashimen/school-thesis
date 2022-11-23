import time, codecs, csv, math, numpy as np, random, datetime, os, pandas as pd, jieba, re, sys, string, lexical_diversity
# import paddlehub as hub
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 150


# 判断词语出现频率
def count_frequency(current_word, sentence_set):
    counter = 0
    for current_sentence in sentence_set:
        if current_word in current_sentence:
            counter += 1
    if counter > 10:
        print(current_word, counter)
    return counter


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    print(">>>正在读取数据。。。")

    attributes_dict = {"Facility": 1, "Parking": 2, "Service": 3, "Child": 4, "Room": 5, "Value": 6, "Booking": 7,
                       "Cleanliness": 8, "Bathroom": 9, "Location": 10, "Surrounding": 11, "Decoration": 12, "Food": 13}

    current_attribute = "Facility"
    print("Current attribute is ", current_attribute)
    words_path = "data/3 Attribute words.xlsx"
    words = pd.read_excel(words_path, nrows=attributes_dict.get(current_attribute), usecols=[1], engine="openpyxl")["Phrases"].values[0]
    words = str(words).split("，")
    print(words)

    # 读取sentence文件
    sentences_path = "data/4 Kansei sentiments.xlsx"
    sentences = pd.read_excel(sentences_path, engine="openpyxl", nrows=debugLength if debug else None)
    current_attribute_sentences = sentences.loc[sentences["Attribute"] == current_attribute]["sentence"]
    # print(current_attribute_sentences)
    # for sentence in current_attribute_sentences:
    #     print("sentence:", sentence)

    # 依次判断每个词汇出现频率
    counters = []  # 保存所有词汇的出现次数
    for word in words:
        counters.append(count_frequency(word, current_attribute_sentences))
        if debug:
            print("word:", word)

    # 保存words和counters
    result = {"Words": words, "Counters": counters}
    result = pd.DataFrame(result)
    save_path = "test/5 word counter-" + current_attribute + ".xlsx" if debug else "result/5 word counter-" + current_attribute + ".xlsx"
    result.to_excel(save_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


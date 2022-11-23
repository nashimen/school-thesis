import time, codecs, csv, math, numpy as np, random, datetime, os, pandas as pd, jieba, re, sys, string, lexical_diversity
# import paddlehub as hub
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 150

origin_dictionary = {
    "Facility": "泳池，机器，电视，健身房，智能",
    "Room": "房间，大床，枕头，楼层，声音",
    "Cleanliness": "卫生，地毯，空气，温度，天气"
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

# 同义词处理：
synonyms = {"房间": ["屋子", "房型", "房子", "房間", "客房", "套房"],
            "大床": ["床品", "床垫", "双床"],
            "机器": ["机器人", "小机器人"]
            }


# 判断词语出现频率
def count_frequency(current_attribute, sentence_set, word_set, kansei_word_set):
    print("正在处理：", word_set)
    # print("正在处理属性：", current_attribute)
    # print("1 sentences' length:", len(sentence_set[current_attribute]))
    local_synonyms = [word_set]
    if word_set in synonyms.keys():
        local_synonyms.extend(synonyms.get(word_set))
    local_words = []  # 存放Kansei words
    local_counters = []  # 存放Kansei word与属性词的共现频率

    # print(df[df['name'].str.contains('li')])
    # 获取包含当前kansei_word的句子
    # print("before sentence_set:", sentence_set)

    for kansei_word in kansei_word_set:
        # print("current kansei word:", kansei_word)
        counter = 0
        # 获取包含当前kansei word的所有句子
        sentence_set_new = sentence_set[sentence_set[current_attribute].str.contains(str(kansei_word), na=False)]
        # print("sentence_set:", sentence_set_new.head())
        if sentence_set_new.empty or len(sentence_set_new[current_attribute]) < 1:
            continue
        if debug:
            print("after sentence_set:", sentence_set_new)
        # 遍历属性词，计算出现次数
        for sentence in sentence_set_new[current_attribute]:
            for local_synonyms_word in local_synonyms:
                if local_synonyms_word in sentence:
                    # print("出现共现：", local_synonyms_word, sentence)
                    # 如果存在，则计数，同时break（避免重复统计）
                    counter += 1
        if counter > 0:
            local_words.append(kansei_word)
            local_counters.append(counter)
        if counter > 5:
            print(word_set, kansei_word, counter)

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

    # 读取Kansei words
    kansei_words_path = "data/3 Kansei words.xlsx"
    kansei_words = pd.read_excel(kansei_words_path, engine="openpyxl")["Kansei words"].tolist()
    print(kansei_words)

    for attribute in attributes:
        # 判断当前方面下各属性词与Kansei words的共现频率
        attribute_words = dictionary.get(attribute)
        # current_attribute_sentences = sentences[attribute]
        for attribute_word in attribute_words:
            words, counters = count_frequency(attribute, sentences, attribute_word, kansei_words)
            if debug:
                print(words)
                print(counters)

            # 保存words和counters
            result = {"Words": words, "Counters": counters}
            result = pd.DataFrame(result)
            save_path = "test/synthesis/7 word counter-" + attribute + "_" + attribute_word + ".xlsx" if debug \
                else "result/synthesis/7 word counter-" + attribute + "_" + attribute_word + ".xlsx"
            result.to_excel(save_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    sys.exit(10086)


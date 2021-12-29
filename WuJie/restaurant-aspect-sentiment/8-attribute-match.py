import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub
import xiangshi as xs

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False

dictionary_path = "result/domain_dictionary_20211229125355.npy"
origin_dictionary = np.load(dictionary_path, allow_pickle=True).item()


def initDictionary():
    dictionary = {}
    # 去重+统计个数
    count = 0
    for attribute, words in origin_dictionary.items():
        # words = words.split("，")
        # print(attribute, "原始长度为", len(words))
        words = list(set(words))
        # print(attribute, "长度为:", len(words))
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


digits = '0123456789'
#文本分句
def cut_sentence(text):
    pattern = r',|\.|;|\?|:|!|\(|\)|，|。|、|；|·|！| |…|（|）|~|～|：|；|“|”|"'
    sentences = list(re.split(pattern, text))
    # print("sentences = ", sentences)
    sentence_list = []
    for w in sentences:
        # 去标点符号
        w = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(w))
        # 去掉数字
        w = w.translate(str.maketrans('', '', digits))
        # 去掉非中文字符
        w = find_Chinese(w)
        if len(w) > 1:
            sentence_list.append(w)
    # print("sentence_list:", sentence_list)
    return sentence_list


dictionary = initDictionary()
# 短文本-属性匹配
def shortText_attribute_match(text):
    # 以字典形式保存
    result = {
        "location_traffic_convenience": "",
        "location_distance_from_business_district": "",
        "location_easy_to_find": "",
        "service_wait_time": "",
        "service_waiters_attitude": "",
        "service_parking_convenience": "",
        "service_serving_speed": "",
        "price_level": "",
        "price_cost_effective": "",
        "price_discount": "",
        "environment_decoration": "",
        "environment_noise": "",
        "environment_space": "",
        "environment_cleaness": "",
        "dish_portion": "",
        "dish_taste": "",
        "dish_look": "",
        "Empty": ""
    }
    # 长评论分为短句
    sentences = cut_sentence(text)

    # 依次判断短句属于哪个方面（遍历领域词汇）
    for sentence in sentences:
        # print("sentence:", sentence)
        flag = False  # 标记是否找到当前sentence所属方面
        # ①遍历领域词汇
        for attribute, words in dictionary.items():
            for word in words:
                if sentence.find(word) >= 0:
                    flag = True
                    result[attribute] += " " + sentence
                    # result[attribute].append(sentence)
                if flag:
                    break
            if flag:
                break
        # ②根据相似度判断
        if not flag:
            max_similarity = 0
            topic = "Empty"
            for attribute, words in dictionary.items():
                similarity = 0
                for word in words:
                    similarity = max(similarity, xs.cossim([sentence, word]))
                if max_similarity < similarity:
                    max_similarity = similarity
                    topic = attribute
            # result[topic].append(sentence)
            result[topic] += " " + sentence
            if topic == "Empty":
                print("未匹配到方面属性：", sentence)

    return result["location_traffic_convenience"], result["location_distance_from_business_district"], result["location_easy_to_find"],\
           result["service_wait_time"], result["service_waiters_attitude"], result["service_parking_convenience"], \
           result["service_serving_speed"], result["price_level"], result["price_cost_effective"], result["price_discount"], \
           result["environment_decoration"], result["environment_noise"], result["environment_space"], result["environment_cleaness"], \
           result["dish_portion"], result["dish_taste"], result["dish_look"]


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    version = time.strftime('%Y%m%d%H%M%S', time.localtime(start_time))

    path = "test/test.csv" if debug else "data/sentiment_analysis_training_set.csv"
    data_global = pd.read_csv(path, encoding="utf-8")
    columns = data_global.columns

    # print(data_global.columns)

    # 短文本-属性匹配,结果以DF方式保存未提及的值为-2
    data_result = pd.DataFrame()
    processed = data_global.apply(lambda row_global: shortText_attribute_match(str(row_global["content"])), axis=1)
    data_result["location_traffic_convenience"], data_result["location_distance_from_business_district"], \
    data_result["location_easy_to_find"], data_result["service_wait_time"], data_result["service_waiters_attitude"], \
    data_result["service_parking_convenience"], data_result["service_serving_speed"], data_result["price_level"], \
    data_result["price_cost_effective"], data_result["price_discount"], data_result["environment_decoration"], \
    data_result["environment_noise"], data_result["environment_space"], data_result["environment_cleaness"], \
    data_result["dish_portion"], data_result["dish_taste"], data_result["dish_look"] = np.array(processed.to_list()).T

    # 将以上结果进行保存
    s_path = "test/shortTexts-test.xlsx" if debug else "result/shortTexts.xlsx"
    data_result.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


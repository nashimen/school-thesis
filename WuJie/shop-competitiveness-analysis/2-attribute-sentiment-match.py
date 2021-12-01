import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False

dictionary_path = "result/domain_dictionary_20211028152823.npy"
origin_dictionary = np.load(dictionary_path, allow_pickle=True).item()


def initDictionary():
    dictionary = {}
    # 去重+统计个数
    count = 0
    for attribute, words in origin_dictionary.items():
        # words = words.split("，")
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


# 判断短文本属于哪个主题，根据dictionary
def judgeTopic(text):
    words = text.split(' ')
    topic_result = ''
    for word in words:
        # 遍历先验词库
        topic_result = "weight" if word in dictionary.get("weight") else "freshness" if word in dictionary.get("freshness") \
            else "color" if word in dictionary.get("color") else "cleanliness" if word in dictionary.get("cleanliness") \
            else "logistics" if word in dictionary.get("logistics") else "service" if word in dictionary.get("service") \
            else "taste" if word in dictionary.get("taste") else "price" if word in dictionary.get("price") \
            else "packaging" if word in dictionary.get("packaging") else "quality" if word in dictionary.get("quality") \
            else "shop" if word in dictionary.get("shop") else "EMPTY"

    return topic_result


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
    texts_dictionary = {"weight": [], "freshness": [], "color": [], "cleanliness": [], "taste": [], "logistics": [], "service": [], "packaging": [], "price": [], "quality": [], "shop": []}
    # 存在每条评论短文本（属性匹配后）对应的正向情感分，所有value的长度都一样。如果同一属性下有多条，则取平均值
    scores_dictionary = {"weight": [], "freshness": [], "color": [], "cleanliness": [], "taste": [], "logistics": [], "service": [], "packaging": [], "price": [], "quality": [], "shop": []}  # 未提及-1
    # 依次遍历每条评论&及其短文本
    for i in range(length):
        texts_dictionary_temp = {"weight": [], "freshness": [], "color": [], "cleanliness": [], "taste": [], "logistics": [], "service": [], "packaging": [], "price": [], "quality": [], "shop": []}  # 当前评论的变量
        scores_dictionary_temp = {"weight": [], "freshness": [], "color": [], "cleanliness": [], "taste": [], "logistics": [], "service": [], "packaging": [], "price": [], "quality": [], "shop": []}  # 当前评论的
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
    version = time.strftime('%Y%m%d%H%M%S', time.localtime(start_time))

    path = "test/2-score-test.xlsx" if debug else "result/2-score.xlsx"
    data_global = pd.read_excel(path)

    # 短文本-属性匹配
    texts_global, scores_global = shortText_attribute_match(data_global["shortTexts"].tolist(), data_global["score"].tolist())
    # 将以上结果进行保存
    for key, value in texts_global.items():
        data_global[key] = pd.Series(value)
        data_global[(key + "_label")] = pd.Series(scores_global.get(key))

    s_path = "test/3-attribute_sentiment-test.xlsx" if debug else "result/3-attribute_sentiment.xlsx"
    data_global.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


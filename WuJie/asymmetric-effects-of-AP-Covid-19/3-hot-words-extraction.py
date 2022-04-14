import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


# 生成停用词表
stop_words = []
f = open(file='stopwords.txt', mode='r', encoding='utf-8')  # 文件为123.txt
sourceInLines = f.readlines()
f.close()
for line in sourceInLines:
    temp = line.strip('\n')
    stop_words.append(temp)


def transMatrix(corpus):
    # print("corpus = ", corpus)
    print("corpus' length = ", len(corpus))
    print("corpus:", corpus)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print("weight:", weight)
    print("weight's length = ", len(weight))
    print("word's length = ", len(words))

    df_temp = pd.DataFrame()
    for i in range(len(weight)):
        tfidf_temp = []
        for j in range(len(words)):
            tfidf_temp.append(weight[i][j])
        df_temp["word"] = pd.Series(words)
        df_temp["tfidf"] = pd.Series(tfidf_temp)
        df_temp = df_temp.sort_values(by="tfidf", ascending=False)
        print(df_temp[:5])
        print("*" * 50)

    # sort = np.argsort(tfidf.toarray(), axis=1)
    # key_words = pd.Index(words)[sort].tolist()
    # print("key_words:", key_words)
    return df_temp


# 判断是否包含中文字符
def contain_Chinese(s):
    for c in s:
        if '\u4e00' <= c <= '\u9fa5':
            return True
    return False


def calculate_tfidf(texts):
    # print("texts:", texts)

    segs = []
    # 去标点符号→分词→去停用词
    for text in texts:
        for word in jieba.cut(text):
            if word not in stop_words and not word.isdigit() and contain_Chinese(word):
                segs.append(word)
        # segs.extend([word for word in jieba.cut(text) if word not in stop_words])
    return transMatrix([" ".join(segs)])


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    print(">>>正在读取数据。。。")
    path_global = "data/short texts for hot words extraction.xlsx"
    data_global = pd.read_excel(path_global, engine="openpyxl")

    attributes = ["Location", "Value", "Room", "Restaurant", "Safety", "Cleanliness", "Parking", "SleepQuality", "Bed", "Internet", "Service", "Decoration", "PublicFacility"]
    # 依次处理疫情前后的数据
    timeline = [0, 1]
    for current in timeline:
        if current == 0:
            print("正在处理疫情之前的数据")
        s_path_global = "data/test/hot-words-" + str(current) + "-test.xlsx" if debug else "result/hot-words-" + str(current) + ".xlsx"
        df = pd.DataFrame()
        print(df.columns)
        # 依次处理每个属性
        for attribute in attributes:
            label = attribute + "_label"
            if current == 0:
                texts_global = data_global.loc[(data_global["year"] < 2020) & (data_global[label] > -1)][attribute]
            else:
                texts_global = data_global.loc[(data_global["year"] >= 2020) & (data_global[label] > -1)][attribute]
            result = calculate_tfidf(texts_global.tolist())
            df = pd.concat([pd.DataFrame({attribute: result["word"]}), pd.DataFrame({label: result["tfidf"]}), df], axis=1)
        df.to_excel(s_path_global, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


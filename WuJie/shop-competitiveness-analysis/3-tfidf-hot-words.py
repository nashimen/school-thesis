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


def calculate_tfidf(texts):
    # print("texts:", texts)

    segs = []
    # 去标点符号→分词→去停用词
    for text in texts:
        for word in jieba.cut(text):
            if word not in stop_words and not word.isdigit():
                segs.append(word)
        # segs.extend([word for word in jieba.cut(text) if word not in stop_words])
    return transMatrix([" ".join(segs)])


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件
    print(">>>正在读取数据。。。")
    path_global = "data/data-focal-v3.xlsx"
    s_path_global = "test/4-hot-words-test-v3.xlsx" if debug else "result/4-hot-words-v3.xlsx"
    name_global = "原始数据"
    data_global = pd.read_excel(path_global, name_global)

    attributes = ["weight", "freshness", "color", "cleanliness", "taste", "logistics", "service", "packaging", "price", "quality", "shop"]
    df = pd.DataFrame()
    print(df.columns)
    # 依次处理每个属性
    for attribute in attributes:
        label = attribute + "_label"
        texts_global = data_global.loc[(data_global[label] < 0.5) & (data_global[label] > -1)][attribute]
        result = calculate_tfidf(texts_global.tolist())
        df = pd.concat([pd.DataFrame({attribute: result["word"]}), pd.DataFrame({label: result["tfidf"]}), df], axis=1)
        # df[attribute] = result["word"]
        # df[label] = result["tfidf"]
    # print(df[: 5])
    df.to_excel(s_path_global, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


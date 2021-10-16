import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


# 生成停用词表
stop_words = []
f = open(file='../stopwords.txt', mode='r', encoding='utf-8')  # 文件为123.txt
sourceInLines = f.readlines()
f.close()
for line in sourceInLines:
    temp = line.strip('\n')
    stop_words.append(temp)


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


# 数据处理，入参为list，去标点符号→分词→去停用词→去数字
def data_processing(content):
    result = []
    for line in content:
        # 去标点符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(line))
        # 去掉非中文字符
        # line = find_Chinese(line)
        # 分词→去停用词
        segs = ' '.join([word for word in jieba.cut(line) if word not in stop_words and not is_number(word)])

        result.append(segs)
    result = ' '.join(result)

    return result


def document_producing(data, all_columns):
    # 获取年份
    years = list(set(data["购买年份"]))
    # print("years = ", years)

    corpus = {}  # 存放整体的语料库（不区分属性）
    corpus_aspects = {}  # 存放不同属性的语料库{属性1：{2018：' ', 2019: ' '}, 属性2：{}}
    for year in years:
        print("正在处理", year)
        for col in all_columns:
            if col in ['类型', '购买年份']:
                continue
            data_year_col = data_processing(data.loc[data["购买年份"] == year][col].tolist())
            if col not in corpus_aspects.keys():
                corpus_aspects[col] = {}
            corpus_aspects[col][year] = data_year_col

    # 合并2017及之前的数据
    for col, values in corpus_aspects.items():
        year_2017 = ""
        for year, content in values.items():
            if year in [2021, 2020, 2019, 2018]:
                continue
            else:
                year_2017 = year_2017 + ' ' + content
        corpus_aspects[col][2017] = year_2017
        # 删除2017年之前的数据
        years = list(values.keys())
        for year in years:
            if year not in [2021, 2020, 2019, 2018, 2017]:
                values.pop(year)

    # 不区分属性数据保存
    for aspect, values in corpus_aspects.items():
        for year, content in values.items():
            if year not in corpus.keys():
                corpus[year] = ""
            corpus[year] += ' ' + content

    return corpus, corpus_aspects


# 计算tfidf
def calculate_tfidf(corpus, K, current_type, aspect=None):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus.values()))
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()

    result = pd.DataFrame(columns=["type", "aspect", "year", "word", "tfidf"])
    years = list(corpus.keys())
    for i in range(len(weight)):
        year = years[i]
        result_temp = pd.DataFrame(columns=["type", "aspect", "year", "word", "tfidf"])
        for j in range(len(words)):
            result_temp = result_temp.append([{"type": current_type, "aspect": aspect, "year": year, "word": words[j], "tfidf": weight[i][j]}])
        # 排序→取topK个词
        result_temp = result_temp.sort_values(by="tfidf", ascending=False)
        result = result.append(result_temp[: 5 if debug else K])

    return result


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 获取语料库（8个属性对应的评论）
    path = "test/新能源感知(2 types)-test_v2.xlsx" if debug else "data/新能源感知(2 types)_v2.xlsx"
    data = pd.read_excel(path, engine="openpyxl", usecols=[9, 13, 15, 17, 19, 21, 23, 25, 27, 36])
    all_columns = data.columns
    print("all_columns:", all_columns)

    result1 = pd.DataFrame(columns=["type", "year", "word", "tfidf"])  # 不分属性下的热词Top50
    result2 = pd.DataFrame(columns=["type", "year", "aspect", "word", "tfidf"])  # 分属性下的热词Top50

    types = set(data["类型"].tolist())
    for current_type in types:
        print("正在处理", current_type)
        # 分别处理当前类型的各个年份，每个年份数据为一个文档，每个年份为一个文档
        df1, df2 = document_producing(data.loc[data["类型"] == current_type], all_columns)

        topK = 150

        # 计算不分属性的tfidf
        print("开始计算不分属性的tfidf。。。")
        words_tfidf = calculate_tfidf(df1, topK, current_type)
        # 保存当前结果
        print("正在保存结果：", current_type)
        s_path1 = "test/hot_words_overall.csv" if debug else "result/hot_words_overall.csv"
        words_tfidf.to_csv(s_path1, mode="a", encoding='utf_8_sig', index=False)

        # 计算分属性的tfidf
        print("开始计算分属性的tfidf")
        for aspect, df2_values in df2.items():
            words_tfidf_2 = calculate_tfidf(df2_values, topK, current_type, aspect)
            # 保存当前结果
            print("正在保存结果：", current_type, aspect)
            s_path1 = "test/hot_words_aspect.csv" if debug else "result/hot_words_aspect.csv"
            words_tfidf_2.to_csv(s_path1, mode="a", encoding='utf_8_sig', index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


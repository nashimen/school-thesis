import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


# 读取领域先验词库
def get_field_words(path):
    result = {"comfort": [], "value": [], "power": [], "manipulate": [], "outside": [], "space": [], "energy": [], "inside": []}
    col_names = ['Attribute', 'Component']
    data = pd.read_csv(path, usecols=col_names)
    data = data.drop_duplicates(col_names)  # 只需要部件名称，因此去重
    # 将部件名称整合
    for key in result.keys():
        current_words = data.loc[data['Attribute'] == key]['Component'].tolist()
        result[key] = current_words
    return result


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


# 预处理为tfidf能处理的格式，例如：'车型 外观 好看'
def text_processing(content):
    result = []

    for line in content:
        # 去标点符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(line))
        # 去掉非中文字符
        # line = find_Chinese(line)
        # 分词→去停用词
        temp = []
        for word in jieba.cut(line):
            if word not in stop_words and not is_number(word):
                temp.append(word)
        if debug:
            temp = temp[: 5]
        segs = ' '.join(temp)
        # segs = ' '.join([word for word in jieba.cut(line) if word not in stop_words and not is_number(word)])

        result.append(segs)
    result = ' '.join(result)

    return result


# 生成所有文档
def produce_corpus(origin_data):
    corpus_purpose = pd.DataFrame(columns=["purpose", "aspect", "year", "corpus"])
    corpus_price = pd.DataFrame(columns=["price", "aspect", "year", "corpus"])
    purposes = set(origin_data["目的类别"].tolist())
    prices = set(origin_data["价格类别"].tolist())
    cols = origin_data.columns
    # 分别处理这两个类型
    # purpose
    for purpose in purposes:
        current_data_purpose = origin_data.loc[origin_data["目的类别"] == purpose]
        years = set(current_data_purpose["购买年份-fix"].tolist())
        for year in years:
            current_data_year = current_data_purpose.loc[current_data_purpose["购买年份-fix"] == year]
            # 处理每个属性
            for aspect in cols:
                if aspect in ["购买年份-fix", "来源类型", "价格类别"]:
                    continue
                print(purpose, year, aspect)
                current_data_year_aspect = current_data_year[aspect]
                current_data_year_aspect = text_processing(current_data_year_aspect)
                corpus_purpose = corpus_purpose.append([{"purpose": purpose, "aspect": aspect, "year": year, "corpus": current_data_year_aspect}])
    # purpose
    for price in prices:
        current_data_price = origin_data.loc[origin_data["价格类别"] == price]
        years = set(current_data_price["购买年份-fix"].tolist())
        for year in years:
            current_data_year = current_data_price.loc[current_data_price["购买年份-fix"] == year]
            # 处理每个属性
            for aspect in cols:
                if aspect in ["购买年份-fix", "来源类型", "价格类别"]:
                    continue
                print(price, year, aspect)
                current_data_year_aspect = current_data_year[aspect]
                current_data_year_aspect = text_processing(current_data_year_aspect)
                corpus_price = corpus_price.append([{"price": price, "aspect": aspect, "year": year, "corpus": current_data_year_aspect}])

    return {"purpose": corpus_purpose, "price": corpus_price}


# 计算tfidf
def calculate_tfidf(corpus, s_path, corpus_name):
    types = corpus[corpus_name].tolist()
    aspects = corpus["aspect"].tolist()
    years = corpus["year"].tolist()
    all_corpus = corpus["corpus"].tolist()

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(all_corpus))
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print("矩阵计算完成。。。")

    for i in range(len(weight)):
        result = pd.DataFrame(columns=[corpus_name, "aspect", "year", "word", "tfidf"])
        current_type = types[i]
        current_year = years[i]
        current_aspect = aspects[i]
        print(current_type, current_year, current_aspect)

        for j in range(len(words)):
            if weight[i][j] == 0:
                continue
            result = result.append([{corpus_name: current_type, "aspect": current_aspect, "year": current_year, "word": words[j], "tfidf": weight[i][j]}])
        result.to_csv(s_path, mode='a', index=False, encoding="utf_8_sig", header=None)

    return result


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    # 获取当天日期+小时+分钟，例如：202110161115
    version = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))

    # 读取领域先验词库
    path = "data/领域先验词库.csv"
    field_words_dict = get_field_words(path)

    # 获取语料库（8个属性的评论）
    path = "test/新能源感知-20211016-test.xlsx" if debug else "data/新能源感知-20211016.xlsx"
    data = pd.read_excel(path, usecols=[14, 16, 17, 19, 21, 23, 25, 27, 29, 31, 35])  # 分属性评论的语料库
    print(data.columns)

    # 获取所有类型×年份×属性的语料库
    print("正在处理语料库")
    corpus_global = produce_corpus(data)  # 分属性评论的语料库
    print("语料库处理结束")

    # 计算tfidf
    print("开始计算tfidf")
    # 分别计算购车目的和价格语料库的tfidf
    for current in ["purpose", "price"]:
        print("正在计算tfidf:", current)
        s_path_global = "test/importance_words_" + current + "_test-" + version + ".csv" \
            if debug else "result/importance_words_" + current + "-" + version + ".csv"
        df = calculate_tfidf(corpus_global.get(current), s_path_global, current)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


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
    corpus_result = pd.DataFrame(columns=["type", "aspect", "year", "corpus"])
    cols = origin_data.columns
    types = set(data["类型"].tolist())
    for current_type in types:
        current_data = origin_data.loc[origin_data["类型"] == current_type]
        years = set(current_data["购买年份-fix"].tolist())
        for year in years:
            current_data_year = current_data.loc[current_data["购买年份-fix"] == year]
            # 处理每个属性
            for aspect in cols:
                print(current_type, year, aspect)
                if aspect in ["购买年份-fix", "类型"]:
                    continue
                current_data_year_aspect = current_data_year[aspect]
                current_data_year_aspect = text_processing(current_data_year_aspect)
                # if debug:
                #     print("current_data_year_aspect:", current_data_year_aspect)
                # print(row)
                # corpus_result = corpus_result.append(row)
                corpus_result = corpus_result.append([{"type": current_type, "aspect": aspect, "year": year, "corpus": current_data_year_aspect}])

    return corpus_result


# 生成所有文档：最满意评论
def produce_corpus_v2(origin_data):
    corpus_result = pd.DataFrame(columns=["type", "year", "corpus"])
    types = set(data["类型"].tolist())
    for current_type in types:
        current_data = origin_data.loc[origin_data["类型"] == current_type]
        years = set(current_data["购买年份-fix"].tolist())
        for year in years:
            print(current_type, year)
            current_data_year = current_data.loc[current_data["购买年份-fix"] == year]["最满意评论"]
            current_data_year = text_processing(current_data_year)
            corpus_result = corpus_result.append([{"type": current_type, "year": year, "corpus": current_data_year}])

    return corpus_result


# 生成所有文档：最不满意评论
def produce_corpus_v3(origin_data):
    corpus_result = pd.DataFrame(columns=["type", "year", "corpus"])
    types = set(data["类型"].tolist())
    for current_type in types:
        current_data = origin_data.loc[origin_data["类型"] == current_type]
        years = set(current_data["购买年份-fix"].tolist())
        for year in years:
            print(current_type, year)
            current_data_year = current_data.loc[current_data["购买年份-fix"] == year]["最不满意评论"]
            current_data_year = text_processing(current_data_year)
            corpus_result = corpus_result.append([{"type": current_type, "year": year, "corpus": current_data_year}])

    return corpus_result


# 计算tfidf
def calculate_tfidf(corpus, s_path):
    types = corpus["type"].tolist()
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
        result = pd.DataFrame(columns=["type", "aspect", "year", "word", "tfidf"])
        current_type = types[i]
        current_year = years[i]
        current_aspect = aspects[i]
        print(current_type, current_year, current_aspect)

        for j in range(len(words)):
            if weight[i][j] == 0:
                continue
            result = result.append([{"type": current_type, "aspect": current_aspect, "year": current_year, "word": words[j], "tfidf": weight[i][j]}])
        result.to_csv(s_path, mode='a', index=False, encoding="utf_8_sig", header=None)
    # 删除0值
    # print("正在删除0值")
    # result = result[~result["tfidf"].isin([0])]

    return result


# 计算tfidf,最满意评论 最不满意评论
def calculate_tfidf_v2(corpus):
    types = corpus["type"].tolist()
    years = corpus["year"].tolist()
    all_corpus = corpus["corpus"].tolist()

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(all_corpus))
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print("矩阵计算完成。。。")

    result = pd.DataFrame(columns=["type", "year", "word", "tfidf"])
    for i in range(len(weight)):
        current_type = types[i]
        current_year = years[i]
        print(current_type, current_year)

        for j in range(len(words)):
            if weight[i][j] == 0:
                continue
            result = result.append([{"type": current_type, "year": current_year, "word": words[j], "tfidf": weight[i][j]}])
    # 删除0值
    # print("正在删除0值")
    # result = result[~result["tfidf"].isin([0])]

    return result


# 从wordTfidf中检索部件词语的tfidf值
def search_aspect_component_tfidf(component_tfidf, field_words_dict):
    df = pd.DataFrame(columns=("type", "year", 'aspect', 'component', 'tfidf'))
    # 检索哦
    for current_type, year, word, tfidf in zip(component_tfidf["type"], component_tfidf["year"], component_tfidf["word"], component_tfidf["tfidf"]):
        for aspect in field_words_dict.keys():
            components = field_words_dict.get(aspect)
            for component in components:
                if component == word:
                    df = df.append([{"type": current_type, "year": year, 'aspect': aspect, 'component': component, 'tfidf': tfidf}], ignore_index=True)

    return df


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取领域先验词库
    path = "data/领域先验词库.csv"
    field_words_dict = get_field_words(path)

    # 获取语料库（8个属性的评论）
    path = "test/新能源感知(2 types)-test_v2.xlsx" if debug else "data/新能源感知(2 types)_v2.xlsx"
    data = pd.read_excel(path, usecols=[9, 10, 13, 15, 17, 19, 21, 23, 25, 27, 37])  # 分属性评论的语料库
    # data = pd.read_excel(path, usecols=[9, 31, 37])  # 最满意评论
    # data = pd.read_excel(path, usecols=[9, 32, 37])  # 最不满意评论
    print(data.columns)

    # 获取所有类型×年份×属性的语料库
    print("正在处理语料库")
    corpus = produce_corpus(data)  # 分属性评论的语料库
    # corpus = produce_corpus_v2(data)  # 最满意评论
    # corpus = produce_corpus_v3(data)  # 最不满意评论
    print("语料库处理结束")
    print(corpus.head())

    # 计算tfidf
    print("开始计算tfidf")
    s_path = "test/hot_words-v2-test.csv" if debug else "result/hot_words-v2.csv"  # 分属性评论的语料库
    df = calculate_tfidf(corpus, s_path)
    # print(df)

    # 最满意评论 最不满意评论 需要匹配到属性
    # df = search_aspect_component_tfidf(words_tfidf, field_words_dict)
    # print(df.head())

    # save to file
    # s_path = "test/hot_words-satisfy-test.csv" if debug else "result/hot_words-satisfy-satisfy.csv"  # 最满意评论
    # s_path = "test/hot_words-unsatisfy-test.csv" if debug else "result/hot_words-unsatisfy.csv"  # 最不满意评论
    # df.to_csv(s_path, index=False, encoding="utf_8_sig")

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


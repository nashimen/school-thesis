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


# 保留中文字符，删除非中文字符
def find_Chinese(doc):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    Chinese = re.sub(pattern, "", doc)
    return Chinese


# 数据处理：合并最满意评论&最不满意评论→判断是否为空→判断是否为数字→去标点符号→分词→去停用词
def data_processing(content_satisfy, content_unsatisfy):
    result = []
    content = content_satisfy + content_unsatisfy
    # print(content)

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


# 分别处理当前类型的各个年份，每个年份数据为一个文档，计算tfidf
def calculate_tfidf(data):
    # 获取年份
    years = list(set(data["购买年份"]))
    print("years = ", years)

    corpus = {}
    year_2017 = ""
    for year in years:
        print("正在处理", year)
        data_year_satisfy = data.loc[data["购买年份"] == year]["最满意评论"].tolist()  # 最满意评论
        data_year_unsatisfy = data.loc[data["购买年份"] == year]["最不满意评论"].tolist()  # 最不满意评论
        data_year = data_processing(data_year_satisfy, data_year_unsatisfy)

        if year in [2021, 2020, 2019, 2018]:
            corpus[year] = data_year
        else:
            # 合并2017及之前的数据
            year_2017 = year_2017 + ' ' + data_year
            # print("year_2017:", year_2017)

    corpus[2017] = year_2017

    # 计算每个年份的component-tfidf
    component_tfidf = transMatrix(corpus)

    return component_tfidf


# 从wordTfidf中检索部件词语的tfidf值
def search_aspect_component_tfidf(current_type, component_tfidf, field_words_dict):
    df = pd.DataFrame(columns=("type", "year", 'aspect', 'component', 'tfidf'))
    # 检索哦
    for year, values in component_tfidf.items():
        for aspect in field_words_dict.keys():
            components = field_words_dict.get(aspect)
            for component in components:
                tfidf = values.get(component, 0)
                if tfidf > 0:
                    df = df.append([{"type": current_type, "year": year, 'aspect': aspect, 'component': component, 'tfidf': tfidf}], ignore_index=True)

    return df


# 计算TFIDF值，保存word-tfidf键值对，方便后面查询
def transMatrix(corpus):
    # print("corpus' length = ", len(corpus))
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus.values()))
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    # print("weight's length = ", len(weight))
    # print("word's length = ", len(words))

    result = {}  # 存放不同年份下的部件及其tfidf
    years = list(corpus.keys())
    for i in range(len(weight)):
        year = years[i]
        result_temp = {}
        for j in range(len(words)):
            result_temp[words[j]] = weight[i][j]  # 可能有0项
        result[year] = result_temp
        # print(result_temp)
        # print("*" * 50)

    return result


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取领域先验词库
    path = "data/领域先验词库.csv"
    field_words_dict = get_field_words(path)

    # 获取语料库（最满意评论&最不满意评论）
    path = "test/新能源感知(2 types)-test_v2.xlsx" if debug else "data/新能源感知(2 types)_v2.xlsx"
    data = pd.read_excel(path, engine='openpyxl', usecols=[9, 31, 32, 36])
    print(data.columns)
    # print(data.head())

    result = pd.DataFrame(columns=["type", "year", "aspect", "component", "tfidf"])
    types = set(data["类型"].tolist())
    for current_type in types:
        print("正在处理", current_type)
        # 分别处理当前类型的各个年份，每个年份数据为一个文档
        current_component_tfidf = calculate_tfidf(data.loc[data["类型"] == current_type])
        # 计算TFIDF值+匹配属性
        df = search_aspect_component_tfidf(current_type, current_component_tfidf, field_words_dict)
        result = result.append(df)

    s_path = "test/attribute_importance.xlsx" if debug else "result/attribute_importance.xlsx"
    result.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


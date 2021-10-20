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
def produce_corpus(origin_data, scope):
    # 根据参数，选择是否合并最满意评论&最不满意评论
    corpus_purpose = pd.DataFrame(columns=["purpose", "year", "corpus"])
    corpus_price = pd.DataFrame(columns=["price", "year", "corpus"])
    purposes = set(origin_data["目的类别"].tolist())
    prices = set(origin_data["价格类别"].tolist())
    # 分别处理这两个类型
    # purpose
    for purpose in purposes:
        current_data_purpose = origin_data.loc[origin_data["目的类别"] == purpose]
        # print(current_data_purpose.columns)

        years = set(current_data_purpose["购买年份-fix"].tolist())
        for year in years:
            current_data_year = current_data_purpose.loc[current_data_purpose["购买年份-fix"] == year]
            print(purpose, year)
            if scope == "最满意评论":
                print("当前处理：最满意评论")
                content = current_data_year.loc[current_data_year["购买年份-fix"] == year]["最满意评论"].tolist()
            elif scope == "最不满意评论":
                print("当前处理：最不满意评论")
                content = current_data_year.loc[current_data_year["购买年份-fix"] == year]["最不满意评论"].tolist()
            else:  # 合并 最满意&最不满意评论
                print("当前处理：最满意&最不满意评论")
                content = current_data_year.loc[current_data_year["购买年份-fix"] == year]["最满意评论"].tolist() + \
                          current_data_year.loc[current_data_year["购买年份-fix"] == year]["最不满意评论"].tolist()
            # data_satisfy = current_data_year.loc[current_data_year["购买年份-fix"] == year]["最满意评论"].tolist()
            # content = data_satisfy + data_unsatisfy
            current_data_year_content = text_processing(content)
            # if debug:
            #     print("current_data_year_aspect:", current_data_year_aspect)
            # print(row)
            # corpus_result = corpus_result.append(row)
            corpus_purpose = corpus_purpose.append([{"purpose": purpose, "year": year, "corpus": current_data_year_content}])

    # purpose
    for price in prices:
        current_data_price = origin_data.loc[origin_data["价格类别"] == price]
        # print(current_data_price.columns)

        years = set(current_data_price["购买年份-fix"].tolist())
        for year in years:
            current_data_year = current_data_price.loc[current_data_price["购买年份-fix"] == year]
            print(price, year)
            if scope == "最满意评论":
                content = current_data_year.loc[current_data_year["购买年份-fix"] == year]["最满意评论"].tolist()
            elif scope == "":
                content = current_data_year.loc[current_data_year["购买年份-fix"] == year]["最不满意评论"].tolist()
            else:  # 合并 最满意&最不满意评论
                content = current_data_year.loc[current_data_year["购买年份-fix"] == year]["最满意评论"].tolist() + \
                          current_data_year.loc[current_data_year["购买年份-fix"] == year]["最不满意评论"].tolist()
            current_data_year_content = text_processing(content)
            corpus_price = corpus_price.append([{"price": price, "year": year, "corpus": current_data_year_content}])

    return {"purpose": corpus_purpose, "price": corpus_price}


# 从wordTfidf中检索部件词语的tfidf值
def search_aspect_component_tfidf(source, year, component_tfidf, corpus_name):
    df = pd.DataFrame(columns=(corpus_name, "year", 'aspect', 'component', 'tfidf'))
    # 检索哦
    for aspect in field_words_dict.keys():
        components = field_words_dict.get(aspect)
        for component in components:
            tfidf = component_tfidf.get(component, 0)
            if tfidf > 0:
                df = df.append([{corpus_name: source, "year": year, 'aspect': aspect, 'component': component, 'tfidf': tfidf}], ignore_index=True)

    return df


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


# 读取领域先验词库
path = "data/领域先验词库.csv"
field_words_dict = get_field_words(path)
# 计算tfidf
def calculate_tfidf(corpus, s_path, corpus_name):
    sources = corpus[corpus_name].tolist()
    years = corpus["year"].tolist()
    all_corpus = corpus["corpus"].tolist()
    # if corpus_name == "price":
    #     print(all_corpus)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(all_corpus))
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print("矩阵计算完成。。。")

    for i in range(len(weight)):
        result_temp = {}  # 每个语料库对应一个
        result = pd.DataFrame(columns=[corpus_name, "year", "word", "tfidf"])
        current_source = sources[i]
        current_year = years[i]
        print(current_source, current_year)

        for j in range(len(words)):
            if weight[i][j] == 0:
                continue
            # if debug:
            #     print(words[j])
            # result_temp[words[j]] == weight[i][j]
            result_temp.update({words[j]: weight[i][j]})  # 记录所有word-tfidf键值对
            # result = result.append([{"source": current_source, "year": current_year, "word": words[j], "tfidf": weight[i][j]}])
        # 属性匹配
        result = search_aspect_component_tfidf(current_source, current_year, result_temp, corpus_name)
        result.to_csv(s_path, mode='a', index=False, encoding="utf_8_sig", header=None)

    return result


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    # 获取当天日期+小时+分钟，例如：202110161115
    version = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))

    path = "test/新能源感知-20211016-test.xlsx" if debug else "data/新能源感知-20211016.xlsx"
    data = pd.read_excel(path, usecols=[14, 16, 35, 36, 37])
    print(data.head())

    # 获取所有 购车目的×年份&价格×年份的语料库
    print("正在处理语料库")
    corpus_scope = "最不满意评论"
    print("正在处理：", corpus_scope)
    corpus_global = produce_corpus(data, corpus_scope)  # 全局变量+“_global”
    print("语料库处理结束")

    # 计算tfidf
    print("开始计算tfidf")
    # 分别计算购车目的和价格语料库的tfidf
    for current in ["purpose", "price"]:
        print("正在计算tfidf:", current)
        s_path_global = "test/importance_words_" + corpus_scope + "_" + current + "_test-" + version + ".csv" \
            if debug else "result/importance_words_" + corpus_scope + "_" + current + "-" + version + ".csv"
        df = calculate_tfidf(corpus_global.get(current), s_path_global, current)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


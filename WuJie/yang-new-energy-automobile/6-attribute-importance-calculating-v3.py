import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True


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
    # 合并最满意评论&最不满意评论
    corpus_result = pd.DataFrame(columns=["source", "year", "corpus"])
    sources = set(origin_data["来源类型"].tolist())
    for source in sources:
        current_data_source = origin_data.loc[origin_data["来源类型"] == source]
        print(current_data_source.columns)

        years = set(current_data_source["购买年份-fix"].tolist())
        for year in years:
            current_data_year = current_data_source.loc[current_data_source["购买年份-fix"] == year]
            print(source, year)
            # data_satisfy = current_data_year.loc[current_data_year["购买年份-fix"] == year]["最满意评论"].tolist()
            data_unsatisfy = current_data_year.loc[current_data_year["购买年份-fix"] == year]["最不满意评论"].tolist()
            # content = data_satisfy + data_unsatisfy
            current_data_year_content = text_processing(data_unsatisfy)
            # if debug:
            #     print("current_data_year_aspect:", current_data_year_aspect)
            # print(row)
            # corpus_result = corpus_result.append(row)
            corpus_result = corpus_result.append([{"source": source, "year": year, "corpus": current_data_year_content}])

    return corpus_result


# 读取领域先验词库
path = "data/领域先验词库.csv"
field_words_dict = get_field_words(path)
# 计算tfidf
def calculate_tfidf(corpus, s_path):
    sources = corpus["source"].tolist()
    years = corpus["year"].tolist()
    all_corpus = corpus["corpus"].tolist()

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(all_corpus))
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print("矩阵计算完成。。。")

    for i in range(len(weight)):
        result_temp = {}  # 每个语料库对应一个
        result = pd.DataFrame(columns=["source", "year", "word", "tfidf"])
        current_source = sources[i]
        current_year = years[i]
        print(current_source, current_year)

        for j in range(len(words)):
            if weight[i][j] == 0:
                continue
            if debug:
                print(words[j])
            # result_temp[words[j]] == weight[i][j]
            result_temp.update({words[j]: weight[i][j]})  # 记录所有word-tfidf键值对
            # result = result.append([{"source": current_source, "year": current_year, "word": words[j], "tfidf": weight[i][j]}])
        # 属性匹配
        result = search_aspect_component_tfidf(current_source, current_year, result_temp)
        result.to_csv(s_path, mode='a', index=False, encoding="utf_8_sig", header=None)

    return result


# 从wordTfidf中检索部件词语的tfidf值
def search_aspect_component_tfidf(source, year, component_tfidf):
    df = pd.DataFrame(columns=("source", "year", 'aspect', 'component', 'tfidf'))
    # 检索哦
    for aspect in field_words_dict.keys():
        components = field_words_dict.get(aspect)
        for component in components:
            tfidf = component_tfidf.get(component, 0)
            if tfidf > 0:
                df = df.append([{"source": source, "year": year, 'aspect': aspect, 'component': component, 'tfidf': tfidf}], ignore_index=True)

    return df


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 获取语料库（最满意&最不满意评论）
    path = "test/新能源感知(2 types)-test_v2.xlsx" if debug else "data/新能源感知-20211006_v2.xlsx"
    data = pd.read_excel(path, usecols=[10, 33, 34, 39])  # 分属性评论的语料库
    print(data.columns)

    # 获取所有类型×年份×属性的语料库
    print("正在处理语料库")
    corpus = produce_corpus(data)  # 分属性评论的语料库
    print("语料库处理结束")
    print(corpus.head())

    # 计算tfidf
    print("开始计算tfidf")
    s_path = "test/importance_words(是否国产)-20211006-test.csv" if debug else "result/importance_words_unsatisfy(是否国产)-20211006.csv"  # 分属性评论的语料库
    df = calculate_tfidf(corpus, s_path)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")



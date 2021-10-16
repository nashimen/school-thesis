import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 200


# 根据日期，给每条数据打标签，例如读取到
def season_fix(day):
    # day_time = datetime.datetime.strptime(day, "%Y/%m/%d")
    year = day.strftime("%Y")
    month = day.strftime("%m")
    if month in ["01", "02", "03"]:
        season = year + "S1"
    elif month in ["04", "05", "06"]:
        season = year + "S2"
    elif month in ["07", "08", "09"]:
        season = year + "S3"
    else:
        season = year + "S4"
    return season


# 根据传入的评论语料，计算热词
# tfidf top30
def calculate_hot_words(corpus):
    hot_words_result = {}  # 热词
    hot_result = {}  # 热度/tfidf

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus.values()))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    quarters = list(corpus.keys())
    # print("quarters:", quarters)
    for i in range(len(weight)):
        quarter = quarters[i]
        df = pd.DataFrame()
        tfidf_temp = []
        for j in range(len(word)):
            tfidf_temp.append(weight[i][j])
        df["word"] = pd.Series(word)
        df["tfidf"] = pd.Series(tfidf_temp)
        df = df.sort_values(by="tfidf", ascending=False)
        hot_words_result[quarter] = ' '.join(df["word"][:5 if debug else 30].tolist())
        hot_result[quarter] = df["tfidf"][:5 if debug else 30].tolist()

    if len(hot_words_result) != len(hot_result):
        print("热词的长度和热度的长度不一致。。。")
        sys.exit(-1)

    return hot_words_result, hot_result


# 根据传入的评论语料，计算频繁词,tfidf
def calculate_frequent_words(corpus):
    frequent_words_result = {}  # 频繁词
    frequency_result = {}  # 频率

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    counter = vectorizer.fit_transform(corpus.values()).toarray()
    word = vectorizer.get_feature_names()

    quarters = list(corpus.keys())
    for i in range(len(counter)):
        quarter = quarters[i]
        df = pd.DataFrame()
        frequency_temp = []
        for j in range(len(word)):
            frequency_temp.append(counter[i][j])
        df["word"] = pd.Series(word)
        df["frequency"] = pd.Series(frequency_temp)
        df = df.sort_values(by="frequency", ascending=False)
        frequent_words_result[quarter] = ' '.join(df["word"][:5 if debug else 30].tolist())
        frequency_result[quarter] = df["frequency"][:5 if debug else 30].tolist()

    if len(frequent_words_result) != len(frequency_result):
        print("频繁词的长度和频繁度的长度不一致。。。")
        sys.exit(-2)

    return frequent_words_result, frequency_result


# 根据传入的评论语料，计算新词,tfidf
# 计算逻辑：判断当前阶段的词在前两个时间段中是否出现过
# t-1&t-2均未出现→0.8，t-1未出现&t-2出现→0.5，t-1出现&t-2未出现→0.2
# 参考魏伟 情报学报
#
def calculate_new_words(corpus):
    new_words_result = {}  # 新词
    new_result = {}  # 新颖度

    quarters = list(corpus.keys())
    # print("quarters:", quarters)
    for i in range(len(quarters)):
        df = pd.DataFrame()
        quarter = quarters[i]
        current_quarter_data = corpus.get(quarter)
        if i < 2:
            new_words_result[quarter] = ""
            new_result[quarter] = []
        else:
            pre_quarter_data = corpus.get(quarters[i - 1])  # 前一个时间段的数据
            if i == 1:  # 当i=2时，如果前一阶段没出现过，则看做新词0.5，否则不是新词
                new_words_result_temp, new_result_temp = judge_new_words(current_quarter_data, pre_quarter_data)
                new_words_result[quarter] = ' '.join(new_words_result_temp)
                new_result[quarter] = new_result_temp
            else:
                pre_pre_quarter_data = corpus.get(quarters[i - 2])  # 前两个时间段的数据
                new_words_result_temp, new_result_temp = judge_new_words(current_quarter_data, pre_quarter_data, pre_pre_quarter_data)
                new_words_result[quarter] = ' '.join(new_words_result_temp)
                new_result[quarter] = new_result_temp

    if len(new_words_result) != len(new_result):
        print("新词的长度和新度的长度不一致。。。")
        sys.exit(-1)

    return new_words_result, new_result


# 判断新词的函数
def judge_new_words(current_data, pre_data, pre_pre_data=None):
    new_words_result = []
    new_result = []
    # 先转换为list格式
    current_data = current_data.split(" ")
    pre_data = pre_data.split(" ")
    if pre_pre_data is None:  # 如果只有两个时间段比较时
        for word in current_data:
            if word not in pre_data:
                new_words_result.append(word)
                new_result.append(0.5)
    else:  # 如果有三个时间段比较时
        pre_pre_data = pre_pre_data.split(" ")
        for word in current_data:
            if word in pre_data and word in pre_pre_data:
                continue
            else:
                new_words_result.append(word)
                if word not in pre_data and word not in pre_pre_data:
                    new_result.append(0.8)
                elif word not in pre_data and word in pre_pre_data:
                    new_result.append(0.5)
                elif word in pre_data and word not in pre_pre_data:
                    new_result.append(0.2)
    return new_words_result, new_result


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
# 处理单条评论
def preprocessing(line):
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

    return line


# 数据预处理，返回分季度的语料库
# 去标点符号→去非中文字符→分词→去停用词
def text_processing(current_data, all_seasons):
    result = {}
    for season in all_seasons:
        reviews_processed = []
        reviews_origin = current_data.loc[current_data["入住季度"] == season]["评论文本"].tolist()
        for review in reviews_origin:
            temp = preprocessing(review)
            reviews_processed.append(temp)
        reviews_processed = ' '.join(reviews_processed)
        result[season] = reviews_processed

    return result


# 统计点赞数
def calculate_like_num(data, seasons):
    result = {}
    for season in seasons:
        temp = data.loc[data["入住季度"] == season]["点赞数"].tolist()
        result[season] = sum(temp)

    return result


# 计算评论数量
def calculate_review_num(data, seasons):
    result = {}
    for season in seasons:
        result[season] = len(data.loc[data["入住季度"] == season]["评论文本"])

    return result


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取数据，切分时间段
    path = "data/test/merged_file+补充数据-test.xlsx" if debug else "data/original/merged_file+补充数据.xlsx"
    data = pd.read_excel(path, nrows=debugLength if debug else None)
    print("数据读取完毕。。。")
    data["入住季度"] = data.apply(lambda row: season_fix(row["入住日期"]), axis=1)

    all_hotels = list(set(data["名称"]))

    # 计算当前酒店在不同时间段内的词汇特征
    for hotel in all_hotels:
        final_result = pd.DataFrame(columns=["location", "star", "hotel", "quarter", "review_number", "like_number",
                                             "hot_words", "hot", "frequent_words", "frequency", "new_words", "new"])
        print("正在处理:", hotel)
        current_data = data.loc[data["名称"] == hotel]
        all_seasons = list(set(current_data["入住季度"]))
        all_seasons.sort()
        location = list(set(current_data["地区"]))
        star = list(set(current_data["星级"]))
        if len(location) > 1 or len(star) > 1:
            print("不同地区有同名酒店：", location, ",", star)
        location = location[0]
        star = star[0]
        # print("all_seasons = ", all_seasons)
        # 数据预处理
        corpus = text_processing(current_data, all_seasons)  # corpus为dict，包含当前酒店在不同时间段内的语料

        # 计算热词
        hot_words, hot = calculate_hot_words(corpus)
        # print(hot_words)
        # print(hot)

        # 计算频繁词
        frequent_words, frequency = calculate_frequent_words(corpus)
        # print(frequent_words)
        # print(frequency)

        # 计算新词
        new_words, new = calculate_new_words(corpus)
        # print(new_words)
        # print(new)

        # 计算评论数量
        review_count = calculate_review_num(current_data, all_seasons)
        # print(review_count)

        # 统计点赞数,累加求和即可
        like_num = calculate_like_num(current_data, all_seasons)
        # print(like_num)

        # 将当前酒店数据添加入final_result
        for season in all_seasons:
            row = [location, star, hotel, season, review_count.get(season), like_num.get(season), hot_words.get(season),
                   hot.get(season), frequent_words.get(season), frequency.get(season), new_words.get(season), new.get(season)]
            final_result = final_result.append([row], ignore_index=True)

        # 每家酒店之后保存结果
        s_path = "data/test/lexicon_features.csv" if debug else "result/lexicon_features.csv"
        final_result.to_csv(s_path, mode='a', index=False, encoding="utf_8_sig", header=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


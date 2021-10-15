import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 显示所有列
pd.set_option('display.max_columns', None)


def dictTest():
    dictionary = {"2star": ["a"], "3star": ["b", "d"]}
    print(dictionary)
    dictionary["4star"] = ["dd", "bb"]
    print(dictionary)
    dictionary["2star"].append("df")
    print(dictionary)
    ll = ["a", "b"]
    np.save("test.npy", ll)
    ll_load = np.load("test.npy", allow_pickle=True).tolist()
    print(ll_load)
    print(type(ll_load))

    name = ["白玉兰酒店(上海磁悬浮总站店)"]
    path = "上海3星_hotel_done_file.npy"
    np.save(path, name)


def dfTest():
    path = "data/test/评论-北京2星.xlsx"
    data = pd.read_excel(path, nrows=2000)
    # print(data.head())
    hotelNames = set(data["名称"].tolist())
    print(hotelNames)
    current_hotel = data.loc[data['名称'] == str("O2轻奢酒店(北京立水桥地铁站店)")]
    # print(current_hotel.head())
    # for name in hotelNames:
    #     print(name)
    for index, row in data.iterrows():
        # print("row's type = ", type(row))
        # print("row:", row)
        print(row["名称"])


# 判断是否全部为非中文字符
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fa5':
            return True
    return False


def ChineseTest():
    '''
    ss = "中国人"
    sss = "中国人😱냉"
    ssss = "😱냉냉"
    print(no_Chinese(ss))
    print(no_Chinese(sss))
    print(no_Chinese(sss))
    '''
    docs = ["中国人", "中国人😱", "😱냉냉", "😱냉냉中"]
    for doc in docs:
        print("doc = ", doc)
        if not is_Chinese(doc):
            # print("delete", doc)
            docs.remove(doc)
    print(docs)

    print("删除非中文字符。。。")
    for doc in docs:
        find_Chinese(doc)


def find_Chinese(doc):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    un_Chinese = re.sub(pattern, "", doc)
    print(un_Chinese)


def indexTest():
    texts = ["a", "CC", "abc", "defss", "fefsa"]
    for i, text in enumerate(texts):
        print(i, text)


def plotting():
    names = ['5', '10', '15', '20', '25']
    x = range(len(names))
    y = [0.855, 0.84, 0.835, 0.815, 0.81]
    y1 = [0.86, 0.85, 0.853, 0.849, 0.83]
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    # pl.ylim(-1, 110)  # 限定纵轴的范围
    plt.plot(x, y, marker='o', mec='r', mfc='w', label=u'y=x^2曲线图')
    plt.plot(x, y1, marker='*', ms=10, label=u'y=x^3曲线图')
    plt.legend()  # 让图例生效
    plt.xticks(x, names, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"time(s)邻居")  # X轴标签
    plt.ylabel("RMSE")  # Y轴标签
    plt.title("A simple plot")  # 标题

    plt.show()


import datetime
def dateTransform():
    day = "2021/12/41"
    day_time = datetime.datetime.strptime(day, "%Y/%m/%d")
    print(day_time)
    day_change = day_time.strftime("%Y%m")
    print(day_change)


def silhouetteTest():
    #初始化原始数字点
    x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
    x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
    #X = np.array([x1,x2])
    X = np.array(list(zip(x1,x2))).reshape(len(x1), 2)
    labels = [1, 3, 2, 2, 2, 3, 2, 3, 2, 1, 2, 1, 1, 3]
    sc_score = silhouette_score(X, labels, metric='euclidean')
    print(sc_score)


def time_fix(day):
    # print("day's type = ", type(day))
    # day_time = datetime.datetime.strftime(day)
    # print("day = ", day)
    # day_time = datetime.datetime.strptime(day, "%Y/%m/%d")
    # print(day_time)
    day_change = day.strftime("%Y%m")
    return day_change


def year_fix(day):
    # day_time = datetime.datetime.strptime(day, "%Y/%m/%d")
    year = day.strftime("%Y")

    return year


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


def findStarLevel(hotels, datasets):
    economy = []
    luxury = []
    df = datasets[["名称", "星级"]]
    df = df.drop_duplicates(subset="名称", keep="first")
    for index, row in df.iterrows():
        # print(row)
        # print(type(row["星级"]))
        name = row["名称"]
        star = str(row["星级"])
        if name not in hotels:
            continue
        if star in ["2", "3"]:
            economy.append(name)
        else:
            luxury.append(name)
    return economy, luxury


debug = False
debugLength = 1000
def hotelStatistics():
    print("正在读取数据。。。")
    path = "data/original/merged_file+补充数据.xlsx"
    df = pd.read_excel(path, engine="openpyxl", nrows=debugLength if debug else None)
    print("数据读取完毕。。。")
    print("length = ", len(df))

    # df = pd.read_csv(path)
    df["入住月份"] = df.apply(lambda row: time_fix(row["入住日期"]), axis=1)
    df["入住年份"] = df.apply(lambda row: year_fix(row["入住日期"]), axis=1)
    df["入住季节"] = df.apply(lambda row: season_fix(row["入住日期"]), axis=1)
    # print("按照地区*星级统计每月都有的酒店。。。")
    print("时间格式处理完毕。。。")

    # print(df.head())
    df = df.loc[df["入住年份"].isin(["2020", "2019", "2021"])]
    print("*" * 50, "按年统计", "*" * 50)
    years = list(set(df["入住年份"].tolist()))
    print("years = ", years)
    hotels = list(set(df.loc[df["入住年份"] == years[0]]["名称"]))
    # print("hotels:", hotels)
    print("hotels' length = ", len(hotels))
    for i, year in enumerate(years):
        if i == 0:
            continue
        current = set(df.loc[df["入住年份"] == year]["名称"])
        hotels = list(set(hotels) & current)
    # print("final hotels:", hotels)
    print("final hotels' length:", len(hotels))
    # 2星&3星、4星&5星酒店都有哪些
    economy, luxury = findStarLevel(hotels, df)
    # print("economy:", economy)
    # print("luxury:", luxury)

    print("*" * 50, "按季节统计", "*" * 50)
    seasons = list(set(df["入住季节"].tolist()))
    print(seasons)
    hotels = list(set(df.loc[df["入住季节"] == seasons[0]]["名称"]))
    # print("hotels:", hotels)
    print("hotels' length = ", len(hotels))
    for i, season in enumerate(seasons):
        if i == 0:
            continue
        current = set(df.loc[df["入住季节"] == season]["名称"])
        hotels = list(set(hotels) & current)
    # print("final hotels:", hotels)
    print("final hotels' length:", len(hotels))
    # 2星&3星、4星&5星酒店都有哪些
    economy, luxury = findStarLevel(hotels, df)
    print("economy:", economy)
    print("economy's length = ", len(economy))
    print("luxury:", luxury)
    print("luxury's length = ", len(luxury))

    print("*" * 50, "按月统计", "*" * 50)
    months = list(set(df["入住月份"].tolist()))
    hotels = list(set(df.loc[df["入住月份"] == months[0]]["名称"]))
    # print("hotels:", hotels)
    print("hotels' length = ", len(hotels))
    for i, month in enumerate(months):
        if i == 0:
            continue
        current = set(df.loc[df["入住月份"] == month]["名称"])
        hotels = list(set(hotels) & current)
    # print("final hotels:", hotels)
    print("final hotels' length:", len(hotels))
    # 2星&3星、4星&5星酒店都有哪些
    economy, luxury = findStarLevel(hotels, df)
    # print("economy:", economy)
    # print("luxury:", luxury)


def meanTest():
    score = [0.1, 0.3, 0.5, 0.1, 0.2]
    print(np.mean(score))
    texts = ["hehe", "haha", "hello"]
    texts2 = []
    texts3 = ["hehe"]
    print(",".join(texts3))


def judgeTopic():
    num = randint(1, 6)
    print(num)
    if num == 1:
        print("ROOM")
    elif num == 2:
        print("SERVICE")
    elif num == 3:
        print("LOCATION")
    elif num == 4:
        print("SLEEP_QUALITY")
    else:
        print("VALUE")


import xiangshi as xs
from snownlp import SnowNLP
def similarityTest():
    input1 = ["楼下餐厅口味也不错"]
    input2 = ["餐厅"]
    input3 = ["房间"]
    print(xs.cossim(input1, input2))
    # print(xs.cossim(input2, input1))
    # print(xs.cossim(input2, input3))
    print(xs.cossim(input1, input3))

    # s = SnowNLP([[input1, input2], [input1, input3]])
    s = SnowNLP(input1)
    print(s.sim(u'口味'))


def maxTest():
    a = max(0.1, 0.5)
    print(a)


def dfTest():
    df = pd.DataFrame()
    dictionary = {"a": [1, 2, 3], "b": [3, 2, 1], "c": [4, 5, 1]}
    for key, value in dictionary.items():
        df[key] = pd.Series(value)
    print(df)


from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
def tfidfTest():
    corpus = [
        '食物 位置 车站',
        '早餐 餐厅',
        '餐厅 公交站',
        '早餐 车站',
    ]
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    print(word)
    print(weight)
    print("打印tfidf最高的词语")
    for i in range(len(weight)):
        df = pd.DataFrame()
        tfidf_temp = []
        for j in range(len(word)):
            tfidf_temp.append(weight[i][j])
        df["word"] = pd.Series(word)
        df["tfidf"] = pd.Series(tfidf_temp)
        df = df.sort_values(by="tfidf", ascending=False)
        print(df[:5])
        print("*" * 50)


def frequencyTest():
    corpus = [
        '酒店 位置 非常好 从 西站 北广场 出来 右转 即 到 紧邻 公交 始发站',
        '酒店 有 餐厅 有 早餐 好像 要 付费 我们 没在 酒店 用餐',
        '酒店 条件 还 可以 服务 质量 一般  就是 外面 有点 太 吵闹 早餐 应该 是 卖点 花样 较多 希望 本店 服务 质量 有待 提高',
        '性价比 很高 客房 服务 也 很好 入住 体验 还 不错',
    ]
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(X.toarray())


def dictTest():
    myDict = {"a": 1, "b": 1.5, "d": 0.6, "aa": 1.1}
    print(myDict.values())


def listSum():
    ll = [1, 2, 3, 5, 1]
    print(sum(ll))
    final_result = pd.DataFrame(columns=["location", "star", "hotel", "quarter", "review_number", "like_number",
                                         "hot_words", "hot", "frequent_words", "frequency", "new_words", "new"])


def digitTest():
    a = '100km'
    b = '12l'
    c = '2l'
    d = "121"
    e = "21"
    print(is_number(a))
    print(is_number(a) and a.isdigit())
    print(is_number(b))
    print(is_number(c))
    print(is_number(d))
    print(is_number(e))


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


def is_number2(s):
    try:
        complex(s)  # for int, long, float and complex
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    print("start...")

    digitTest()

    print("end...")


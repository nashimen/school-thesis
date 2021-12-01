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


import paddlehub as hub
def sentiment_calculating():
    senta = hub.Module(name='senta_bilstm')
    texts = ["腊肉非常好吃", "炒菜味道不错", "整体一般吧", "我们去了公园"]
    res = senta.sentiment_classify(data={"text": texts})
    print(res)
    # for text in texts:


# 文本相似度计算
import Levenshtein as lev
def similarity_calculating():
    texts = ["你我他", "他我你他", "你你你", "我我我"]
    dist = lev.distance(texts[0], texts[1])
    print(dist)
    print(len(texts[0]))
    # simialrity = 1 - (dist / max(len(texts[0].split(" ")), len(texts[1].split(" "))))
    simialrity = 1 - (dist / max(len(texts[0]), len(texts[1])))
    print(simialrity)

    # 遍历所有两两组合
    length = len(texts)
    for i in range(length):
        for j in range(i, length):
            if i == j:
                continue
            print("i = ", i, ",j = ", j)


def save_csv():
    path = "data/merged_file2.xlsx"
    # 读取数据
    data_global = pd.read_excel(path)

    # 增加日期，例如 202101、202010
    data_global["日期-fix"] = data_global.apply(lambda row: time_fix(row["入住日期"]), axis=1)
    data_global.to_excel("data/merged_file_month.xlsx")


# 种子词
seed_dictionary = {
    "freshness": "新鲜，生产日期，保质期，日期，变质，馊味儿，馊味，防腐，腐烂，腐败，陈旧，月份，新鲜出炉，鲜食，臭肉，保鲜，硝酸盐，鲜度，亚硝酸盐，期限，生鲜食品，鲜肉，长期保持，新鲜肉，六月份，二月份，八月份，腐臭，湿度，状态，多长时间，储存，冻猪肉",
    "color": "颜色，色泽，红色，鲜艳，发黄，色，成色，蓝绿色，浅褐色，润透，透亮，红润，晶莹剔透，晶莹，鲜红色，发黑，黑色，样子，熏色，黄亮，色料，品色，琥珀色，红亮，白色，色泽鲜明，图案，蛋黄，发黑，黄潢，金黄色，鲜红色，脱色，浅褐色，色相，原色，黄金，表色，鲜艳，增色，腊黄，黄色，红润，颜值，红色，金黄，油黄，乳白色，色彩，货色，厚色，色素，黄黄的，色调，暗红色，发黄，黑亮，透亮，流黄，彩色，深色，黄油，淡黄色，焦黄，外观，外观设计，产品设计",
    "cleanliness": "干净，卫生，异物，脏，异味，发霉，霉斑，霉点，脏兮兮，臭味，臭味儿，发臭，恶心，拉肚子，臭臭，肚子疼，肚子痛，肠胃炎，安全卫生，黑水，杂质，股味，股味儿，油污，污垢，污渍，脏东西，难闻，肉臭，泥巴，散发出，黄泥巴，天臭，脏味，净，污，臭水沟，油污，黑脏，发臭，安全卫生，肠胃炎，拉肚子，异物，粪土，变质",
    "taste": "味道，好吃，辣辣的，辣，咸，香，辣味，腊味，咸味，咸味儿，香味，香味儿，麻辣，不好吃，难吃，有点咸，口感，正宗，不腻，腻，酒味，酒味儿，甜味，甜味儿，地道，不肥，肥美，肥，肥肉，瘦肉，很浓，美味，微辣，微微辣，排骨，肥瘦，香肠，适中，味儿，肉质，油腻，太咸，太辣，太腻，太肥，太瘦，品尝，烟味，烟味儿，口味，甜味，甜味儿，鲜美，味道鲜美，腊肠，风味，脆骨，腊肉，甜口，滋味，滋味儿，咸太多，腌制品，精肥，精瘦，肥瘦相间，盐分，真香，肥肥，红辣椒，辣椒，瘦润肥，肥度，地地道道，腥味儿，原汁原味，盐分，香气，辣肉，鸡肉，蒸肉，骨肉，牛排，羊腿，脚跟，辣肉，鲜味，肉身，瘦点，香浓，甜度，咸点，巴适，浓香，咸香咸，麻椒味，鲜香，咸甜，太咸太咸，香精，肥肉，肥瘦，肉肥，香料，咸鲜肥，咸口，肥点，香甜可口，咸香超，色香，筋道，香咸微，精肥，米饭，火腿肠，焖饭，糯米饭，火腿，调味，菜品，调味品，烤肠，肉香",
    "logistics": "物流，快递，速度，太慢，很快，慢，快，收到，送货，收货，送快递，收快递，快递服务，发货，发货慢，发货快，配送，配送点，快递点，快递站，中通，京东物流，京东快递，圆通，百世，百世汇通，韵达，单号，邮费，快递费，包邮，免运费，不包邮，京准达，冷链，顺丰，邮政，外省，送快递，行货，货差，订货，退换货，速度慢，配点，物流业，神速，丰巢",
    "service": "服务，客服，态度，服务态度，优质服务，售后，售前，售后服务，售前服务，售中服务，售后处理，售后态度，冷漠，热情，不主动，不热情，很冷漠，态度差，态度好，态度不好，态度不行，口气，态度恶劣，服务质量，气愤，歉意，礼貌",
    "packaging": "包装，外包装，内包装，包装袋，密封，密封性，漏液，漏了，漏，小袋，变形，真空，真空包装，袋子，塑料包装，纸箱装，纸箱，包装品，封口，包装品，纸盒，自封袋，豪装，纸盒包装，纸盒子，套装，纸袋，保鲜袋，箱子，礼盒装，礼品盒，礼袋，大礼盒，肠盒，礼袋",
    "price": "大小，分量，重量，千克，kg，KG，Kg，g，克，一斤，二斤，半斤八两，三斤，500g，每斤，大大的，小小的，净重，价格，性价比，物美价廉，便宜，偏贵，贵，便宜，不便宜，不贵，优惠，优惠券，活动，买一送一，几块钱，廉价，块钱，贵点，便宜点，市场价，涨价，半价，特惠，全品券，节省，太值，值，价值，物超所值，物有所值，收费，价低",
    "quality": "不好，质量，质量一般，质量不好，质量太差，太差，好，差，总体，整体，物有所值，物超所值，好坏，不错，东西，产品质量，货品，诱人，太湿，差评，品质，产品品质，真品，正品，优质，成品，出品，特差，高端，食品质量，太假，注重质量，伪品，外观设计，废肉，材质，真肉，假肉，韧劲，给力，货给力，失望透顶，碎屑，真棒，太坑，太棒了，太棒，太次，含淀粉，淀粉，前胛",
    "shop": "店铺，店家，卖家，京东，品牌，体验，购物，公司，信誉，老字号，商家，农户，农民，电商助农，小作坊，作坊，小店，大店，品牌店，旗舰店，供货，批量生产，电商，店里，铺子，靠谱，店老板，厂商，商标，偏远地区，网店，老板，商城，平台，乡村，农村，民族特色，食品厂，人民日报，贵州，云贵川，江苏，中国，原产地内蒙，生产线，石家庄市"
}
dictionary = {}  # 最终的领域词典
all_words = set()  # 已存在的词汇
counter = 0
for key, value in seed_dictionary.items():
    value = value.split("，")
    all_words.update(set(value))
    # print("value:", value)
    dictionary[key] = set(value)
    counter += len(set(value))
    print(key, len(set(value)))
print("counter = ", counter)
print("dictionary:", dictionary)
print("*" * 100)


def similarity_calculating():
    word = "啊啊啊"
    attribute = "EMPTY"
    max_similarity = 0
    for key, value in dictionary.items():
        sim = 0
        for v in value:
            sim = max(sim, xs.cossim([word], [v]))
        if max_similarity < sim:
            max_similarity = sim
            attribute = key
        print(key, max_similarity)
    print("attribute:", attribute)


def dictionary_process():
    dictionary_temp = {'weight': {'大字', '大盘', '换大', '皇比', '小鱼', '部分', '小结', '洗三', '小牙', '大片', '销量', '小石头', '份量', '慎重考虑', '三观', '半死', '大块', '二星', '小口肠', '大股', '比较稳定', '零分', '比比', '大陆', '大锅', '重量', '小火', '小猪', '大写', '销售量', '比顺', '过分', '两码事', '大胆', '半斤', '太重', '大量', '轻', '小康社会', '小妹妹', '小册子', '比重', '大长', '第三世界', '大雨', '小虫子', '大伙', '小段', '用的', '小土豆', '大拇指', '大厨', '小片', '大山深处', '重点', '分贝', '小姐姐', '克重', '减轻负担', '比鲜', '小哥', '小册', '小腿', '轻重', '小姐', '半肥半', '重麻', '半星', '大点', 'g', '双重', '千克', '大会', '的士', '小姨', '大众', '相克', '小草', '三星', '比松桂', '小包', '前半部', '克刚', '肉量', '大火', '买的', '小节', '大门', '小区', '皇大', '比率', '装分', '杨大爷', '大人', '三鲜', '足量', '大家', '称重量', '大力', '大件', '小丫头', '新比', '比美', '大兴区', '小心', '含水量', '大蒜', '饭量', '大峡谷', '大腿', '裝袋量', '大头菜', '小条', '克', '三斤', '大众化', '每箱', '大门口', '皇的', '小刀', '大子', '大头', '掌柜的', '克干', '500g', '大师', '评分', '数量', '大餐', '比较突出', '大口', '大房子', '睁大眼睛', '小镇', '重庆', '大肉', '大润发', '大学', '比肉', '三思', '三省', '工作量', '大油', '大姐', 'Kg', '北大荒', '平分', '小时', '分层', '小事', '泡大杯', '大雪', '巧克力', 'kg', '力量', '大太', '盐重', '皮上的', '大派', '太小', '年轻人', '水量', '大家伙', '比瘦', '大赞', '一斤', '大大的', '小石子', '小肠', '小家伙', '须三思', '半斤八两', '层次分明', '味重', '公分', '水分', '大肠', '小时候', '定量', '小米', '三锅', '小肉', '小菜', '方大', '小孩子', '小块', '分量', '小酌', '任重道远', '成分', '小孩', '小城', '大牌', '小伙伴', '净重', '需要的话', '大爷', '比例', '大气', '小朋友', '小兵', '大包', '红白分明', '总量', '二老', '二斤', '小手', '小伙子', '区分', '王大', '大方', '充分保证', '大热天', '毛重', '大哥大', '百分制', '分部', '的哥', '小吃', '重阳节', '成分表', '含水分', '斤两', '半桶水', '网络小说', '小提示', '长大', '小巧', '含量', '大叔', '半透明', '大运', '大吉', '大袋', '小金刚', '小排', '大料', '二手货', '大哥', '每斤', 'KG', '分干', '大小', '正量', '大力支持', '小小的'}, 'freshness': {'食盐', '馊味', '坏果', '质地', '浓肉', '浓味', '碎肉', '烂点', '硬味', '食物', '盐味', '猪肚', '多长时间', '新世界', '腐败', '肉包', '酸奶', '肉太柴', '长度', '味会', '自食', '酱牛肉', '吉林长春', '吃度', '全肉', '状况', '短时间', '哺乳期', '防腐剂', '酒肉', '鸭肉', '牛肉', '保证体系', '九月份', '精肉', '麻婆豆腐', '五肉', '肉灌', '油味', '猪骚', '份儿', '肉馅', '鲜度', '肉闻', '额度', '陈货', '袋肉', '皇广味', '食客', '硝酸盐', '湿气', '内质', '储备物资', '味带', '纯肉', '有效期', '肉肉', '宽度', '前段时间', '鲜食', '怪味', '亚硝酸钠', '深度', '湿包', '食者', '上学时', '肉片', '时间', '汽油味', '段时间', '日用', '鲜', '食欲', '本质区别', '冻笋', '白肉', '酸酸', '肉块', '产期', '味肉', '人造肉', '后味', '出厂日期', '饮食习惯', '盐度', '老肉', '时效', '五月份', '多长', '日晒', '湿油', '变质', '靓肉', '二月份', '肉段', '猪场', '钾肉', '衣食', '性味', '臀肉', '新意', '后食', '保管费', '力度', '放心肉', '味烧', '保姆', '肉摊', '房间', '饮食', '保鲜剂', '花肉', '十一月份', '肉用', '角度', '浓选肉', '时侯', '天肉', '食材', '卤肉饭', '生肉', '味全', '息肉', '时会', '猪蹄', '果仁', '储蓄卡', '先存', '肉味', '买肉', '豆腐干', '新冠', '储藏', '时肉', '直播间', '干度', '土味', '腐烂变质', '湿度', '物质', '臭肉', '形态', '价保', '肉酱', '日期', '甲肉', '间隔', '寒酸', '信任度', '食用油', '日光', '时放', '腐臭', '豆腐', '期', '猪头', '档期', '新奇', '过度', '鲜肉', '猪崽', '油肉', '保质期', '肉源', '嫩肉', '食用盐', '时用', '长根', '食后', '皮肉', '十月十日', '连肉', '心态', '厚度', '熏肉', '肉能', '咸猪肉', '食谱', '衣食住行', '味蒸', '腌肉', '后肉', '年份', '生产日期', '野猪', '信用度', '有酸味', '同肉', '豆腐渣', '时刻', '存货', '亚硝酸盐', '块状', '太烂', '变相', '猪婆肉', '食味', '肉膜', '货味', '年味', '单肉', '双份', '盐份', '猪婆', '时所', '味放点', '肉价', '苹果', '味足', '变味', '鹅肉', '硝盐', '保质保量', '八例', '长期保持', '酱肉', '膻味', '丘肉', '六星', '紧密度', '血豆腐', '食料', '水份', '松干度', '炒酸笋', '皇肉', '牛肉饼', '蒸食', '新货', '肉孜', '新鲜味', '新旧', '长条', '时尚', '食用', '费时间', '份上', '长沙', '果木', '鲜血', '满意度', '本味', '质问', '酸豆角', '长痘痘', '羊肉', '效果', '猪肉', '库存', '日子', '零食', '状', '力肉', '干湿度', '新春佳节', '新岗', '期限', '夹肉', '新鲜度', '烤肉', '条长', '猪母', '肉选', '肉食', '卤味', '偏麻肉', '冷度', '将肉', '炖肉', '臘肉', '蜡肉', '权限', '肉花', '烟肉', '肉丁', '变份', '计时', '气味', '冰冻', '鲜剂', '猪瘟', '偶然间', '消食', '戒肉', '五花肉', '海鲜', '拿肉', '六月份', '陈', '保质', '肉粒', '新鲜出炉', '指肉', '回锅肉', '胸肉', '肉菜', '猪血', '月份', '笋长', '霉味', '时泡', '食用方法', '质感', '部份', '余味', '行鲜', '馊味儿', '熏味', '过份', '豆腐乳', '保温杯', '牛肉干', '味觉', '烧食', '皮质', '豆腐皮', '新鲜肉', '保价', '粮食', '子骚味', '腐烂', '硬度', '丝状', '新肉', '养猪', '片状', '湿润', '延时', '肉表', '肉太', '健康长寿', '柴味', '纸质', '形状', '鲜笋', '腐肉', '保险', '卖肉', '杂味', '八月份', '生猪肉', '肉放点', '肉感', '存点', '保真', '陈旧', '成度', '股肉', '奇异果', '肉汁', '小猪猪', '热度', '素质', '肉丝', '月饼', '绞肉机', '会长', '广味', '切肉', '六楼', '太度', '干肉', '味精', '药味', '方肉', '伙食', '盐肉', '肉酸', '肉汤', '时令', '猪尿味', '含肉', '防腐', '扣肉', '熟肉', '腐乳', '新春', '亚', '烂肉', '股份', '下肉', '储存', '肉眼', '豬肉', '状态', '鲜藕', '味质', '肉类', '猪猪', '新疆', '鲜甜', '好长时间', '肉因', '种猪', '卤肉', '长辈', '肉压', '猪皮', '程度', '肉皮', '过段时间', '生活用品', '猪肝', '腐臭味', '胶味', '红肉', '浪费时间', '骚味', '酱豆腐', '新鲜', '肉铺', '味能', '猪味', '成份', '保安', '淋巴肉', '酱味', '苦味', '生鲜食品', '评肉', '川味', '嫩味', '肌肉', '味蕾', '菜肉', '肉沫', '肉肠', '肉糜', '防腐涂料', '美食', '冻猪肉', '过长', '较长时间', '泡肉', '肉切段', '苏味', '胶状物', '酸味', '臊味', '保鲜', '肉儿', '时期', '选肉', '水果', '肉长', '皇味', '画质'}, 'color': {'都油', '红外', '解决方案', '识货', '货为', '油炸', '货写', '油亲', '外形', '吸油', '个人观点', '松花蛋', '图样', '外观', '发酸', '成本', '做油', '带子', '外公', '黄老', '酱油', '厚入', '块腊制', '卖货', '货', '浮油', '太高腊', '橘子', '蚝油', '红润', '里子', '油黄', '蛋白', '样样', '红线', '盘子', '鸽子', '金额', '黄纸', '发黑', '客观', '柿子椒', '色泽鲜明', '坨油', '嗓子', '熏色', '示样', '色正味', '无货', '外皮', '备货', '黑易', '元素', '红糖', '末子', '老牌子', '风腊', '淘货', '长白', '鸡蛋', '黄鳝', '金华市', '馅料', '估子', '荤素', '外地', '色调', '阵子', '口子', '色泽', '卖相', '白花钱', '满屋子', '单子', '样式', '成片', '油流', '油儿', '太白', '虫子', '屯货', '材料', '样子', '表色', '乳头', '柏子树', '奖学金', '流油', '干油', '黄油', '筷子', '产品设计', '外用', '红色', '渣子', '外表', '皮挺厚', '油脂', '发展潜力', '网红饭', '熏熏', '腊鱼', '子蛤', '流口水', '蛋包饭', '成都', '油水', '乳房', '广腊', '预料中', '成型', '子骨刀', '蒸蛋', '油热', '买腊', '仪表盘', '刷子', '补货', '屋子里', '次货', '煤油', '包子', '彩页', '汉流', '白色', '蛋清', '汤油', '黄色', '皮厚', '货时', '柏丫熏', '给我发', '白嘴儿', '脖子', '发黄', '颜值', '红萝卜', '红包', '绝子', '主观', '白水', '太油', '白玉', '代表性', '绿豆汤', '表层', '外壳', '红火', '靓货', '边角料', '基本相同', '表情', '的腊', '哈喇子', '哨子', '锤子', '配料', '图说', '物料', '有货', '水流', '不雅观', '金融', '盐料', '圆白菜', '蛋炒饭', '菌子', '流黄', '松子', '油亮', '焦黄', '黄里', '说明书', '后油', '运货', '证明', '外箱', '鸭蛋', '皇腊', '暗红色', '金光', '百货', '金豆', '相片', '白开水', '装货', '货蛮', '白豆', '白花花', '外观设计', '凉调', '货物', '种子', '白菜', '图片吧', '黄亮', '金盏', '柴米油盐', '土货', '费油', '实践证明', '肉色', '附图', '深色', '红葡萄酒', '压仓货', '茄子', '花样', '女孩子', '成丁', '子女', '哈子', '沟通交流', '包金', '不油', '白砂糖', '壳子', '白糖', '交流', '佐料', '黑糖', '货色', '红薯', '焦炭', '乌龙', '菜油', '鲜红色', '皮色', '点货', '货闻', '意料', '黄光', '沙子', '深藏', '抹料', '茭白', '幌子', '豆子', '地摊货', '百货商店', '厚道', '油膏', '油会', '用油', '柴米油盐酱醋茶', '皮太厚', '发条', '色素', '油感', '鸭子', '因素', '预料', '过腊', '流速', '券货', '腊制', '明白', '货刚', '头发', '脱盐', '白饭', '胡图图', '坑货', '白汤', '色彩', '亮点', '脑子', '熏透', '外国', '货运', '有油', '将货', '金华', '蛋羹', '外婆家', '压货', '黑胡椒', '选料', '货是', '真货', '柜子', '浅褐色', '牌子', '发消息', '蓝绿色', '晶莹', '红白', '时油', '燃料', '货太', '烟熏', '泡发', '深山老林', '年货', '把子', '色椒', '流浪', '有料', '很漂亮', '外层', '金黄', '粽子', '缺货', '饺子', '成色', '货后', '黄瓜', '广发', '脱色', '外盒', '全白', '傻子', '电子称', '面子', '强子', '外罩', '金字', '蛋黄', '透明胶', '原色', '货咯', '鬼子', '调剂', '黄黄的', '腊鸭', '电子秤', '鲜艳', '表面', '叶子', '绿豆', '丸子', '下脚料', '外地人', '腊货', '用料', '熏烟', '废料', '油汁', '彩云', '色料', '色相', '头子', '观感', '能腊透', '外理', '流水线', '妹子', '足金', '菜籽油', '肠子', '油太', '开发票', '油炒', '蛋花汤', '透亮', '流水', '逊色', '笋子', '油润', '滋润', '货光', '鞋子', '彩', '黑店', '实物图片', '沙发', '货放', '淡黄色', '晶莹剔透', '村子', '儿子', '金钱', '白酒', '镊子', '绿色', '红肠', '油渣', '饲料', '腌透', '红烧', '成块', '金黄色', '脚丫子', '素菜', '朴素无华', '颜色', '和腊', '外婆', '水货', '摊子', '光泽', '红艳', '流程', '色', '特色', '肘子', '图案', '老子', '案板', '照腊片', '义乌', '机油', '橙子', '言表', '神腊', '配料表', '吃货', '白萝卜', '金装', '品色', '金玉其外', '柴油', '猪油', '美观', '观是', '假货', '放油', '图片', '表皮', '润透', '白跑', '蹄子', '拿油', '皮子', '油光', '火红色', '明友', '黄金', '切末图', '冒油', '答案', '乳白色', '焦', '屋子', '货己', '荷包蛋', '白胡子', '炭黑', '腌腊', '货顶', '给面子', '爆油汁', '油出', '黄潢', '油锅', '货真', '榨油', '计划', '特色小吃', '白嘴', '油是', '黄金时间', '辅料', '金牌', '黑色', '美颜', '调料', '直流', '白如', '逃脱责任', '方案', '爪子', '油层', '太黑', '图解', '黑斑', '图文', '红亮', '股子', '白条', '黄沙', '金子', '熊样', '店子', '断货', '腊黄', '粮油', '成就', '白搭', '皇和金', '厚色', '干货', '哥子', '彩色', '特色美食', '增色', '熏烤', '蛋壳', '腿子', '乌', '熏肠', '琥珀色', '黑猪', '图货', '代表', '现金', '黑乌乌', '蜀腊记', '随货', '孩子', '酸熏', '空调', '股油', '黑点', '院子', '饮料', '相符合', '面料', '图示', '少油', '皮油', '金东', '熏黑', '柴熏', '货都', '火油', '黑亮', '表象'}, 'cleanliness': {'伤心', '心意', '老爷爷', '泥土', '降点', '定点', '内心', '温水', '高点', '老头', '礼物', '爱心', '力点', '卡拉', '黑水', '水兆', '水泡', '莫拉', '心塞', '物理', '巴登', '男生', '都巴', '孝心', '水蒸', '花生米', '注意安全', '试水', '老主顾', '脏东西', '泥巴', '皇老', '毛太', '异客', '核心', '淋巴', '炒点', '植物', '心动', '花生糖', '特点', '水会', '钟点工', '老公', '水才', '肚子疼', '皮太干', '隔天', '杂毛', '杂面', '信心', '老酒', '时放点', '老菜', '巴滋', '老实', '换水', '网点', '丁点', '凉水', '心坎', '天才', '股味', '土豆片', '老区', '今天下午', '长老', '物总', '巴掌', '锅巴', '变质', '天通', '蜂蜜水', '土豆泥', '杀菌', '鹽巴', '萝卜干', '梅干菜', '关心', '毛豆', '要点', '干菜', '长天', '老母', '土豆', '费水', '水浸泡', '水灾', '老广', '实点', '正水', '天广', '冲天', '水滴', '干巴', '皇点', '心愿', '带点', '脏', '天宝', '短点', '物资', '老道', '长霉', '土土', '话巴', '豆干', '才干', '心理', '异地', '干饭', '细闻', '卫生纸', '貨物', '老干妈', '景点', '异味', '备点', '口水', '心理作用', '罗干', '干净利落', '消毒水', '老爹', '老姜', '热天', '空心', '难闻', '人物', '老客户', '拉稀', '土法', '土特产', '用心', '天南海北', '拉肚子', '正点', '污渍', '缺点', '血水', '匠心', '盐水', '臭味', '老伴', '土猪', '方心', '干柴', '过水', '笋干', '肚饿', '老爸', '老家', '晒干', '神干', '生气', '老百姓', '异乡', '老丈人', '肚子痛', '天津', '雨水', '天门', '散架', '老妈', '水暖', '忠心', '动物', '天呐', '长毛', '心灵', '毛病', '生姜', '站点', '美心', '肠胃', '恶心', '尾巴', '用点', '医生', '点儿', '黑脏', '杏干', '生抽', '买巴', '下巴', '嘴巴', '人生', '天臭', '干猪', '毛太长', '瘦干', '发霉', '水准', '多闻', '民心', '点菜', '饮用水', '远点', '清水', '股酒', '名闻', '水患', '老汉', '沸水', '地点', '女生', '买点', '汁水', '巴托', '干净', '老老少少', '软点', '羽毛球拍', '山清水秀', '冷水', '花点', '白点', '有心人', '老朋友', '发臭', '肚子', '馊臭', '蒸放点', '加点', '泡水', '老人', '白毛', '太干', '油污', '土产', '风干', '天得', '老妈妈', '沥干', '本心', '老人家', '热干面', '杂牌', '土豪', '热水', '心想', '老资格', '猪毛', '土特', '老婆', '生霉', '意难', '天时地利', '肉臭', '水收干', '饼干', '散发出', '饭放点', '中心', '门卫', '先生', '回老家', '异常现象', '臭水沟', '干笋', '老祖宗', '前水', '精心', '洪水', '天气', '衍生物', '水开后', '本土', '晚点', '脏兮兮', '干肠', '老年人', '老牌', '老鼠', '偏心', '炎炎夏日', '省心', '隔水', '热心', '脏味', '细心', '水煮开', '天猫', '烧毛', '屯点', '故土', '老乡', '死难', '泥沙', '霉斑', '生长', '评水', '朝天椒', '污垢', '叶干', '霉菌', '心仪', '中点', '水平', '炖点', '填充物', '长点', '异物', '老忠粉', '放点', '腌渍', '灰心', '干拆包', '今天上午', '先点', '江南水乡', '淘米水', '卫视', '股海', '心幕', '白霉点', '黄泥巴', '加水煮', '拉拉', '安全卫生', '水军', '心心念念', '肚皮', '弄点', '飞毛腿', '死心', '泥蒿', '夹心', '少放点', '点点', '杂质', '管理水平', '粪土', '巴实', '赞点', '克拉拉', '心情', '优点', '心家', '物会', '水电费', '干身', '老师', '出点', '心寒', '老少皆宜', '物体', '混合物', '肠胃炎', '毛孔', '净含量', '净', '先水', '霉点', '药水', '含水', '肺炎', '臭味儿', '臭臭', '加班加点', '实物', '淡点', '污', '用水', '老抽', '可点', '哈拉', '心目', '股儿', '股味儿', '柴沟堡', '棒点', '阿拉', '盐巴', '土生土长', '老父', '卫生', '评臭', '经水', '水浸', '列巴'}, 'taste': {'咸度', '不好吃', '长半肥', '辣鸡', '晚饭', '散肠', '芳香', '中意', '王中王', '真系', '胡椒粉', '酷感', '气候', '蒸箱', '亲身', '咸闻', '有点咸', '红辣椒', '太长', '原以为', '太美', '越香', '中多', '蒸菜', '烤箱', '精准', '香麻', '气管', '蒸后', '调味品', '地盖', '地地道道', '亲身经历', '皮占太', '肥筋', '口香', '松滋', '洋气', '中段', '太咸太咸', '香醇', '饭菜', '不太想', '关火', '不带', '排队', '菜苔', '地神', '肥', '烩面', '后脚', '咸麻', '感人', '五香', '辣跟', '羊脂', '超笋', '特地', '鲜美', '中买', '最肥', '勇气', '破地', '肉香', '气温', '腌菜', '比肥', '瘦润肥', '有券', '甜口', '责任感', '瘦肉精', '香盐', '皇蒸', '麻辣', '特咸', '腥味儿', '美女', '香甜可口', '咸菜', '玩意儿', '正宗', '货肥', '咸大咸', '楼道', '汤真牛', '蒸肉', '先香', '淡盐', '断面', '咸', '烟腻', '中奖', '酱汁', '香薰', '蛮肥', '猪排', '人口', '咸立丰', '有肥', '糯', '豆瓣儿酱', '肥肉', '排腊香', '气氛', '倍感', '美美', '辣口', '渠道', '脆骨', '香春', '肠衣', '碗饭', '正儿八经', '香味', '习酒', '泡菜', '香素', '泡制', '咸甜', '辣味', '咸淡', '瘦点', '强太', '香水', '川辣', '太太', '咸柴', '酸辣粉', '甜者', '火气', '真宗', '牛排', '稻香村', '咸中', '香干', '咸重', '滋味', '花椒', '芹菜', '太正粉', '甜酒', '香略', '广肠', '做菜', '花椒面', '半肥', '不象', '幸福感', '辣点', '不费牙', '女儿', '盐淹', '香咸微', '会儿', '牛皮纸', '动感', '醇美', '火把', '都太肥', '甜味儿', '开口', '不华算', '饭店', '香先', '咸皮', '舒适感', '甜度', '咸蛋', '口味', '肥皮', '油烟味', '辣爱', '香度', '价太', '菜鸟', '原味', '菜式', '硬骨头', '太阳', '地会', '太多生', '咸香咸', '清香', '太咸太', '倍儿', '咸淡适中', '放太', '排骨汤', '太迟', '全面', '骨头', '交口称赞', '费酒', '甜味', '咸香超', '瘦真', '微波炉', '火蹄', '火侯', '川菜', '咸咸咸', '醇香', '肥肠', '干辣椒', '香肠', '鸡肉', '不盐', '烟灰', '微辣', '过火', '烤肠肉', '甜放', '鸡腿', '原因', '酒味', '酒味儿', '麻是', '口服液', '花儿', '肥肥', '口袋', '火候', '口放', '香', '辣辣的', '香腸', '太咸', '湘菜', '那可真', '电饭煲', '啤酒', '等会儿', '宝贝儿', '辣油', '肠', '地直', '正事', '不桐', '香料', '公道', '腿肉', '太短', '火腿肠', '咸放点', '辣椒酱', '包香', '古香古色', '暴风雪', '菜花', '蒸碗', '辣肠', '香放饭', '肥瘦', '腥味', '腻', '不太麻', '肥太肥', '皮儿', '鲜味', '香蒸', '咸点', '赞美', '不腻', '清香味', '肥油', '咸味', '粤菜', '麻麻', '原汁原味', '感情', '咸太咸', '麻绳', '蔬菜', '行肥', '用鸡', '盐缸', '巨肥', '大米饭', '饭煲', '味道鲜美', '太精', '美味', '地瘦', '太甜太甜', '菜会', '太腻', '整腿', '甜肠', '酒缸', '太鼓', '节蜡肠', '太费', '香腊', '酱菜', '和筋', '问道', '尖椒', '当地人', '鲜香蒸', '辣和想', '会肥', '菜饭', '迟太多', '香用', '肥度', '酒气', '纤排', '油肥', '粉太', '养肥', '品味', '难吃', '太辣太咸', '火速', '烟盒', '腊香', '中评', '咸味儿', '太肥', '佐酒', '道理', '辣椒面', '菜刀', '香港', '气死', '全肥', '全身', '猪儿', '下口', '喜感', '米饭', '香锅', '伤感', '酸菜', '真肥', '甜香', '王道', '咸鲜肥', '蒸面', '香香', '好吃', '地带', '蒸压', '中加', '烟草味', '玉米汤', '汤汁', '制品', '精光', '偏麻', '地步', '烟火', '煤烟', '秋风', '字香', '烟薰', '地方风味', '香鲜', '急火', '芝麻', '咸干咸', '断口', '麻嘴', '咸香', '辣度', '里边儿', '运气', '咸量', '太空', '香赞', '农家饭', '酒店', '灌肠', '香味儿', '小咸', '硬感', '火腿肉', '顺口', '麻咸', '麻椒', '借口', '地位', '牛津', '饿瘦', '热米', '巴适', '香儿', '口音', '不太能', '风俗', '咸正', '香精', '麻味', '咸腊', '尼麻', '团圆饭', '鲜香', '火烧', '口齿', '块儿', '太辣', '市面上', '太咸挺', '风里雨', '糖精', '鸡桶', '太材', '香烟', '满正', '香熏', '认真负责', '热饭', '做面', '静美', '精华', '腊肠', '家常菜', '麻又辣', '肥会', '小辣椒', '炒面', '胃口', '肠身', '韧香', '有骨头', '鲜咸', '香颂', '个人感觉', '中哈', '饭碗', '酒家', '辣椒', '气炸', '菜椒', '霸气', '蒸', '辣妹', '面蒸', '珍馐美味', '米酒', '麻椒味', '原料', '真假', '甜带', '腊味', '端菜', '买菜', '地址', '醇正', '偏肥', '看运气', '托儿', '青椒', '入口', '风味', '品尝', '中山', '麦菜', '地域', '蜗牛', '肉制品', '骨肉', '肉肥', '滋味儿', '菜谱', '熏香', '晒太阳', '蒸会', '麻辣味', '油腻', '口肥', '全腿', '菜用', '龙骨', '美的', '回事儿', '色香味', '气息', '汤有', '界面', '香赞赞', '精肥', '下酒菜', '层次感', '口吃', '亲切感', '香纯', '咸辣为', '多得很', '皮太', '口服', '味香', '成肥', '肉身', '有益', '费饭', '热气', '香柴', '杭椒', '地道', '肉骨', '后米香', '梅菜', '松饭', '个菜', '香蕉', '炖菜', '肉太肥', '太盐', '儿菜', '排行榜', '连盐', '和蔼可亲', '放盐', '湘菜馆', '咸咸', '口罩', '松香', '地标', '弹感', '香酥', '面条', '辣肉', '灵气', '花椒粉', '香气', '烩菜', '别太肥', '风昧', '微带', '糯米饭', '客气', '鸡', '羊排', '橡皮筋', '味口', '做饭吃', '满口', '咸大', '道菜', '白米饭', '切口', '人间烟火', '微波', '原材', '咸也', '美丽', '不争气', '咸肉', '香加', '娃儿', '咸是正', '原故', '多汁', '辣人', '稀饭', '玩儿', '香菇', '炒青菜', '香椿', '香肥', '炒米', '太甜', '风格', '翻面', '身价', '原存', '咸特', '湖南菜', '味肥', '不太会', '咸都', '痛风', '事儿', '焖饭', '香肉', '合口味', '韭菜', '鸡汤', '咸咸的', '火腿', '手感', '火炕', '美其名曰', '来风', '火锅店', '口杯', '太牛', '软骨', '咸口', '甘香咸', '市面', '真是太', '咸超', '浓香', '调味', '咸得', '凉拌菜', '好肥', '口碑', '香甜', '尝品', '一番滋味', '柴香', '香咸', '距今已有', '道肥', '真让人', '肥肥的', '盐都', '气死人', '饭桌', '牛气', '咸拿', '劲儿', '吃盐', '菜品', '太少先', '肥美', '烟味', '菜板', '分肥', '后腿', '特肥', '真气', '饭馆', '胡椒', '不柴', '神秘感', '风冷', '多肥', '页面', '麻和辣', '调味剂', '原材料', '蜡肠', '盖饭', '太香', '味道', '红椒', '太普通', '饭局', '包菜', '太错', '巨咸', '不肥', '香浓', '肥点', '气人', '酒精', '地切', '真真正正', '油香', '全肥油', '太农', '色香', '咸鲜', '宜多吃', '颗粒感', '有面', '身影', '烟味儿', '余香', '烤串', '肥油史', '淳正', '原本', '辣得', '精力', '鸡精', '米线', '早饭', '饭', '原图', '羊腿', '不麻', '顺风', '辛辣味', '三肥', '辣麻味', '多太才', '菜肴', '精神', '米粉', '咸辣', '川香', '很浓', '瘦肉', '腊肉', '腿脚', '酱制', '猪脚', '咸焯', '火灾现场', '炒菜', '本地人', '酱香', '扶风', '太瘦', '原则', '真香', '不惯', '虾米', '相间', '尝个', '饮酒', '公正', '职业道德', '力气', '中入', '香和', '孬很', '糯米', '广椒', '香挺咸', '太久', '豆制品', '排骨', '脚跟', '决口', '黑太咸', '中辣', '太多', '刮风', '咸盐', '风险', '适中', '咸鱼', '香辛料', '酒香', '不亲', '菜蔬', '不太肥', '通风', '鱼排', '辣条', '正种', '味儿', '青菜头', '腌制品', '太辣太', '原生态', '街道', '肉质', '劲道', '火锅', '太真', '口粮', '有腊香', '方面', '凉菜', '多尔', '牙口', '血肠', '肥瘦相间', '牛奶', '不粉', '香油', '青菜', '太气', '酒肴', '咸水', '界面显示', '集美', '筋道', '辣椒素', '电饭锅', '皮太硬', '正口', '盐业', '满足感', '辣能', '肥块', '肥皂', '真牛', '蒸鱼', '原价', '腌臘', '花菜', '煲仔饭', '幽道', '胶原蛋白', '乳制品', '孬不孬', '大太咸', '太冷', '烧菜', '二肥', '玉米粒', '微微辣', '香嫩', '口感', '玉米面', '辣子鸡', '香美', '柴火', '侍酒', '咸太', '烟秋', '火石', '配菜', '焦香', '刀口', '肥温', '真香真', '精瘦', '榨菜', '辣酱', '咸太多', '方方正正', '烟笋', '头儿', '蔬菜汤', '太合', '身体', '精者', '肠皮', '口渴', '台风', '脸面', '硬道理', '辣白菜', '太柴', '太细', '中雨', '中午饭', '两碗饭', '辣', '盐分', '太精肥', '玉米', '中肥', '前腿', '微', '标太', '中华', '原谅', '太瘦太', '饭会', '剖面', '烤肠', '甘甜', '越肥', '香芹', '蒸锅', '浓', '脚踝', '太闲', '蕨菜', '地方', '忆乡甜', '肥太', '特香', '柴太', '用盐', '品香', '媳妇儿', '触感', '原装', '咸鸭蛋', '辣子', '肥闻', '太麻', '滋滋'}, 'logistics': {'不包邮', '纹路', '申通', '换货', '神器', '手机', '很快', '收验', '拼单', '河北省', '被包', '行帮', '快京准达', '哈站', '站式', '路费', '海运', '圆通', '百非', '企业', '普通型', '冷链', '快件', '鼓包', '百洁布', '配面', '节奏快', '换哈', '京东快递', '朝发夕至', '单帮', '百世', '龟速', '拖拉机', '省', '发货慢', '收件人', '配饼', '内行', '乘机', '成达达', '处理速度', '单第', '京东物流', '邮', '事业', '有神', '百合', '步行街', '退换货', '外省人', '旗号', '配送点', '立丰', '需送友', '包罗', '光速', '费事', '套路', '和顺', '工业', '账号', '神', '能省', '中通站', '买单', '收快递', '快', '飞机', '送快递', '速度慢', '超顺', '中国邮政', '普通级', '配点', '自行车', '战斗机', '收货', '皮包', '收货人', '荣业', '全送人', '人行', '路程', '型号', '发货快', '和立丰', '钱包', '冷藏', '员送件', '交通', '快递服务', '产业', '定单', '快递点', '快递', '路途', '单位', '收礼', '中通', '货送', '青神', '行情', '催单', '湖北省', '快递费', '邮件', '发货', '交学费', '快递站', '和利丰', '货差', '省钱', '面包店', '单价', '飞速', '物流', '至极', '拉链', '收件', '冷盘', '行货', '下单', '邮政', '慢', '物流业', '韵达', '浪费', '差货', '京准达', '外省', '政策', '手机号', '包邮', '敬业', '机构', '单号', '订单', '包皮', '发货单', '广东省', '速度', '百世汇通', '配篱', '行会', '邮费', '神额', '消费者', '关机', '送货', '神速', '顺丰', '走路', '神马', '名号', '百叶', '世界', '丰牌', '普通', '费时费力', '都行', '挂号费', '行行', '韵', '教程', '太慢', '工程', '地配点', '免运费', '省事', '单人', '工号', '单方面', '单才', '程序', '机会', '过程', '少包', '物流配送', '单啊', '熟门熟路', '实际行动', '订货', '信号', '世界级', '丰巢', '驿站', '单数', '省力', '收到', '换', '团圆', '黄中通', '行业', '百姓', '计算机', '专业', '配送', '机械化', '卡通', '机器人', '省时省力', '机器', '面包', '顺带', '包仔', '上路', '路边摊', '通话', '链接', '电话费', '车号', '浙江省', '买手机', '配方', '有机', '路况', '先朝', '神经', '丰都', '全世界', '礼包'}, 'service': {'人士', '后皮', '友情', '优质服务', '冷漠', '疫情', '坑人', '汛情', '宜人', '热情', '待客', '意思', '伊人', '工作人员', '人性化', '懒人', '热锅', '哥们', '售前', '售后', '过来人', '买客', '全家人', '礼合', '肌理', '业务', '员工', '人能', '山东人', '工人', '前槽', '店主人', '后台', '佳人', '态度差', '醉人', '客户', '态度好', '阴凉处', '人会', '亲友们', '家里人', '年礼', '谢谢', '理想', '无人', '乡情', '私人', '宰客', '售后态度', '后备箱', '后入', '主角', '理论', '家人', '服务态度', '感谢信', '延后', '态度不好', '歉意', '感谢', '人用', '售后处理', '理念', '态度', '我意', '人群', '员人', '衣服', '胡建人', '后湖', '理由', '做客', '人们', '纹理', '回头客', '情怀', '感谢电', '女主播', '同情', '后略', '人间', '令人', '华人', '常客', '顾客', '礼貌', '前泡', '亲们', '客人', '客家', '自由人', '泡后', '售中服务', '夫人', '熟人', '热狗', '意识', '经理', '不热情', '家务', '很冷漠', '人工', '亲人', '谢谢你们', '态度恶劣', '热会', '誘人', '后本', '帮别人', '疯人', '湖南人', '人才', '情况', '口气', '闽南人', '农人', '后会', '售后服务', '礼拜', '不主动', '人慎', '哈人', '会员', '崇礼', '售前服务', '媒人', '气愤', '服务质量', '员们', '服务中心', '做人', '玩意', '长沙人', '爱人', '员工福利', '员才', '同胞们', '多谢', '四川人', '南方人', '服务到位', '专员', '后备', '人意', '意义', '个人', '喜人', '服务公司', '人员', '意见', '钟意', '事情', '客服', '态度不行', '人理', '人说', '苏州人', '服务', '唐人', '任务'}, 'packaging': {'双拼', '真空', '包装', '罐装', '产品包装', '索性', '袋装', '企业形象', '双', '礼盒', '自提', '纸盒', '保鲜袋', '内层', '包装用', '实用性', '填写内容', '形', '纸包装', '漏液', '开袋', '封牢', '精装', '电冰箱', '内盒', '双门', '八袋', '皮袋', '钞箱', '报纸', '包装箱', '垃圾箱', '自封袋', '大礼盒', '胶袋', '急性', '漏', '形象', '漏了', '纸箱', '包装纸', '内袋', '内包装', '条装', '包装袋', '密蒙花', '袋子', '变形', '真空包装', '双层', '纸袋子', '盒马', '小袋', '塑料包装', '秘密', '大包装', '塑装', '礼品', '双节', '封口膜', '空间', '双方', '密封胶', '食品包装', '袋食', '塑料', '内涵', '纸壳', '姜袋', '手提袋', '搞了个', '打了个', '礼袋', '冰袋', '冰箱', '密封性', '纸包', '纸盒子', '包装盒', '能装', '内碎', '真空袋', '纸箱装', '塑料纸', '套装', '豪装', '空空', '封袋', '内容', '小袋装', '野蛮装卸', '肠盒', '双袋', '自带', '妹纸', '外包装', '涨袋', '拆袋', '密封', '封城', '封口', '纸袋', '整袋', '装备', '硬包装', '性能', '装的', '封箱', '礼品盒', '纸箱包装', '性命', '小包装', '漏发', '包装品', '漏气', '背箱', '输液', '纸盒包装', '箱子', '有形', '漏油', '自学', '餐盒', '形容', '装盘', '塑料袋', '盒子', '经常性', '弹性', '散长装', '空易', '开箱', '形式', '性赖', '习惯性', '合理性', '时效性', '礼盒装', '薄纸'}, 'price': {'规格', '一家人', '价是', '特具', '期望值', '低糖', '买光', '特棒', '节约', '偏贵', '价格不菲', '贵贵贵', '皆宜', '选块', '动手', '特产', '不值', '价实', '全是', '推荐值', '试一下', '资格', '半价', '贵点', '几节课', '市场行情', '优惠政策', '节省', '会场', '节奏', '市场价', '券用', '六块钱', '节有券', '特价', '全', '动静', '价平', '全都', '参考价值', '学一学', '购买价', '价美', '几秒钟', '母亲节', '家具市场', '涨工资', '特写镜头', '高价', '优惠券', '优惠价格', '严格把关', '便宜', '下块', '优先权', '方便面', '饵块', '面值', '价低', '物价', '贵人', '性价比', '一率', '价格合理', '价格公道', '优惠', '要值', '值', '动力', '一碗水', '全球', '价位', '优势', '所需', '派上用场', '用场', '现场直播', '安平县', '全部', '券', '活动', '惠顾', '奇特', '活扣', '全品券', '一颗颗', '那贵', '整块', '物超所值', '涨价', '评价', '上市', '一流', '正常值', '一闻', '太贵', '比价', '结钱', '买一送一', '亲自动手', '低温', '价格比', '物美价廉', '有块', '钱加', '一盘菜', '贵太', '了一', '价格便宜', '市场', '偏向', '廉价', '熏值', '贵阳', '不便宜', '美物廉', '价格低廉', '不贵', '细节', '特特特特', '猪价', '贵量', '产品价格', '极格', '特惠', '板块', '关节', '动能', '片场', '商品价格', '菜市场', '优惠价', '子一', '不太值', '全肋', '价能', '便宜点', '价值', '物有所值', '买块', '价格', '季节', '一星', '泡一泡', '一事', '同价位', '小贵', '贵店', '小贵送', '都市', '切块', '价廉', '真值', '节目', '偏酸', '考虑一下', '正值', '价比', '节瓜', '十块钱', '环节', '周全', '一分钱', '价格低', '合格', '动用', '有钱出钱', '收费', '齐全', '一碟', '狂欢节', '贵', '不值钱', '价比高', '来块', '花钱买', '降价', '一分货', '元霄节', '全长', '领券', '早市', '市场经济', '物美', '块钱', '宝贵', '公平', '一带', '佳节', '节省时间', '保证价格', '价钱', '全部都是', '现场', '冰块', '便宜货', '有所', '方便使用', '一家子', '太值', '小贵哈', '保值', '块块', '平价', '扁平', '涨莓', '一段距离', '查一查', '几块钱'}, 'quality': {'高端', '单品', '出场', '京粉', '品控', '新品', '档次', '代差', '高品质', '粉粉', '假', '选材', '强力', '顶顶顶', '比差', '用力', '好极', '出厂', '效率高', '品酒', '品质', '西芹', '太棒了', '质量一般', '碎粒', '写差', '高赞赞', '农产品', '次品', '评尼妈', '商品', '高的', '才品', '挺棒', '超棒', '高龄', '步步高升', '买品', '上品', '质量保证', '竞争力', '差肉', '假肉', '车西', '顶级', '化学品', '不好', '好孩子', '绵粉', '肉品', '棒棒', '碎末', '真肉', '伪品', '奶粉', '无力', '实力', '坑害', '必备品', '极差', '差白', '油品', '东肃肃', '文化用品', '注重质量', '用品', '员差', '食品质量', '好事', '高温', '木材', '品象', '不错', '劣质品', '买差', '产品编号', '高手', '劲', '嘉品', '淀粉', '压力', '高能', '质量', '差价', '总想', '集体', '阿西', '鄂西', '力龟', '劣质', '品商', '真差', '太湿', '劲会', '产品图片', '评论', '智力', '转粉', '好个屁', '出库', '好东东', '差', '暴力', '好物', '错爱', '整体', '样品', '忠粉', '品级', '极端', '小东西', '太假', '西班牙', '产品品质', '粉丝', '消失', '真棒', '主打产品', '体会', '棒极了', '味差', '商品经济', '化妆品', '步错', '品种', '外观设计', '品是', '级差', '地好', '真品', '保健品', '粉', '赔偿损失', '货给力', '差距', '品类', '错误', '废话', '高哈', '东北', '东西', '物超所值', '事差', '次数', '质量上乘', '员给力', '好友', '辅材', '半成品', '个差', '农副产品', '日用品', '助力', '评差', '好料', '东莞', '好味', '太差', '评语', '实体', '西南', '好坏', '好菜', '好品', '含淀粉', '废肉', '好心', '太棒', '损失', '产品质量', '合渣粉', '码好', '差品', '物品', '广东', '极品', '整煲', '评官', '佳品', '诱人', '错', '好搭档', '残次品', '差太', '魅力', '西昌', '加工品', '江西', '残次', '好歹', '保证质量', '前胛肉', '粉是', '产品', '差点', '质量太差', '质量第一', '出品', '失德', '家庭用品', '能力', '差差', '质量不好', '件物品', '好点', '物有所值', '高汤', '好略', '顶用', '出远门', '诱惑力', '正品', '细品', '货好', '特差', '圈粉', '初品', '总能', '中差', '化学药品', '失望透顶', '购买力', '品鉴', '出面', '太次', '待品', '好运', '铁粉', '给你个', '碎屑', '差异', '商品质量', '饮品', '瘦品', '太坑', '压力锅', '東西', '教材', '赠品', '东瓜', '总统', '肉差', '力赞', '备品', '品像', '胶东', '有力', '面粉', '新品种', '地给力', '货太坑', '满怀希望', '总会', '差评', '品藏', '了太甜', '优质', '高挺', '韧性', '力愿', '坑光', '不错呀', '卖力', '总体', '好膈', '高鸿', '实体店', '差错', '日常用品', '好评率', '棒极', '成品', '卖东西', '精品', '湘西', '韧劲', '西瓜', '系列产品', '清洁用品', '产品种类', '星差', '西兰花', '好', '给力', '总结', '好感', '护肤品', '粉面', '体系', '好闻', '体面', '出力', '奖品', '商品信息', '肩胛', '体力', '高压锅', '体贴', '惯总', '总算', '药品', '粉渣', '材质', '品相', '错觉', '酸粉', '西张庄', '货品', '差劲', '东东', '高质量', '前差', '前胛'}, 'shop': {'农历', '上市公司', '铺子', '中国', '公婆', '地产', '家用', '厂家', '宅家', '孙云峰', '江南', '制作方法', '购物', '来家', '知名品牌', '福南', '名牌', '商家', '浙江', '城南', '板鸭', '商场', '守信用', '亲民', '小店', '乃斯', '四川话', '少数民族', '枝江', '丽江', '当家的', '锅里煮', '文字', '信賴', '嘴里塞', '品牌形象', '体验', '广西', '食品厂', '产生', '河北地区', '骗人', '奶奶家', '区域', '京东方', '武蛮', '厂商', '阳台', '受骗上当', '故乡', '店名', '电器', '模板', '廣州', '石家庄市', '乡愁', '助农', '卖家', '安娜', '京东', '云南', '家常', '回娘家', '加工厂', '旗舰店', '信耐', '安徽', '湖南', '松兹馆', '罗卜', '往里面', '协商', '信懒', '皇里', '利川', '哈尔滨', '家宴', '农忙', '用户', '电视', '店信', '字', '信奈', '北京市区', '原产地', '广州', '朋友家', '电脑', '智商', '店铺', '门店', '走遍全国', '入户', '化验', '标准', '商业街', '姐信', '桂坊', '土家', '京都', '县城', '武汉', '国产', '工业化生产', '店长', '东京', '耒风', '线下', '居家', '批量生产', '家宅', '网络', '铺面', '京東', '娘家', '衡水', '哈那', '卡里', '衡水市', '讲信用', '杭州', '信赖', '昌盛', '农民', '安逸', '底线', '名誉', '山区', '电商助农', '凉雾乡', '江苏', '代工厂', '北京', '石头', '海报', '标题', '民族特色', '公公', '字帖', '标明', '恰坊', '凤馆', '亲戚家', '标识', '信誉', '恩施土家族苗族自治州', '馆', '家伙', '品牌店', '里哈', '公益', '偏远地区', '徐州', '公庄镇', '产线', '国家', '城市', '威武', '结石', '网站', '老板', '耶斯', '盘州', '平台', '著名品牌', '北京市', '家用电器', '制造商', '商店', '村镇', '专卖店', '耒', '食品类', '京豆', '思乡', '商标', '馆会', '购物车', '名字', '韩国', '商城', '买家', '地区', '店', '波斯', '卖方', '海南', '生产线', '武昌', '短信', '指南', '里蒙', '信用', '京华', '蒜苗', '家店', '家庭', '作坊', '安井', '老板娘', '电梯', '灶台', '团', '内蒙', '皇家', '字牌', '海信', '小作坊', '家乡话', '信息产业', '靠谱', '店主', '官网', '关公', '光线', '川式', '标签', '人民', '店网', '团部', '考古学家', '土家族', '晚安', '俄罗斯', '南瓜', '家族', '青城山', '字数', '购物网', '家婆', '供货', '奈斯', '江湖', '公司', '店商', '标打', '专家', '手信', '农村', '文化遗产', '博罗县', '质朴', '广式', '馆里', '回家', '标记', '农户', '凤凰古城', '南派', '商业化', '肉食品', '农药', '广告', '信任', '乡村', '老字号', '店家', '电话', '西安', '台湾', '肉联厂', '四州', '四川', '贵州', '灾区', '荆州', '家养', '农家', '安静', '松桂坊', '声誉', '耒风馆', '写字', '三家店', '电商', '皇牌', '信福满', '全家', '苏州', '南竹', '南粤', '集团', '生命线', '板', '南丁', '大店', '本店', '石慧', '风馆', '原产地内蒙', '钻石', '标志', '荆州市', '工厂化', '广进哈', '网友', '石锅', '鄂湘川', '合作伙伴', '福州', '回川', '办公室', '京仔', '商户', '广告宣传', '公猪', '全网', '生产厂家', '板板', '家乡', '乡间', '云贵', '家居', '云贵川', '产的', '信息', '购买者', '十字', '江津', '标的', '店老板', '里加', '诚信', '时铺', '家具', '区别', '上京', '宣城', '名牌产品', '食品级', '自治州', '离家', '柜里', '梅州', '离谱', '字眼', '人民日报', '昧着良心', '先农', '汤里', '非标', '网店', '经验', '联系', '用线', '全国', '产地', '网购', '家长', '坊', '品牌', '商铺', '京友', '食品', '网页', '店里', '苗族', '沃尔玛', '文字游戏', '民族'}}
    for key, value in dictionary_temp.items():
        value = "，".join(value)
        # print("\"" + key + "\"", value)
        print(key, value)


if __name__ == "__main__":
    print("start...")

    dictionary_process()

    print("end...")

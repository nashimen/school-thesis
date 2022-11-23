import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
import paddlehub as hub
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 30

origin_dictionary = {
    "Service": "客运站，同行，同学，村镇，休闲游，小车，模式，旅途，回程，大牌坊，船形，门罩，网红，明珠，地图，人间烟火，检票员，舌尖，状态，车位，素质，温度，师傅，形态，视频，小道，大宅子，木刻，早饭，战乱，大宅，精品，假日，鸡汤，汽车，气味，指示牌，自助机，乐趣，美照，地道，大人，政策，电话，散客，格局，市井，当地居民，午餐，面条，司机，农家，特色菜，野菜，咖啡店，咖啡馆，速度，垃圾桶，班车，老板娘，早餐，烧饼，原汁，餐馆，茶叶，店家，旺季，店铺，网络，人气，小店，大气，人群，县城，距离，水面，旅游团，态度，对联，垃圾，衣服，商店，刺史，宅院，小姐姐，家家户户，徽菜，遗憾，全镇，西门，路边，停车场，美食，牛肠，饭菜，口味，客人，糖果，行李，很多人，人们，公交，服务态度，专业，自驾游，美味，售票处，毛豆，人工，早点，索道，灯光秀，人员，小吃，交通，设施，酒吧，饭店，大巴，臭鳜鱼，导游，景点，讲解员，音乐，餐饮，车程，风味，餐厅，旅行团，高铁，路线，村民，线路，路面，年轻人，饮食，邮局，笋干，高铁站，咖啡，饭馆，汽车站，公交车，平台，客运，三轮车，小酒馆，服务员，机器，节目，工作人员，旅游车，水果，村民们，竹笋，臭桂鱼，腊肉，夜市，晚饭，食堂，小贩，牛肚，",
    "Hotel": "疫情，客栈，老板，房间，味道，民宿，酒店，空调，客房，管家，窗户，实名制，院子，墙面，实名，阳台，前台，房东，店主，卫生间，旅馆，装饰，厕所，",
    "Culture": "年间，底蕴，艺术，砖雕，巷子，小巷，灯光，夜景，家族，人文，故事，风格，代表，祠堂，布局，典型，黛瓦，季节，马头墙，白天，人家，古树，技艺，徽娘，宝库，地标，地位，象征，履福堂，烟火，纪念品，窄巷，标志性，民间，遗产，之三华里处，细节，小村庄，体验感，木结构，砖墙，世纪，古城，宗族，胡姓，厅堂，老街，规模，天井，门前，古巷，马头，古建，角落，工艺品，山村，米酒，石板，精神，人生，人间，声音，家训，习惯，本地，古迹，诗意，意外，灯笼，住宅，典范，沧桑，气势，建筑物，趣味性，商业街，慢生活，教育，石板路，可玩性，农村，基地，风土人情，美院，知识，敬修堂，爱国主义，油墨，高墙，青石，水幕，美誉，古色，经典，氛围，居民，民风，粉墙，品味，印象，建筑群，雕刻，街巷，建筑风格，牌楼，徽式，小桥，徽商，中国画，书院，承志堂，传统，徽派，古建筑，村落，文化，民居，文化遗产，徽派建筑，石雕，木雕，古村，历史，建筑，古村落，村子，世界，牌坊，气息，商业，村口，月沼，特色，白墙，乡村，古民居，学生，老宅，公元，内涵，结构，民俗，古人，老房子，名录，房屋，时光，徽派民居，重点，文物，小巷子，代表性，红灯笼，气氛，灰瓦，韵味，村庄，古镇，生活，佳作，后人，全貌，博物馆，年代感，敬爱堂，飞檐，徽派建筑风格，时代，财富，木质，气派，雕花，子孙，后代，宝地，举家，家风，千秋，小院，成就，民宅，屋檐，标志，光影秀，古风，家乡，门楼，商业味，青瓦，手艺，三畏堂，灵气，历史感，风韵，名宿，园林，雕饰，德义堂，有人文，老宅子，有意境，院落，文字，人物，古黟桃花源，古韵，故居，时节，祖先，牛角，大理石，生活习惯，心灵，年代，情怀，名声，文艺，遗产地，字体，官家，非遗，民族，白瓦，传说，古民居群，阁楼，大院，前世今生，微派，印记，官宅，老屋，遗迹，习俗，三雕，商铺，黑瓦，风情，古建筑群，工艺，商家，寓意，原住民，大户，岁月，小村落，古宅，特产，人文景观，青砖，乐叙堂，",
    "Nature": "溪水，观景台，好地方，城市，小桥流水，水乡，原味，自然景观，风水，色彩，风貌，画卷，景观，名气，精华，牛形，山色，大树，画里，塔川，秋天，景区，地方，环境，风景，美景，水墨画，自然，照片，夜晚，水墨，山坡，湖景，月亮，小河，背景，意境，晴天，胜地，大自然，天气，空气，山水，油菜花，流水，倒影，村头，好去处，雨天，荷塘，月份，一景，世外桃源，错峰，小雨，景象，蓝天白云，云雾，水墨丹青，外观，教育基地，山林，秋色，太阳，美如画，浓墨重彩，取景地，水渠，清泉，位置，喷泉，风光，山水画，水系，池塘，大片，观景亭，余晖，视觉，美术，天下，落日，镜头，绿水，河流，阴雨，墙面，桃花源，山路，河水，小山，群山，观景，雨声，高峰，地区，风景区，地理，水流，地势，地点，气候，水池，水质，晨雾，大早，场景，细雨，野趣，水景，灯火，江南古镇，冬季，薄雾，故宫，景色，情怀，面貌，油画，天堂，花园，环境保护，水塘，油菜，晚霞，湿地，向日葵，水沟，花海，风景线，大山，竹海，暴雨，天空，石桥，半月形，水利，诗情画意，垂柳，秋季，夕照，月湖，晨曦，秋景，夜色，青山绿水，日落，街道，仙境，魅力，青石板，荷花，湖面，田园，原生态，夕阳，溪流，色调，空间，小溪，全景，炊烟，小村，画面，青山，美感，蓝天，白云，烟雨，星空，景致，日出，南屏，阳光，湖水，湖光，春晓，环山，",
    "Price": "停车费，成人，小朋友，经济，生意，费用，商品，票机，车票，总体，性价比，套票，票价，身份证，联票，物价，节假日，半价，一个人，价值，价格，门票，便宜，天地，价钱，折扣，定价，"
}


# 加载停用词
def getStopList():
    stoplist = pd.read_csv(filepath_or_buffer='../stopwords.txt').values
    return stoplist


# 判断是否包含中文字符
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


# 读取文件，文件读取函数
def read_file(filename):
    # with open(filename, 'rb')as f:
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        # 返回list类型数据
        text = text.split('\n')
    return text


# 读取停用词表
stop_words = read_file(r"../stopwords.txt")
print('origin stop length: ' + str(len(stop_words)))


# 去停用词函数
def del_stopwords(words):
    # 去除停用词后的句子
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words


# 提取短文本
def get_abstract(content):
    # print("content = ", content)
    # 使用逗号替换空格
    content = re.sub(r" +", ",", str(content))
    pattern = r',|\.|;|\?|:|!|\(|\)|，|。|、|；|·|！| |…|（|）|~|～|：|；'
    docs = list(re.split(pattern, content))
    # print("docs pre = ", docs)
    docs = list(filter(lambda x: len(str(x).strip()) > 0, docs))
    # 去掉不包含中文字符的短文本
    for doc in docs:
        if not is_Chinese(doc):
            print("非中文：", doc)
            docs.remove(doc)

    return docs


# 加载数据+提取短文本
def load_data(path):
    current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")

    # 提取短文本
    print("正在提取短文本")
    current_data["shortTexts"] = current_data.apply(lambda row: get_abstract(row['评论文本']), axis=1)
    print(current_data.head())
    # 删除空白行
    colNames = current_data.columns
    current_data = current_data.dropna(axis=0, subset=colNames)
    # 保存文件
    s_path = "test/6 raw data + shortTexts-v3.xlsx" if debug else "data/6 raw data + shortTexts-v3.xlsx"
    current_data.to_excel(s_path, index=False)


def calculate_sentiment_by_row(texts):
    texts = texts.strip('[')
    texts = texts.strip(']')
    texts = texts.replace("'", "")
    texts = texts.replace(" ", "")
    texts = texts.split(',')
    input_dict = {"text": texts}
    res = senta.sentiment_classify(data=input_dict)
    # print("res:", res)
    positive_probs = []
    for r in res:
        positive_probs.append(r['positive_probs'])
    if len(texts) != len(positive_probs):
        print("长度不一致！！出错辣！！！")
        print(positive_probs)
        print(texts)
    return positive_probs


# 删除没有情感分数的文本
def remove_texts_scores(path):
    current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")
    print("正在删除没有情感分数的文本。。。")
    current_data[['score-fix', 'shortTexts-fix']] = current_data.apply(lambda row: remove_10086_texts(row['score'], row['shortTexts']),
                                                                       axis=1, result_type='expand')
    s_path = "test/6 raw data + score-v4.xlsx" if debug else "data/6 raw data + score-v4.xlsx"
    current_data.to_excel(s_path, index=False)


# 删除-10086的文本,即没有情感分数的文本
def remove_10086_texts(scores, texts):

    scores = scores.strip('[')
    scores = scores.strip(']')
    scores = scores.replace(" ", "")
    scores = scores.split(',')
    # print("scores:", scores)

    texts = texts.strip('[')
    texts = texts.strip(']')
    texts = texts.replace("'", "")
    texts = texts.replace(" ", "")
    texts = texts.split(',')

    if len(scores) != len(texts):
        print("scores texts长度不一致:", scores, texts)

    length = len(scores)
    temp = []
    for i in range(length):
        if scores[i] != '-10086':
            temp.append(i)

    # print("temp:", temp)

    scores = list(map(float, scores))
    result_scores = np.array(scores)[temp].tolist()
    result_texts = np.array(texts)[temp].tolist()

    if len(result_scores) != len(result_texts):
        print("长度不一致！！出错辣！！！")
        print(result_scores)
        print(result_texts)
    return result_scores, result_texts


# 计算短文本情感
senta = hub.Module(name='senta_cnn')
def calculate_sentiment(path):
    current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")
    print("正在计算情感分数。。。")
    current_data['score'] = current_data.apply(lambda row: calculate_sentiment_by_row(row['shortTexts']), axis=1)
    s_path = "test/6 raw data + score-v3.xlsx" if debug else "data/6 raw data + score-v3.xlsx"
    current_data.to_excel(s_path, index=False)


# 保留中文字符，删除非中文字符
def find_Chinese(doc):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    Chinese = re.sub(pattern, "", doc)
    # print(Chinese)
    return Chinese


digits = '0123456789'
# 文本预处理,shortTexts-list type
def text_processing(shortTexts, scores):
    result = []
    result_scores = []
    result_origin = []
    for current_scores in scores:
        current_scores = current_scores.strip('[')
        current_scores = current_scores.strip(']')
        current_scores = current_scores.replace(" ", "")
        current_scores = current_scores.split(',')
        # print("current_scores:", current_scores)
        # print("current_scores' type:", type(current_scores))
        result_scores.append(list(map(float, current_scores)))

    for texts in shortTexts:
        texts = texts.strip('[')
        texts = texts.strip(']')
        texts = texts.replace("'", "")
        texts = texts.replace(" ", "")
        texts = texts.split(',')
        # print("texts:", texts)
        result_origin.append(texts)

        current_result = []
        for line in texts:
            origin_line = line
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
            if len(line.strip()) == 0:
                print("当前短文本预处理后为空：", origin_line)
            current_result.append(line)
        result.append(current_result)
    # print("result_origin:", result_origin)
    return result, result_scores, result_origin


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件+短文本提取
    print(">>>正在读取数据。。。")
    # path = "test/1 Raw data-test.xlsx" if debug else "data/1 Raw data.xlsx"
    # load_data(path)

    # 计算情感
    path = "test/6 raw data + shortTexts-v3.xlsx" if debug else "data/6 raw data + shortTexts-v3.xlsx"
    calculate_sentiment(path)

    # 移除没有情感分数的评论
    path = "test/6 raw data + score-v3.xlsx" if debug else "data/6 raw data + score-v3.xlsx"
    remove_texts_scores(path)

    sys.exit(10086)
    # 短文本-属性匹配
    path = "test/Beijing & Shanghai + score-v3.xlsx" if debug else "data/Beijing & Shanghai + score-v3.xlsx"
    current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")
    texts_result, scores_result = shortText_attribute_match(current_data["shortTexts"].tolist(), current_data["score"].tolist())
    # 将以上结果进行保存
    for key, value in texts_result.items():
        current_data[key] = pd.Series(value)
        current_data[(key + "_label")] = pd.Series(scores_result.get(key))

    s_path = "test/aspect_sentiment_result-v3.xlsx" if debug else "result/aspect_sentiment_result-v3.xlsx"
    current_data.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")



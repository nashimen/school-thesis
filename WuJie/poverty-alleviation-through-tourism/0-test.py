import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.metrics.pairwise import cosine_similarity
import jieba.posseg as pseg


def similarity_test():
    a = [1, 0, 2, 1]
    b = [1, 0, 1, 1]

    sim = cosine_similarity([a], [b])

    print("sim = ", sim[0][0])


def jieba_test():
    sentence = "必须五星好评！酒店的装修真的非常好看，无论是外观还是内部装饰，都是绝绝子。服务非常体贴，我们本来的房间有一点异味，酒店主动帮我们换了房间。还提醒我们附近施工可能有点吵，给我们送了耳塞(虽然并没有被吵到)。行李寄存的时候还有小牌牌。 ◎夜宵也超棒，小混沌口味一流。而且而且厨师人也非常nice，我们从迪士尼回来已经十一点了，他还答应帮我们做了夜宵。◎打扫房间的阿姨工作也超认真负责，当初就是她主动提出房间有异味，帮我们换了房。房间卫生特别好，还有很多小细节都做得很好。 ◎最最关键是交通，酒店离地铁站特别近，去哪儿都很方便的！"

    current_result = []
    words = pseg.cut(sentence)
    for word, flag in words:
        print(word, flag)
        # if str(flag) is 'a' or str(flag) is 'd':


def find():
    sentence = "因为是淡季所以价格比平时便宜很多"
    # word = "淡季"
    word = "不便宜"
    if word in sentence:
        print("存在")
    else:
        print("不存在")


def list_slice():
    arr1 = [1, 3, 111, 1112, 15123, 10098, 20012, 2005]
    arr1 = np.array(arr1)
    temp = [0, 2, 4]
    print(arr1[temp].tolist())


import gensim.models
# 读词向量CBOW200
def gensim_test():
    model = gensim.models.Word2Vec.load('gensim-model-43gruuh9-CBOW200-mincount100')
    word_vector = model.wv
    def gensim_test_innter():
        sent = "中国"
        print(word_vector[sent])


def dict_test():
    synonyms = {"房间": ["屋子", "房型", "房子", "房間", "客房", "套房"],
                "大床": ["床品", "床垫", "双床"],
                "机器": ["机器人", "小机器人"]
                }
    local_words = ["房间"]
    if "房间" in synonyms.keys():
        local_words.extend(synonyms.get("房间"))
    print(local_words)


debug = False
debugLength = 50
def contain_test():
    # 读取sentence文件
    sentences_path = "data/6 aspect_sentences.xlsx"
    sentences = pd.read_excel(sentences_path, engine="openpyxl", nrows=debugLength if debug else None)
    # sentence_set = sentences[sentences["Facility"].str.contains("设施")]  # 获取包含当前kansei word的所有句子
    sentence_set = sentences[sentences["Facility"].str.contains("有效", na=False)]  # 获取包含当前kansei word的所有句子
    print(sentence_set)
    print(len(sentence_set["Facility"]))

from LAC import LAC
# 初始化分词模型
lac = LAC(mode='seg')
def lac_test():
    text = "最惊喜的是陶渊明故居"
    word_list = lac.run(text)
    print(word_list)

origin_dictionary = {
    "Food": "舌尖，形态，早饭，鸡汤，气味，地道，午餐，面条，特色菜，野菜，咖啡店，咖啡馆，早餐，烧饼，原汁，餐馆，茶叶，对联，垃圾，徽菜，美食，牛肠，饭菜，口味，糖果，美味，毛豆，早点，小吃，酒吧，饭店，臭鳜鱼，餐饮，风味，餐厅，饮食，笋干，咖啡，饭馆，小酒馆，水果，竹笋，臭桂鱼，腊肉，夜市，晚饭，食堂，小贩，牛肚，小吃店，",
    "Hospitality": "疫情，客栈，老板，房间，味道，民宿，酒店，空调，客房，管家，窗户，实名制，院子，墙面，实名，阳台，前台，房东，店主，卫生间，旅馆，装饰，厕所，宾馆，休闲游，小车，模式，客运站，旅途，检票员，回程，车位，素质，温度，师傅，汽车，指示牌，自助机，电话，散客，格局，当地居民，司机，农家，速度，垃圾桶，班车，老板娘，店家，旺季，店铺，网络，人气，小店，大气，人群，旅游团，态度，商店，小姐姐，家家户户，停车场，客人，行李，很多人，人们，公交，服务态度，专业，自驾游，售票处，人工，索道，人员，交通，设施，大巴，导游，讲解员，车程，旅行团，高铁，村民，线路，路面，年轻人，邮局，高铁站，汽车站，公交车，平台，客运，三轮车，服务员，机器，工作人员，旅游车，村民们，服务，卫生，管理，客栈，互动，互动性，停车，安排，",
    "Culture": "年间，底蕴，艺术，砖雕，巷子，小巷，灯光，夜景，家族，人文，故事，风格，代表，祠堂，布局，典型，黛瓦，季节，马头墙，白天，人家，古树，技艺，徽娘，宝库，地标，地位，象征，履福堂，烟火，纪念品，窄巷，标志性，民间，遗产，之三华里处，细节，小村庄，体验感，木结构，砖墙，世纪，古城，宗族，胡姓，厅堂，老街，规模，天井，门前，古巷，马头，古建，角落，工艺品，山村，米酒，石板，精神，人生，人间，声音，家训，习惯，本地，古迹，诗意，意外，灯笼，住宅，典范，沧桑，气势，建筑物，趣味性，商业街，慢生活，教育，石板路，可玩性，农村，基地，风土人情，美院，知识，敬修堂，爱国主义，油墨，高墙，青石，水幕，美誉，古色，经典，氛围，居民，民风，粉墙，品味，印象，建筑群，雕刻，街巷，建筑风格，牌楼，徽式，小桥，徽商，中国画，书院，承志堂，传统，徽派，古建筑，村落，文化，民居，文化遗产，徽派建筑，石雕，木雕，古村，历史，建筑，古村落，村子，世界，牌坊，气息，商业，村口，月沼，特色，白墙，乡村，古民居，学生，老宅，公元，内涵，结构，民俗，古人，老房子，名录，房屋，时光，徽派民居，重点，文物，小巷子，代表性，红灯笼，气氛，灰瓦，韵味，村庄，古镇，生活，佳作，后人，全貌，博物馆，年代感，敬爱堂，飞檐，徽派建筑风格，时代，财富，木质，气派，雕花，子孙，后代，宝地，举家，家风，千秋，小院，成就，民宅，屋檐，标志，光影秀，古风，家乡，门楼，商业味，青瓦，手艺，三畏堂，灵气，历史感，风韵，名宿，园林，雕饰，德义堂，有人文，老宅子，有意境，院落，文字，人物，古黟桃花源，古韵，故居，时节，祖先，牛角，大理石，生活习惯，心灵，年代，情怀，名声，文艺，遗产地，字体，官家，非遗，民族，白瓦，传说，古民居群，阁楼，大院，前世今生，微派，印记，官宅，老屋，遗迹，习俗，三雕，商铺，黑瓦，风情，古建筑群，工艺，商家，寓意，原住民，大户，岁月，小村落，古宅，特产，人文景观，青砖，乐叙堂，灯光秀，电影，大牌坊，人文景点，人间烟火，明珠，大宅子，木刻，大宅，精品，宅院，音乐，节目，商业化，故居，陶渊明故居，名胜古迹，名胜，村落，名人，宅子，对联，古宅，",
    "Nature": "溪水，观景台，好地方，城市，小桥流水，水乡，原味，自然景观，风水，色彩，风貌，画卷，景观，名气，精华，牛形，山色，大树，画里，塔川，秋天，景区，地方，环境，风景，美景，水墨画，自然，照片，夜晚，水墨，山坡，湖景，月亮，小河，背景，意境，晴天，胜地，大自然，天气，空气，山水，油菜花，流水，倒影，村头，好去处，雨天，荷塘，月份，一景，世外桃源，错峰，小雨，景象，蓝天白云，云雾，水墨丹青，外观，教育基地，山林，秋色，太阳，美如画，浓墨重彩，取景地，水渠，清泉，位置，喷泉，风光，山水画，水系，池塘，大片，观景亭，余晖，视觉，美术，天下，落日，镜头，绿水，河流，阴雨，墙面，桃花源，山路，河水，小山，群山，观景，雨声，高峰，地区，风景区，地理，水流，地势，地点，气候，水池，水质，晨雾，大早，场景，细雨，野趣，水景，灯火，江南古镇，冬季，薄雾，故宫，景色，情怀，面貌，油画，天堂，花园，环境保护，水塘，油菜，晚霞，湿地，向日葵，水沟，花海，风景线，大山，竹海，暴雨，天空，石桥，半月形，水利，诗情画意，垂柳，秋季，夕照，月湖，晨曦，秋景，夜色，青山绿水，日落，街道，仙境，魅力，青石板，荷花，湖面，田园，原生态，夕阳，溪流，色调，空间，小溪，全景，炊烟，小村，画面，青山，美感，蓝天，白云，烟雨，星空，景致，日出，南屏，阳光，湖水，湖光，春晓，环山，景色，景点，乐趣，水面，景点，看点，南山，",
    "Price": "停车费，成人，小朋友，经济，生意，费用，商品，票机，车票，性价比，套票，票价，身份证，联票，物价，节假日，半价，一个人，价值，价格，门票，便宜，天地，价钱，折扣，定价，联票，超值，免费，购票，携程，订票，实惠，"
}


def initDictionary():
    dictionary = {}
    # 去重+统计个数
    count = 0
    for attribute, words in origin_dictionary.items():
        words = words.split("，")
        # print(attribute, "原始长度为", len(words))
        words = list(set(words))
        print(attribute, "长度为:", len(words))
        count += len(words)
        dictionary[attribute] = words
    # 统计个数
    print("词典总长度为", count)
    return dictionary

dictionary = initDictionary()
print(dictionary)

def dict_test():
    word = "陶渊明故居"
    hospitality = dictionary.get("Culture")
    print(hospitality)
    print(word in hospitality)


# 观点提取函数
# 输入sentences，返回两个list，一个特征词list，一个观点词list
def extract_opinion(sentences, interface):
    features = []
    opinions = []
    for sentence in sentences:
        elements = interface(sentence)[0]
        if len(elements) == 0:
            print("sentence:", sentence)
            continue
        elements = elements["评价维度"]
        for element in elements:
            # print("element:", element)
            feature = element["text"]
            opinion = element['relations']
            # 没有观点词
            if len(opinion) == 0 or "观点词" not in opinion.keys():
                print("element:", element)
                continue
            # print("opinion:", opinion)
            opinion = opinion["观点词"][0]["text"]
            if len(feature) > 0 and len(opinion) > 0:
                features.append(feature)
                opinions.append(opinion)

    return features, opinions

from paddlenlp import Taskflow
schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']} # Define the schema for opinion extraction
ie = Taskflow("information_extraction", schema=schema)

def opinion_extraction_test():
    sentence = ["宏村的风景很漂亮", "但村民很冷漠"]
    print(extract_opinion(sentence, ie))


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    opinion_extraction_test()

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")
    sys.exit(10001)


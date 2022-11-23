import time, numpy as np, pandas as pd, jieba, re, sys
import gensim.models

# 读词向量CBOW200
model = gensim.models.Word2Vec.load('model/gensim_cbow200_5.model')
word_vector = model.wv

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 50

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


# 读取主题词
topic = pd.read_excel('data/9 Attribute words.xlsx', engine='openpyxl')
topic['Phrases'] = topic['Phrases'].apply(lambda x: x.split('，'))
all_word = topic['Phrases'].tolist()
# 构造词典：{word->topic}
word_dict = {}

for i, topic_words in enumerate(all_word):
    for topic_word in topic_words:
        word_dict[topic_word] = topic['Feature'][i]

topic_word_df = pd.DataFrame()
topic_word_df['word'] = word_dict.keys()
topic_word_df['topic'] = word_dict.values()

# 加载主题词（属性词）的词向量
'''
embeddings = []

word_number = len(topic_word_df['word'])
words = topic_word_df['word'].tolist()
for i in range(0, word_number):
    print("current word is: ", words[i])
    embeddings.append(word_vector[words[i]])
'''

topic_word_df['embedding'] = topic_word_df['word'].apply(lambda x: word_vector[x])

# print(topic_word_df['word'])
# print(topic_word_df['topic'])
# sys.exit(61)


# 读取文件，文件读取函数
def read_file(filename):
    # with open(filename, 'rb')as f:
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        # 返回list类型数据
        text = text.split('\n')
    return text


def getStopList():
    stoplist = pd.read_csv(filepath_or_buffer='stopwords.txt').values
    return stoplist

# 读取停用词表
stoplist = getStopList()
print('origin stop length: ' + str(len(stoplist)))# 加载停用词

from LAC import LAC
# 初始化分词模型
lac = LAC(mode='seg')

# 处理文本：分词，去停用词
def text_process(text):
    word_list = lac.run(text)
    word_list = list(filter(lambda x: x not in stoplist, word_list))
    # word_list = list(filter(lambda x: x in word_vector, word_list))
    return word_list


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
            if debug:
                print("before:", line)
            origin_line = line
            # 去标点符号
            line = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(line))
            # 去掉数字
            line = line.translate(str.maketrans('', '', digits))
            # 去掉非中文字符
            line = find_Chinese(line)
            if debug:
                print("after 1:", line)
            # 分词
            line = text_process(line)
            # line = jieba.cut(line)
            # 去停用词
            # line = [word.strip() for word in line if word not in stoplist]
            line = ' '.join(line)
            if len(line.strip()) == 0:
                print("当前短文本预处理后为空：", origin_line)
            current_result.append(line)
            if debug:
                print("after 2:", line)
                print("*" * 50)
        result.append(current_result)
    # print("result_origin:", result_origin)
    return result, result_scores, result_origin


dictionary = initDictionary()
# 短文本-属性匹配
def shortText_attribute_match(shortTexts, scores):
    # 对shortTexts进行预处理：去标点符号→分词→去停用词
    shortTexts, scores, origin_shortTexts = text_processing(shortTexts, scores)
    if len(shortTexts) != len(scores):
        print("shortTexts和scores长度不一致。。。")
        sys.exit(-2)
    length = len(shortTexts)
    # 存在每条评论的短文本（属性匹配后），所有value的长度都一样。如果同一属性下有多条，则合并
    texts_dictionary = {"Food": [], "Hospitality": [], "Nature": [], "Culture": [], "Price": []}
    # 存在每条评论短文本（属性匹配后）对应的正向情感分，所有value的长度都一样。如果同一属性下有多条，则取平均值
    scores_dictionary = {"Food": [], "Hospitality": [], "Nature": [], "Culture": [], "Price": []}  # 未提及-1
    # 依次遍历每条评论&及其短文本
    for i in range(length):
        texts_dictionary_temp = {"Food": [], "Hospitality": [], "Nature": [], "Culture": [], "Price": []}  # 当前评论的变量
        scores_dictionary_temp = {"Food": [], "Hospitality": [], "Nature": [], "Culture": [], "Price": []}  # 当前评论的
        current_texts_origin = origin_shortTexts[i]  # 原始的评论文本
        current_texts = shortTexts[i]
        current_scores = scores[i]
        current_length = len(current_texts)
        if current_length != len(current_scores):
            # print("current_texts:", current_texts)
            # print("current_scores:", current_scores)

            print("短文本条数和scores个数不一致。。。")
            sys.exit(-3)
        for j in range(current_length):
            text_origin = current_texts_origin[j]  # 原始的短文本
            text = current_texts[j]
            # 如果短文本预处理完之后为空，则continue
            if len(text.strip()) == 0:
                continue
            score = current_scores[j]
            topic = judgeTopic(text)
            # print("1", text, topic)

            # 如果根据先验词典未匹配到属性，则计算相似度，选择相似度最大的属性作为当前属性
            if topic == "EMPTY":
                topic = judgeTopicByEmbedding(text)
            if topic == "EMPTY":
                print("最终仍未匹配到属性：", text_origin, ", ", text)
                continue

            texts_dictionary_temp[topic].append(text_origin)
            scores_dictionary_temp[topic].append(score)

        # 每处理完一条评论（可能包含多个短文本），进行以此合并和平均汇总，存放的应该是原始的评论文本（短文本）
        for key, value in scores_dictionary_temp.items():
            value_texts = ",".join(texts_dictionary_temp.get(key))
            # 如果value有多个，则求平均;同时合并短文本
            if len(value) > 0:
                mean = np.mean(value)
            else:
                mean = 1992  # 如果不存在当前属性，则标记为-1
            texts_dictionary[key].append(value_texts)
            scores_dictionary[key].append(mean)

    # 判断长度是否一致
    lengths = []
    for key, value in texts_dictionary.items():
        # print(key, "'s length = ", len(value))
        lengths.append(len(value))
    for key, value in scores_dictionary.items():
        # print(key, "'s length = ", len(value))
        lengths.append(len(value))
    if len(set(lengths)) > 1:
        print("匹配结果有问题，长度不一致")
        sys.exit(-4)
    else:
        print("匹配结束。。。")
    return texts_dictionary, scores_dictionary


# 判断短文本属于哪个主题，根据dictionary
def judgeTopic(text):
    words = text.split(' ')
    topic_result = ''
    for word in words:
        # 遍历先验词库
        topic_result = "Nature" if word in dictionary.get("Nature") else "Culture" if word in dictionary.get("Culture") \
            else "Hospitality" if word in dictionary.get("Hospitality") else "Price" if word in dictionary.get("Price") \
            else "Food" if word in dictionary.get("Food") else "EMPTY"
        if topic_result != "EMPTY":
            break

    return topic_result


# 生成句向量
def sentence_vector(words):
    word_num = len(words)
    count = 0
    vector_sum = 0
    for word in words:
        if word in word_vector.index_to_key:
            count += 1
            vector_sum += word_vector[word]
    if count == 0:
        return 0
    return vector_sum / count


from sklearn.metrics.pairwise import cosine_similarity
topic_word_embedding = topic_word_df['embedding'].tolist()
# 根据相似度匹配属性，使用词向量
def judgeTopicByEmbedding(text):
    words = text.split(' ')
    sentence_embedding = sentence_vector(words)
    # print("length:", len(sentence_embedding))
    # print("type:", type(sentence_embedding))
    # print(sentence_embedding)
    if type(sentence_embedding) is int:
        return "EMPTY"
    sentence_embedding = sentence_embedding.reshape(1, -1)
    cos_sim = cosine_similarity(sentence_embedding, topic_word_embedding)
    max_index = np.argmax(cos_sim)
    topic = topic_word_df.iloc[max_index]['topic']
    # print(text, topic)

    return topic


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件+短文本提取
    print(">>>正在读取数据。。。")

    # 短文本-属性匹配
    path = "test/8 raw data + score-v4.xlsx" if debug else "data/8 raw data + score-v4.xlsx"
    current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")
    texts_result, scores_result = shortText_attribute_match(current_data["shortTexts-fix"].tolist(), current_data["score-fix"].tolist())
    # 将以上结果进行保存
    for key, value in texts_result.items():
        current_data[key] = pd.Series(value)
        current_data[(key + "_label")] = pd.Series(scores_result.get(key))

    s_path = "test/9 aspect_sentiments-test.xlsx" if debug else "result/9 aspect_sentiments.xlsx"
    current_data.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")
    sys.exit(10001)



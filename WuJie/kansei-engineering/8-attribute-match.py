import time, numpy as np, pandas as pd, jieba, re, sys
import gensim.models

# 读词向量CBOW200
model = gensim.models.Word2Vec.load('gensim-model-43gruuh9-CBOW200-mincount100')
word_vector = model.wv

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 5

origin_dictionary = {
    "Facility": "区域，健身房，恒温，水温，温水，池子，休息区，游泳池，泳池，自带，露天，篮球场，运动，打篮球，更衣室，水质，游泳馆，温泉池，温泉，泡温泉，泡池，公共，spa，私汤，汤池，半山，設施，窗帘，电视，灯光，开关，网速，电影，机器人，投屏，系统，智能，网络，wifi，信号，科技感，电视机，马桶盖，高科技，小机器人，空气净化器，新风，模式，音乐，机器，频道，家居，无线，蓝牙，音箱，音响，小度，语音，插卡，加湿器，新风机，智能化，感应，有意思，摆设，基础设施，设备，家具，空调，中央空调，暖气，冷气，设施，柜子",
    "Parking": "停车场，停车费，车位，箱子，出租车，停车位，车辆，通道，车子，停车券，车库，地下",
    "Service": "阿姨，小姐姐，工作人员，服务员，店员，人员，帅哥，美女，小哥，小哥哥，姑娘，管家，员工，主管，小姑娘，姐姐，先生，经理，大姐，女士，男生，小伙，接待员，小妹妹，女孩子，接待人员，小伙子，女生，妹妹，同学，男士，小姐，眼镜，名字，客房部，领班，女孩，小路，工号，宾客，客房服务员，海霞，销售部，戴眼镜，主任，大使，需求，住客，商家，电话，客人，店家，服务台，流程，老板，团队，手续，发票，顾客，信息，客户，身份证，台办，速度，房卡，规定，柜台，同行，接电话，事项，现金，角度，什么时候，押金，喜好，说明，部门，总机，信用卡，回复，微信，通知，预订部，没有人，折腾，接送机，送机，接机，行李，路线，师傅，大叔，保安，快递，安保，服务生，司机，礼宾，门童，礼宾部，大爷，行李员，大哥，车上，前台，态度，服务态度，素质，印象，客服，服务，颜值，专业，效率，语气，前厅，耐心，亲和力，规范，笑容，热情，负责任，素养，笑脸，面带，细心，客气，始终，礼貌，態度，人員，服務，商务，出差",
    "Special_care": "假期，亲子，疫情，亲子游，聚会，婴儿，太太，老人，小朋友，宝宝，孩子，朋友们，住房，女儿，儿子，小孩，小孩子，小伙伴，父母，孩子们，老婆，妈妈，小宝宝，宝贝，爸妈，爸爸，过生日，儿童，帐篷，滑梯，玩具，大人，乐园，游乐场，游乐园，专属",
    "Room": "走廊，房子，房门，房间，窗户，过道，客房，楼道，房間，一人，双人，一个人，大床，公寓，窗房，内窗，房型，楼层，两人，两个人，大床房，早餐券，标间，标准间，双人床，朝向，榻榻米，特价，行政，双床，家庭房，酒廊，单人，商务房，三个人，套房，行政房，门票，平米，套间，高楼层，客厅，主楼，一厅，贵宾，loft，床品，浴巾，毛巾，床垫，枕头，被子，床铺，床单，被套，被褥，软硬，餐具，耳塞，衣服，电源，插座，手机，遥控器，床头，水壶，吹风机，充电器，电脑，床头柜，插头，usb，床边，台灯，穿着，一边，声音，私密性，睡眠，安全感，大声，蚊子，屋子，门锁，睡觉，面积，空间，冷风，椅子",
    "Value": "性價比，价格，性价比，舒适度，经济，性价，房价，便宜，超值，公道，经济实惠，实惠，偏高，合适，酒店，五星，宾馆，质量，四星，价位，全季，招待所，品质，经济型，档次，水准，价钱，水平，毛病，级别，星级，四星级，星级酒店，五星级酒店，五星级，五星酒店，三星，定位，预期，硬件，软硬件，老酒，软件，精品酒店，老酒店，年头，高端，老牌，老五星，体验感，品牌，城市，旗下，桔子，牌子，形象，民宿，全国，服务业，美豪，水晶，大酒店，系列，橘子，美豪酒店，假日，亚朵，国企，帝都，华尔道夫，皇冠，万豪，秋果，五分，喜来登，锦江，总结，京城，吐槽，威斯汀",
    "Booking": "会员，积分，图片，差价，评论，订单，费用，照片，携程，房费，平台，入住率，艺龙，用户，评分，网站，备注，订房，果断，不可",
    "Cleanliness": "异味，烟味，季节，味儿，隔壁，卫生，臭味，空气，天气，温度，光线，霉味，气味，感冒，透气，通风，头发，垃圾，状况，污渍，下水道，地面，水池，蟑螂，垃圾桶，镜子，洗手台，地板，灰尘，厨房，地漏，地毯，墙壁，衣柜，桌子，沙发，角落，颜色",
    "Bathroom": "卫浴，花洒，卫生间，浴室，马桶，厕所，玻璃，干湿，淋浴间，热水，喷头，水流，淋浴房，卧室，洗手间，洗手池，洗澡间，水压，水龙头，浴缸，泡澡，淋雨，出水，用品，备品，纸巾，拖鞋，杯子，牙刷，梳子，牙具，沐浴露，洗发水，牙膏，浴袍，洗衣液，一次性，酒精，洗手液，护发素，用具，环保",
    "Location": "地方，广场，商场，便利店，周边，美食，大商场，超市，购物中心，商城，餐馆，小吃店，水果店，饭店，商店，一条街，楼下，世纪，美食街，电影院，万达广场，万达茂，万达，吃喝玩乐，新世界，附近，家乐福，交通，位置，大悦城，环境，商圈，散步，环境卫生，商业，地点，商业区，位子，汽车，学校，博物馆，公园，商业街，巷子，大使馆，大学，热闹，大剧院，体育馆，江边，胡同，使馆区，鸟巢，东方明珠，ktv，体育场，地铁站，地铁口，地理，外滩，公交车，路口，步行街，签证，地铁，公交站，路程，景点，人民广场，公交，距离，出口，单车，市区，自行车，地址，范围，车程，景区，医院，地铁线，线路，二号线，火车站，车站，科技馆，世纪大道，使馆，演唱会，动物园，高速，植物园，前门，升国旗，后海，长城，北京南站，三元桥，十里河，天坛，右转，西客站，溜达，大栅栏，天坛公园，升旗，天文馆，故宫，新天地，公里，火车，机场，高铁，高铁站，专车，班车，迪士尼乐园，班机，巴士，大巴，航站楼，野生动物园，坐车，滴滴，南站，首都，地段，中心，市中心，地带，核心，会展，地区，郊区，二环，国贸，三环，长安街，地处，cbd，公司，单位",
    "Surrounding": "夜景，景观，视野，大楼，风景，景色，飘窗，阳台，花园，阳光，落地窗，院子，庭院，顶楼，美景，露台，窗外，江景，江景房，小院，夜晚，别墅，裤衩，观景，環境，工地，马路，小区，高架，园区，白天，路边，全景，好找，修路，噪音，吓人",
    "Decoration": "装饰，大堂，风格，设计感，大厅，历史，时尚，氛围，有感觉，艺术，酒店大堂，建筑，气氛，文艺，理念，设计师，格调，主题，元素，大气，气息，香味，品味，有情调，红酒店，文化，低调，气派，自然，进门，古典，四合院，新颖，有特色，别致，现代，富丽堂皇，布局，外观，麻雀，格局，结构，新装修，装潢，陈设，房屋，年代感，内部，传统，老式，配置，设计",
    "Food": "外卖，宵夜，夜宵，水果，饮品，冰箱，可乐，牛奶，口罩，饮料，咖啡，下午茶，饼干，茶水，咖啡机，暖心，糕点，礼物，洗衣房，吧台，小食，酸奶，苹果，矿泉水，热茶，果汁，洗衣机，礼品，甜点，点心，小蛋糕，零食，面膜，烘干机，小点心，果盘，饮用水，糖果，雨伞，啤酒，甜品，蛋糕，暖胃粥，minibar，畅饮，限量，按摩椅，微波炉，一杯，迷你，早餐，口味，厨师，味道，中餐，自助餐，馄饨，餐厅，小馄饨，早饭，晚餐，油条，鸡蛋，豆浆，菜品，样式，口感，美味，西餐，品类，中西式，形式，午餐，面条，营养，餐食，食品，座位，火锅，煎蛋，面包，包子，食材，饭菜，食物，中西餐，早点，晚饭，外国人，中餐厅，老外，花样，套餐，海鲜，餐品，西餐厅，中西，煎饼，可口，点餐，自助，餐饮，小吃，饮食，庆丰包子，酒吧，特色，日料，地道，烧烤，咖啡厅，人气，网红，全聚德，海底捞，老字号，烤鸭，餐饮部"
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
topic = pd.read_excel('data/4 clusters_cbow200_kmeans-words fixed-Wu-20221010.xlsx', engine='openpyxl')
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


# 读取停用词表
stop_words = read_file(r"../stopwords.txt")
print('origin stop length: ' + str(len(stop_words)))

# 去掉停用词中的情感词
# 情感词与停用词有重合导致一些文本分数为0
stop_df = pd.DataFrame(stop_words)
senti_df = pd.read_excel('data/5 Kansei word sentiment lexicon-20221002.xlsx', engine="openpyxl")
stop_df.columns = ['word']
duplicated = pd.merge(stop_df, senti_df, on='word')['word'].tolist()
stoplist = list(filter(lambda x: x not in duplicated, stop_words))
print('remove sentiment stop length: ' + str(len(stoplist)))


from LAC import LAC
# 初始化分词模型
lac = LAC(mode='seg')

# 处理文本：分词，去停用词
def text_process(text):
    word_list = lac.run(text)
    word_list = list(filter(lambda x: x not in stoplist, word_list))
    word_list = list(filter(lambda x: x in word_vector, word_list))
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
            origin_line = line
            # 去标点符号
            line = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(line))
            # 去掉数字
            line = line.translate(str.maketrans('', '', digits))
            # 去掉非中文字符
            line = find_Chinese(line)
            # 分词
            line = text_process(line)
            # line = jieba.cut(line)
            # 去停用词
            # line = [word.strip() for word in line if word not in stoplist]
            line = ' '.join(line)
            if len(line.strip()) == 0:
                print("当前短文本预处理后为空：", origin_line)
            current_result.append(line)
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
    texts_dictionary = {"Facility": [], "Parking": [], "Service": [], "Special_care": [], "Room": [], "Value": [], "Booking": [], "Cleanliness": [], "Bathroom": [], "Location": [], "Surrounding": [], "Decoration": [], "Food": []}
    # 存在每条评论短文本（属性匹配后）对应的正向情感分，所有value的长度都一样。如果同一属性下有多条，则取平均值
    scores_dictionary = {"Facility": [], "Parking": [], "Service": [], "Special_care": [], "Room": [], "Value": [], "Booking": [], "Cleanliness": [], "Bathroom": [], "Location": [], "Surrounding": [], "Decoration": [], "Food": []}  # 未提及-1
    # 依次遍历每条评论&及其短文本
    for i in range(length):
        texts_dictionary_temp = {"Facility": [], "Parking": [], "Service": [], "Special_care": [], "Room": [], "Value": [], "Booking": [], "Cleanliness": [], "Bathroom": [], "Location": [], "Surrounding": [], "Decoration": [], "Food": []}  # 当前评论的变量
        scores_dictionary_temp = {"Facility": [], "Parking": [], "Service": [], "Special_care": [], "Room": [], "Value": [], "Booking": [], "Cleanliness": [], "Bathroom": [], "Location": [], "Surrounding": [], "Decoration": [], "Food": []}  # 当前评论的
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
        topic_result = "Facility" if word in dictionary.get("Facility") else "Parking" if word in dictionary.get("Parking") \
            else "Service" if word in dictionary.get("Service") else "Special_care" if word in dictionary.get("Special_care") \
            else "Room" if word in dictionary.get("Room") else "Value" if word in dictionary.get("Value") \
            else "Booking" if word in dictionary.get("Booking") else "Cleanliness" if word in dictionary.get("Cleanliness") \
            else "Bathroom" if word in dictionary.get("Bathroom") else "Location" if word in dictionary.get("Location") \
            else "Surrounding" if word in dictionary.get("Surrounding") else "Decoration" if word in dictionary.get("Decoration") \
            else "Food" if word in dictionary.get("Food") else "EMPTY"

    return topic_result


import xiangshi as xs
# 根据相似度匹配属性
def judgeTopicBySimilarity(text):
    # 计算text与字典中所有key的相似度，取最大值为最终结果
    max_similarity = 0
    topic = "EMPTY"
    for key, value in dictionary.items():
        sim = 0
        for v in value:
            # print("text:", text, ", v:", v)
            sim = max(sim, xs.cossim([text, v]))  # 找到当前key下最大的相似度
        if max_similarity < sim:
            max_similarity = sim
            topic = key

    return topic


# 生成句向量
def sentence_vector(words):
    word_num = len(words)
    vector_sum = 0
    for word in words:
        vector_sum += word_vector[word]
    return vector_sum / word_num


from sklearn.metrics.pairwise import cosine_similarity
topic_word_embedding = topic_word_df['embedding'].tolist()
# 根据相似度匹配属性，使用词向量
def judgeTopicByEmbedding(text):
    words = text.split(' ')
    sentence_embedding = sentence_vector(words)
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
    path = "test/Beijing & Shanghai + score-v4.xlsx" if debug else "data/Beijing & Shanghai + score-v4.xlsx"
    current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")
    texts_result, scores_result = shortText_attribute_match(current_data["shortTexts-fix"].tolist(), current_data["score-fix"].tolist())
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
    sys.exit(10001)



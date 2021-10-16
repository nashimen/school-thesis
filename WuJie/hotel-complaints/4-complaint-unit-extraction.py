# -*- coding: utf-8 -*-

import pandas as pd, numpy as np, jieba.posseg as pseg, sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# 显示所有列
pd.set_option('display.max_columns', None)

debug = False
debug_length = 100

# 先验词库
room = '隔音 太冷 噪音 异响 隔音性 耳塞 咳嗽声 咳嗽 嗓音 吵闹 临街 动静 噪声 杂音 车声 猫猫 猫 撸猫 声音 闹腾 房间 拖鞋 枕头 房型 房間 大小 标间 双人房 老房子 地方 双人间 房间内 标准间 套房 商务房 精品房 总统套房 ' \
       '四间房 旅社 旅舍 屋子 窗房 屋 隔间 标房 居室 民宅 豪庭 贵宾房 换衣间 褥子 衣柜'
room = room.split(' ')
environment = '霉味 卫生状况 烟味 景观灯 印渍 灰尘 风景 風景 排泄物 烟雾 墙角 潮虫 蜈蚣 条腿 有脏 二手烟 透光度 酸味 香烟 臭味 床太潮 除味剂 湿疹 皮肤过敏 过敏 空气净化 鼻涕 臭气 股子 一股子 一股 味儿 环境恶劣 ' \
              '潮味 甲醛 不脏 爬虫 脏 湿气 空气流通 积水 吸烟区 油烟味 有烟味 环境 卫生条件 '
environment = environment.split(' ')
value = '不值 性价比 乱收费 价格 贵 太值 服务费 价钱 费用 收费 停车费 价位 性价 分值 太坑 坑 大坑 坑人 不便宜 一分货 同价 价格优势 太贵 一分钱 价 一毛钱 不太值 很贵 快递费 昂贵 非常规 盈利 毛钱'
# value = '停车位 停车场 停车 车费 停车费 停车库 车库 停车处 停车点 卡停 停靠 停一边 停到 停在 停 不值 性价比 乱收费 价格 贵 太值 服务费 价钱 费用 收费 停车费 价位 性价 分值 太坑 坑 大坑 坑人 不便宜 一分货 同价 价格优势 太贵 一分钱 价 一毛钱 不太值 很贵 快递费 昂贵 非常规 盈利 毛钱'
value = value.split(' ')
facility = '被子 浴室 淋头 空调 毛巾 音响 音箱 网速 墙纸 桌子 加湿器 下水道 卫浴 花洒 汤池 马桶刷 泡池 溢水 洗澡水 梳子 镜子 洗漱间 水渍 洗手盆 浴袍 牙刷毛 牙刷 牙膏 洗面奶 洗发液 沐浴露 便池 浴缸 蓬头 排水口 ' \
           '洗发液 冷热水 浴液 洗手台 欧舒丹 浴帽 小便池 卫生洁具 化妆镜 电热器 置物架 浴房 洗澡时 盆浴 浴盆 干衣机 淋浴房 隔柜 排气 排气扇 肥皂 洗发水 牙具 洗发水 淋浴 浴帘 帘子 帘 卫生设施 沐浴露 热水 厕所 ' \
           '热水澡 卫生间 马桶 玻璃 洗手间 水盆 水龙头 停车位 停车场 停车 车费 停车费 停车库 车库 停车处 停车点 卡停 停靠 停一边 停到 停在 停 信号 网络 没信号 没网 Internet internet wifi WIFI Wifi wi-fi Wi-Fi WI-FI ' \
           '网 网页 网卡 卡顿 太卡 太卡了 卡 中央电视台 电视台 北京电视台 断网 无线网络 无线网 有线网 電視 网线 网太差 电视频道 频道 网络 網絡 电视信号 路由器 电信 手机信号 电视 温泉 婴儿床 铁床 卧椅 滑轨 地热 ' \
           '发电机 豪华设施 消防设施 座机 装饰 地摊 插头 火灾 卫生设备 暖气片 地板 服务设施 电梯 卷帘 洗衣房 清洁机 柜角 打印机 电子秤 壁灯 密码锁 电源插座 画框 墩布 顶灯 阳台门 座椅 酒店家具 投影仪 高尔夫 空调主机 ' \
           '办公椅 植物 电动 灯光 遥控 榻榻米 楼梯 方砖 遥控器 笤帚 床头 窗户纸 数据线 窗子 木地板 插座 洗衣机 充电器 暖风 冷风 沙发 电水壶 热水壶 电热水壶 水壶 衣架 电暖气 床垫 插排 床单 窗台 门窗 挡风玻璃 写字台 ' \
           '健身房 太旧 床 空调 床头柜 家具 飘窗 漏风 设施 门 酒店设施'
facility = facility.split(' ')
food = '鸡蛋 早餐券 食物 品种 咖啡机 套餐 口味 农夫山泉 热饮 咖啡 西餐厅 绿茶 汽水 炒菜 肉包 茶包 混沌 小吃 果汁 面条 家常菜 瓶装水 啤酒 微波炉 饮料机 休闲吧 烤面包机 面包机 盘子 冰箱 餐 菜量 胃口 茶叶 食材 ' \
       '猕猴桃 晚餐 菜品 餐饮 下午茶 蒸点 凯撒 面包干 芝麻 菜 吸油烟机 热菜 凉菜 早餐 海鲜 苹果 餐食 厨房 餐台 油条 欧式 面包 包子 电磁炉 美食 黄油 零食 午餐 饺子 饮品 番茄酱 菜单 杂粮 矿泉水 微波炉 便器 水果 ' \
       '西瓜 果盘 用餐 苹果汁 咖啡杯 牛奶 中餐厅 餐厅 葡萄 煎饼果子 自助餐厅 餐具'
food = food.split(' ')
service = '阿姨 退房 管理 服務員 管家 行政 保安 服务 经理 换房 前台 态度 态度恶劣 服务员 人员 服务态度 解决问题 极差 大堂 經理 经理 清洁员 中层干部 干部 服务区 房嫂 男士 吧台 老客户 办理 员工 客户 水平 投诉 清洁工 ' \
          '投诉无门 业务素质 小姐 服务体系 工作人员 小姐姐 小姑娘 女服务员 男服务员 素质 电话 专业 极差 客服 服务水平 联系 业务 办事 迎宾 小伙子 女士 美女 总台 职业 客房部 办手续 人手 热情 帅哥 押金 师傅 素养 ' \
          '客户经理 服务中心 服务行业 管理人员'
service = service.split(' ')
# location = '位置 方位 角落 地段 公园 宅急送 外景 中餐馆 滑雪场 环内 北影 夜景 西路 地图 地理位置 不好找 好找 难找 偏僻 偏 周围 附近 商业街 经典 茶馆 地鐵站 公交车 公交车站 车流 线站 车辆 公共交通 换乘 线路 主路 二号线 地铁 机场 地铁站 高铁 公交 路线 火车站 火车 地铁口 车站'
location = '方位 周边 周围 基站 庆丰 走路 偏僻 僻静 中医院 位置 周边 周边环境 肿瘤医院 方位 科技馆 体检中心 角落 地段 距离 东长安街 北门 工地 路边 马路边 工体 工人体育馆 对面 人工湖 阜外医院 商铺 公园 雍和宫 维景 ' \
           '維景 北二环 五环 马桥 东单 西单 团结湖 潘家园 北四环 北三环 商场 大兴区 妇产医院天安门城楼 亮马桥 宅急送 外景 北京宾馆 度假村 中餐馆 滑雪场 毛主席纪念堂 平安里 后花园 奥林匹克公园 大型超市 四环路 建国路 ' \
           '朝阳医院 娱乐区 国家电网 北京大学 清华大学 北大 清华 北京天文馆 百货店 使馆区 环内 北大第一医院 五环 小吊梨 小吊梨汤 东南西北 地坛公园 东安市场 南楼 美食城 北京动物园 西二环 首都医科大学 西三环 内环 ' \
           '北京医院 镇政府 北影 产业园 夜景 同仁医院 西路 昆仑饭店 西直门 医院 中国医学科学院肿瘤医院 故宫 虎坊桥 地图 地理位置 玉渊潭 卡地亚 三庆园 不好找 好找 难找 偏僻 偏 周围 附近 快餐店 麦当劳 肯德基 餐饮店 ' \
           '水果店 体育馆 羽毛球馆 商业街 经典 茶馆 地鐵站 公交车 公交车站 车流 四惠东 线站 车辆 公共交通 换乘 北京地铁 线路 主路 二号线 地铁 机场 地铁站 高铁 公交 路线 火车站 火车 地铁口 车站'
location = location.split(' ')
noise = '隔音 噪音 异响 隔音性 耳塞 咳嗽声 咳嗽 嗓音 吵闹 临街 动静 噪声 杂音 车声 猫猫 猫 撸猫 声音 闹腾'
noise = noise.split(' ')
bathroom = '下水道 卫浴 花洒 汤池 马桶刷 泡池 溢水 洗澡水 梳子 镜子 洗漱间 水渍 洗手盆 浴袍 牙刷毛 牙刷 牙膏 洗面奶 洗发液 沐浴露 便池 浴缸 蓬头 排水口 洗发液 冷热水 浴液 洗手台 欧舒丹 浴帽 小便池 卫生洁具 化妆镜 ' \
           '电热器 置物架 浴房 洗澡时 盆浴 浴盆 干衣机 淋浴房 隔柜 排气 排气扇 肥皂 洗发水 牙具 洗发水 淋浴 浴帘 帘子 帘 卫生设施 沐浴露 热水 厕所 热水澡 卫生间 马桶 玻璃 洗手间 水盆 水龙头'
bathroom = bathroom.split(' ')
parking = '停车位 停车场 停车 车费 停车费 停车库 车库 停车处 停车点 卡停 停靠 停一边 停到 停在 停'
parking = parking.split(' ')

stoplist = []
f = open('stopwords.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    stoplist.append(line.strip())
advlist = []
f = open('advWords.txt', 'r')
lines = f.readlines()
for line in lines:
    advlist.append(line.strip())
# 只去停用词的语料
corpus_nostopwords = []
# 原始语料
corpus_origin = []
# 数据预处理
def data_preprocess(texts):
    print("生成词袋。。。")
    corpus = []
    for current_texts in texts:
        # print("current_texts = ", current_texts)
        current_corpus = []
        current_corpus_nostopwords = []
        current_corpus_origin = []
        for text in current_texts:
            # print("text = ", text)
            # 词性标注过滤，保留名词、处所词、方位词、动词、代词
            words = pseg.cut(text)
            temp = ' '.join(w.word.strip() for w in words if (str(w.flag).startswith('n')))
            # print("temp = ", temp)
            temp = temp.split(' ')
            # 去停用词
            temp = ' '.join(word.strip() for word in temp if word not in stoplist and not word.isdigit() and word not in advlist)
            current_corpus.append(temp)
            # 只去停用词的语料
            words = pseg.cut(text)
            temp2 = ' '.join(w.word.strip() for w in words)
            # print("words = ", words)
            # print("temp2 = ", temp2)
            temp2 = temp2.split(' ')
            # print("temp3 = ", temp2)
            temp2 = ' '.join(word.strip() for word in temp2 if word not in stoplist and not word.isdigit() and word not in advlist)
            current_corpus_nostopwords.append(temp2)
            current_corpus_origin.append(text)
        corpus.append(current_corpus)
        corpus_nostopwords.append(current_corpus_nostopwords)
        corpus_origin.append(current_corpus_origin)
    return corpus


topic_number = 10
# 判断主题类型
facility_topics = [6]
environment_topics = [3]
value_topics = [0, 4]
service_topics = [7]
room_topics = [2, 9]
food_topics = [8, 1]
location_topics = [5]

parking_topics = [5]
bathroom_topics = []
noise_topics = [11]

# 保存方面文本摘要
abstract_texts = {'room': [], 'service': [], 'location': [], 'facility': [], 'value': [], 'environment': [], 'food': []}
def judgeTopic(texts_list):
    texts_topics_list = []
    length = len(texts_list)
    # for texts in texts_list:
    for i in range(length):
        texts = texts_list[i]
        current_texts_topics_list = []
        texts_nostopwords = corpus_nostopwords[i]
        texts_origin = corpus_origin[i]
        length_texts = len(texts)
        length_texts_nostopwords = len(texts_nostopwords)
        if length_texts != length_texts_nostopwords:
            print(length_texts, length_texts_nostopwords)
            print("出错辣！！！长度不一致！！！")
        # for text in texts:
        for j in range(length_texts):
            text = texts[j]
            text_nostopwords = texts_nostopwords[j]
            text_origin = texts_origin[j]
            # print("text_nostopwords = ", text_nostopwords)
            words = text.split(' ')
            prob_result = -1
            topic_result = ''
            for word in words:
                # 首先遍历先验词库
                topic_result = 'room' if word in room else 'service' if word in service else 'environment' if word in environment else 'value' if word in value \
                    else 'facility' if word in facility else 'location' if word in location \
                    else 'food' if word in food else ''
                if len(topic_result) > 0:
                    break
            if len(topic_result) == 0:
                for word in words:
                    # 依次遍历主题，判断属于哪个
                    probs = []
                    for i in range(topic_number):
                        topic = "Topic" + str(i)
                        probs.append(topicWordsProb_load.get(topic).get(str(word), -1.0))
                    max_index = probs.index(max(probs))
                    max_prob = probs[max_index]
                    if max_prob > prob_result:
                        prob_result = max_prob
                        topic_result = 'service' if max_index in service_topics else 'room' if max_index in room_topics else \
                            'environment' if max_index in environment_topics else 'value' if max_index in value_topics else 'facility' \
                                if max_index in facility_topics else'location' if max_index in location_topics \
                                else 'food' if max_index in food_topics else ''
            # if len(topic_result) > 0:
            #     print(text, topic_result)
            current_texts_topics_list.append(topic_result)
            if topic_result in abstract_texts.keys():
                abstract_texts[topic_result].append(text_origin)
        texts_topics_list.append(current_texts_topics_list)
    return texts_topics_list


def produceTopicScore(topics_list, scores_list):
    topic_score = {'room': [], 'service': [], 'location': [], 'facility': [], 'value': [], 'environment': [], 'food': []}
    length = len(topics_list)
    topics = ['room', 'service', 'location', 'facility', 'value', 'environment', 'food']
    for i in range(length):
        current_topics = topics_list[i]
        current_scores = scores_list[i]
        for topic in topics:
            if topic not in current_topics:
                topic_score[topic].append(None)
            else:
                indices = [i for i, x in enumerate(current_topics) if x == topic]
                if len(indices) == 1:  # 只有一个，则直接存入
                    topic_score[topic].append(current_scores[indices[0]])
                else:  # 有多个，则求平均值
                    # 之后测试：对于location取最小值
                    '''
                    if topic in ['location', 'parking']:
                        min_value = 0.7
                        for i in indices:
                            if current_scores[i] > min_value:
                                min_value = current_scores[i]
                        topic_score[topic].append(min_value)
                    else:
                    '''
                    s = 0
                    for i in indices:
                        s += current_scores[i]
                    topic_score[topic].append(s / len(indices))

    return topic_score


# 展示结果
def showResult(keyWords, topK, keys):
    # print("》》》打印每个topic的topK个关键词")
    length = len(keyWords)
    print("length = ", length)
    for i in range(length):
        if i == 0:
            print(keys[i], ':', keyWords[i][-topK:])


# 从wordTfidf中检索TopK词语的tfidf值
def searchTfidf(keyWords, topK, keys):
    print("searchTfidf keys = ", keys)
    topKWordsTfidf = {'room': {}, 'service': {}, 'location': {}, 'facility': {}, 'value': {}, 'environment': {}, 'food': {}}
    length = len(keyWords)
    print("length = ", length)
    for i in range(length):  # 依次遍历七个主题
        # print("i = ", i)
        # print(keys[i])
        topKWords = keyWords[i][-topK:]
        for word in topKWords:
            topKWordsTfidf[keys[i]][word] = wordTfidf.get(keys[i]).get(word)

    return topKWordsTfidf


# 保存word-tfidf键值对，方便后面查询
wordTfidf = {'room': {}, 'service': {}, 'location': {}, 'facility': {}, 'value': {}, 'environment': {}, 'food': {}}
def transMatrix(corpus):
    # print("corpus = ", corpus)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()

    # print("word = ", word)
    # print(X.toarray())

    transformer = TfidfTransformer()

    # print("transformer = ", transformer)
    tfidf = transformer.fit_transform(X)
    # print(tfidf.toarray())

    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    # print("weigth.shape = ", weight.shape)
    # print("tfidf.shape = ", tfidf.shape)
    # print(weight)
    print("weight's length = ", len(weight))
    print("word's length = ", len(word))
    keys = ['room', 'service', 'location', 'facility', 'value', 'environment', 'food']
    if len(weight) != len(keys):
        print("出错辣！！！")
        sys.exit(-3)
    for i in range(len(weight)):
        for j in range(len(word)):
            # if weight[i][j] > 0:
            wordTfidf[keys[i]][word[j]] = weight[i][j]
            # if i == 0:
            #     print(word[j], weight[i][j])
    '''
    '''

    sort = np.argsort(tfidf.toarray(), axis=1)
    # print("sort.shape = ", sort.shape)
    # top10 = sort[:, -topK:]
    # print("top10", top10)

    key_words = pd.Index(words)[sort].tolist()
    # print(key_words[:10])

    return key_words


if __name__ == "__main__":
    print("start...")

    pre = 'data/result/baidunlp/'
    path = pre + str(topic_number) + '-topic-words-prob.npy'
    topicWordsProb_load = np.load(path, allow_pickle=True).item()
    topics = topicWordsProb_load.keys()

    # 读取数据
    path = pre + 'three-star-negative-business.csv'
    df = pd.read_csv(path, nrows=debug_length if debug else None)

    texts = df['评论文本-negative']
    # 将texts处理为list
    texts_list = []
    for text in texts:
        text = text.strip('[')
        text = text.strip(']')
        text = text.replace("'", "")
        text = text.replace(" ", "")
        text = text.split(',')
        texts_list.append(text)
    print("texts' length = ", len(texts))

    scores = df['score-negative']
    # 将score处理为list
    scores_list = []
    # print("scores = ", scores)
    for score in scores:
        score = score.strip('[')
        score = score.strip(']')
        score = score.split(',')
        score = [float(i) for i in score]
        scores_list.append(score)
    print("scores' length = ", len(scores))

    # 文本预处理，分词 去停用词，方便后面判断主题类型
    texts_list = data_preprocess(texts_list)
    # print("corpus_nostopwords = ", corpus_nostopwords)

    # 判断文本的主题类型
    texts_topics_list = judgeTopic(texts_list)

    if len(df) != len(texts_topics_list) or len(df) != len(texts_list):
        print("数据处理过程有问题，长度不一致。。。")
        sys.exit(-1)

    # 将结果处理为df形式，
    topic_score = produceTopicScore(texts_topics_list, scores_list)
    topic_score = pd.DataFrame(topic_score)
    final_df = pd.concat([df, topic_score], axis=1)
    # print(final_df.columns)
    # print(final_df.head())
    # 删掉所有主题均为空的行
    # final_df = final_df.dropna(axis=0, how='all', subset=['room', 'service', 'location', 'facility', 'environment', 'value'])
    # print(final_df.head())
    # 保存至文件
    path = pre + 'all-star-negative-aspect-debug.csv' if debug else pre + 'all-star-negative-aspect-family.csv'
    # final_df.to_csv(path)

    # 将list合并为一个字符串
    abstract_split_texts = {'room': [], 'service': [], 'location': [], 'facility': [], 'value': [], 'environment': [], 'food': []}
    for key in abstract_texts.keys():
        length = 0
        current_value = abstract_texts.get(key)
        # print("current_value = ", current_value)
        i = 0
        temp = []
        for v in current_value:
            temp.append(v)
            i += 1
            if i == 5:
                abstract_split_texts[key].append(','.join(temp))  # 每5个短文本合并为一个单元，方便使用接口抽取观点
                i = 0
                temp = []
            length += 1
            if length > 25:
                break

        for v in current_value:
            temp.extend(v.split(' '))
        print("temp = ", temp)
        # print(' '.join(temp))
        abstract_texts[key] = ','.join(temp)
    print(abstract_texts)
    print(abstract_split_texts)

    # 根据abstract_texts计算tf-idf
    # print(abstract_texts)
    keys = list(abstract_texts.keys())
    print("keys:", keys)
    keyWords = transMatrix(abstract_texts.values())
    # print(wordTfidf['room'])
    topK = 50
    # showResult(keyWords, topK, keys)
    topKWordsTfidf = searchTfidf(keyWords, topK, keys)
    for key in keys:
        print("current key is ", key)
        for word, tfidf in topKWordsTfidf[key].items():
            print(word, tfidf)
        print("*" * 50)
        print(key, topKWordsTfidf[key])


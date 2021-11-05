import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub
import jieba, pandas as pd, numpy as np, jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import xiangshi as xs

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 30

# 种子词
seed_dictionary = {
    "weight": "大小，分量，重量，千克，kg，KG，Kg，g，克，一斤，二斤，半斤八两，三斤，500g，每斤，大大的，小小的，净重，轻重，半斤，比重，轻",
    "freshness": "新鲜，生产日期，保质期，日期，变质，馊味儿，馊味，防腐，腐烂，腐败，陈旧，月份，新鲜出炉，鲜食，臭肉，保鲜，硝酸盐，鲜度，亚硝酸盐，期限，生鲜食品，鲜肉，长期保持，新鲜肉，六月份，二月份，八月份，腐臭，湿度，状态，多长时间，储存，冻猪肉，月份，坏果",
    "color": "颜色，色泽，红色，鲜艳，发黄，色，成色，蓝绿色，浅褐色，润透，透亮，红润，晶莹剔透，晶莹，鲜红色，发黑，黑色，样子，熏色，黄亮，色料，品色，琥珀色，红亮，白色，色泽鲜明，图案，蛋黄，发黑，黄潢，金黄色，鲜红色，脱色，浅褐色，色相，原色，黄金，表色，鲜艳，增色，腊黄，黄色，红润，颜值，红色，金黄，油黄，乳白色，色彩，货色，厚色，色素，黄黄的，色调，暗红色，发黄，黑亮，透亮，流黄，彩色，深色，黄油，淡黄色，焦黄，外观，外观设计，产品设计，黑乌乌，熏黑",
    "cleanliness": "干净，卫生，异物，脏，异味，发霉，霉斑，霉点，脏兮兮，臭味，臭味儿，发臭，恶心，拉肚子，臭臭，肚子疼，肚子痛，肠胃炎，安全卫生，黑水，杂质，股味，股味儿，油污，污垢，污渍，脏东西，难闻，肉臭，泥巴，散发出，黄泥巴，天臭，脏味，净，污，臭水沟，油污，黑脏，发臭，安全卫生，肠胃炎，拉肚子，异物，粪土，变质，霉菌，老鼠，白霉点，白毛",
    "taste": "味道，好吃，辣辣的，辣，咸，香，辣味，腊味，咸味，咸味儿，香味，香味儿，麻辣，不好吃，难吃，有点咸，口感，正宗，不腻，腻，酒味，酒味儿，甜味，甜味儿，地道，不肥，肥美，肥，肥肉，瘦肉，很浓，美味，微辣，微微辣，排骨，肥瘦，香肠，适中，味儿，肉质，油腻，太咸，太辣，太腻，太肥，太瘦，品尝，烟味，烟味儿，口味，甜味，甜味儿，鲜美，味道鲜美，腊肠，风味，脆骨，腊肉，甜口，滋味，滋味儿，咸太多，腌制品，精肥，精瘦，肥瘦相间，盐分，真香，肥肥，红辣椒，辣椒，瘦润肥，肥度，地地道道，腥味儿，原汁原味，盐分，香气，辣肉，鸡肉，蒸肉，骨肉，牛排，羊腿，脚跟，辣肉，鲜味，肉身，瘦点，香浓，甜度，咸点，巴适，浓香，咸香咸，麻椒味，鲜香，咸甜，太咸太咸，香精，肥肉，肥瘦，肉肥，香料，咸鲜肥，咸口，肥点，香甜可口，咸香超，色香，筋道，香咸微，精肥，米饭，火腿肠，焖饭，糯米饭，火腿，调味，菜品，调味品，烤肠，肉香，火腿肉，玉米面",
    "logistics": "物流，快递，速度，太慢，很快，慢，快，收到，送货，收货，送快递，收快递，快递服务，发货，发货慢，发货快，配送，配送点，快递点，快递站，中通，京东物流，京东快递，圆通，百世，百世汇通，韵达，单号，邮费，快递费，包邮，免运费，不包邮，京准达，冷链，顺丰，邮政，外省，送快递，行货，货差，订货，退换货，速度慢，配点，物流业，神速，丰巢，飞机，朝发夕至，路程",
    "service": "服务，客服，态度，服务态度，优质服务，售后，售前，售后服务，售前服务，售中服务，售后处理，售后态度，冷漠，热情，不主动，不热情，很冷漠，态度差，态度好，态度不好，态度不行，口气，态度恶劣，服务质量，气愤，歉意，礼貌，谢谢你们，谢谢，感谢，人员",
    "packaging": "包装，外包装，内包装，包装袋，密封，密封性，漏液，漏了，漏，小袋，变形，真空，真空包装，袋子，塑料包装，纸箱装，纸箱，包装品，封口，包装品，纸盒，自封袋，豪装，纸盒包装，纸盒子，套装，纸袋，保鲜袋，箱子，礼盒装，礼品盒，礼袋，大礼盒，肠盒，礼袋，双袋，精装，内袋，袋食，内盒",
    "price": "价格，性价比，物美价廉，便宜，偏贵，贵，便宜，不便宜，不贵，优惠，优惠券，活动，买一送一，几块钱，廉价，块钱，贵点，便宜点，市场价，涨价，半价，特惠，全品券，节省，太值，值，价值，物超所值，物有所值，收费，价低，平价",
    "quality": "不好，质量，质量一般，质量不好，质量太差，太差，好，差，总体，整体，物有所值，物超所值，好坏，不错，东西，产品质量，货品，诱人，太湿，差评，品质，产品品质，真品，正品，优质，成品，出品，特差，高端，食品质量，太假，注重质量，伪品，外观设计，废肉，材质，真肉，假肉，韧劲，给力，货给力，失望透顶，碎屑，真棒，太坑，太棒了，太棒，太次，含淀粉，淀粉，前胛，劣质",
    "shop": "店铺，店家，卖家，京东，品牌，体验，购物，公司，信誉，老字号，商家，农户，农民，电商助农，小作坊，作坊，小店，大店，品牌店，旗舰店，供货，批量生产，电商，店里，铺子，靠谱，店老板，厂商，商标，偏远地区，网店，老板，商城，平台，乡村，农村，民族特色，食品厂，人民日报，贵州，云贵川，江苏，中国，原产地内蒙，生产线，石家庄市，哈尔滨，俄罗斯，土家族，苗族，广西，肉联厂，质朴，骗人，昧着良心，京東，衡水市，武昌，徐州，云南，集团，西安，耒风馆，耒风"
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
# dictionary["weight"].update(["日期"])
# print("dictionary:", dictionary)
# seed_dictionary处理
for key, value in seed_dictionary.items():
    value = value.split("，")
    seed_dictionary[key] = value


stoplist = []
f = open('../stopwords.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    stoplist.append(line.strip())
# print("stoplist = ", stoplist)


# 数据预处理
def data_preprocess(texts):
    print("生成词袋。。。")
    corpus = []
    for text in texts:
        # print("text = ", text)
        # 词性标注过滤，保留名词、处所词、方位词、动词、代词
        words = pseg.cut(text)
        # temp = ' '.join(w.word.strip() for w in words)
        temp = ' '.join(w.word.strip() for w in words if (str(w.flag).startswith('n')))
        # temp = ' '.join(w.word.strip() for w in words if (str(w.flag).startswith('n') or str(w.flag).startswith('b') or str(w.flag).startswith('s')))
        # print("temp1 = ", temp)
        temp = temp.split(' ')
        # 去停用词
        temp = ' '.join(word.strip() for word in temp if word not in stoplist and not word.isdigit())
        # temp = ' '.join(word.strip() for word in temp if word not in stoplist and not word.isdigit() and word not in advlist)
        # print("temp = ", temp)
        # print("*" * 50)
        corpus.append(temp)
        # if len(temp) > 0:
        #     print(temp)
    return corpus


def judge(word):
    # print("word:", word)
    attribute = "EMPTY"
    max_similarity = 0
    for key, value in seed_dictionary.items():
        sim = 0
        for v in value:
            sim = max(sim, xs.cossim([word], [v]))
        if max_similarity < sim:
            max_similarity = sim
            attribute = key
        # print(key, max_similarity)
    print(word, attribute)

    return attribute


dictionary_temp = {"freshness": set(),
                   "color": set(),
                   "cleanliness": set(),
                   "taste": set(),
                   "logistics": set(),
                   "service": set(),
                   "packaging": set(),
                   "price": set(),
                   "quality": set(),
                   "shop": set()
                   }
# 判断词汇是否存在于种子词典中
def judgeWords(corpus):
    for line in corpus:
        # print("line:", line)
        words = line.split(" ")
        # print("words:", words)
        for word in words:
            if word in all_words:
                continue
            all_words.update(word)
            attribute = judge(word)
            # print("judgeWords:", attribute)
            # print("*" * 50)
            if attribute != "EMPTY":
                dictionary[attribute].update([word])
                # dictionary_temp[attribute].update([word])


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    version = time.strftime('%Y%m%d%H%M%S', time.localtime(start_time))
    # 读取语料库，获取所有名词
    path_global = "test/merged-test.xlsx" if debug else "data/merged.xlsx"
    data = pd.read_excel(path_global)
    # 去标点符号
    data['words'] = data['评价内容'].apply(lambda x: re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？、~@#￥%……&*（）．；：】【|]+", "", str(x)))
    docs = data["words"].tolist()

    corpus_global = data_preprocess(docs)
    '''
    corpus_global = ['餐椅 感谢信 助力 京华 经理 福州']
    '''
    # 判断是否存在
    judgeWords(corpus_global)
    print(dictionary)
    # print(dictionary_temp)

    # 保存dictionary
    s_path_global = "test/domain_dictionary_" + str(version) + "-test.npy" if debug else "result/domain_dictionary_" + str(version) + ".npy"
    np.save(s_path_global, dictionary)

    dictionary_load = np.load(s_path_global, allow_pickle=True).item()
    print("dictionary_load:", dictionary_load)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


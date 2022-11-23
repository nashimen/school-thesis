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



if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    contain_test()

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")
    sys.exit(10001)


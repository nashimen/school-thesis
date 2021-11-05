import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub
import jieba, pandas as pd, numpy as np, jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


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


# 文本处理
def texts_process(texts):
    texts = texts.strip('[')
    texts = texts.strip(']')
    texts = texts.replace("'", "")
    texts = texts.replace(" ", "")
    texts = texts.split(',')
    # print("texts = ", texts)
    texts = ','.join(text for text in texts)
    # print("texts = ", texts)
    return texts


# 加载停用词
def load_stopwords(path='../stopwords.txt'):
    with open(path, encoding="utf-8") as f:
        stopwords = list(map(lambda x: x.strip(), f.readlines()))
    stopwords.extend([' ', '\t', '\n'])
    return frozenset(stopwords)
    return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))


topicWordsProb = {}
# 主题词+概率打印函数
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        current_topic = 'Topic' + str(topic_idx)
        # print(current_topic)
        current_dict = {}
        for i in range(len(topic)):
            current_dict[feature_names[i]] = topic[i]
            # if feature_names[i] == '房型':
            #     print('before:', current_topic, topic[i])
        topicWordsProb[current_topic] = current_dict

        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[::-1]]))
        # print(current_dict)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    # 获取当天日期+小时+分钟，例如：202110161115
    version = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))

    path_global = "test/merged-test.xlsx" if debug else "data/merged.xlsx"
    data = pd.read_excel(path_global)
    data["评价内容-processed"] = data.apply(lambda row: texts_process(row['评价内容']), axis=1)
    docs = data["评价内容-processed"].tolist()

    corpus = data_preprocess(docs)

    tfidf_model = TfidfVectorizer(stop_words=load_stopwords())
    tfidf_matrix = tfidf_model.fit_transform(corpus)

    topic_number = 12
    lda_model = LatentDirichletAllocation(n_components=topic_number, max_iter=10)
    # 使用TF-IDF矩阵拟合LDA模型
    lda_model.fit(tfidf_matrix)
    # corpus = np.array(corpus)
    # corpus = corpus.reshape(-1, 1)
    # lda_model.perplexity(corpus)

    n_top_words = 5
    tf_feature_names = tfidf_model.get_feature_names()
    print("tf_feature_names' length = ", len(tf_feature_names))
    print_top_words(lda_model, tf_feature_names, n_top_words)

    # 保存主题-单词概率字典
    # print("dict = ", topicWordsProb)
    s_path_global = "test/lda/" + str(topic_number) + "-" + str(version) + ".npy" if debug else "result/lda/" + str(topic_number) + "-" + str(version) + ".npy"
    np.save(s_path_global, topicWordsProb)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


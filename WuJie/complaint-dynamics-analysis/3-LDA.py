# -*- coding: utf-8 -*-

import jieba, pandas as pd, numpy as np, jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation

debug = False
debug_length = 50

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


# 加载停用词
def load_stopwords(path='stopwords.txt'):
    with open(path, encoding="utf-8") as f:
        stopwords = list(map(lambda x: x.strip(), f.readlines()))
    stopwords.extend([' ', '\t', '\n'])
    return frozenset(stopwords)
    return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))


stoplist = []
f = open('stopwords.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    stoplist.append(line.strip())
# print("stoplist = ", stoplist)
advlist = []
f = open('disturbingwords.txt', 'r')
lines = f.readlines()
for line in lines:
    advlist.append(line.strip())
# print("advlist = ", advlist)
# 数据预处理
def data_preprocess(texts):
    print("生成词袋。。。")
    corpus = []
    for text in texts:
        # print("text = ", text)
        # 词性标注过滤，保留名词、处所词、方位词、动词、代词
        words = pseg.cut(text)
        temp = ' '.join(w.word.strip() for w in words if (str(w.flag).startswith('n')))
        # temp = ' '.join(w.word.strip() for w in words if (str(w.flag).startswith('n') or str(w.flag).startswith('b') or str(w.flag).startswith('s')))
        # print("temp1 = ", temp)
        temp = temp.split(' ')
        # 去停用词
        # temp = ' '.join(word.strip() for word in temp if word not in stoplist)
        temp = ' '.join(word.strip() for word in temp if word not in stoplist and not word.isdigit() and word not in advlist)
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


if __name__ == "__main__":
    print("start...")
    pre = 'data/test/'
    path = pre + 'complaint_merged-negative.csv'
    df = pd.read_csv(path, usecols=[9], nrows=debug_length if debug else None)
    # print(df.columns)
    # print(df.head())
    df['shortTexts-negative-new'] = df.apply(lambda row: texts_process(row['shortTexts-negative']), axis=1)
    docs = df['shortTexts-negative-new'].tolist()
    # print(docs)
    corpus = data_preprocess(docs)

    tfidf_model = TfidfVectorizer(stop_words=load_stopwords())
    tfidf_matrix = tfidf_model.fit_transform(corpus)

    topic_number = 10
    lda_model = LatentDirichletAllocation(n_components=topic_number, max_iter=10)
    # 使用TF-IDF矩阵拟合LDA模型
    lda_model.fit(tfidf_matrix)
    lda_model.perplexity(corpus)

    n_top_words = 5
    tf_feature_names = tfidf_model.get_feature_names()
    print("tf_feature_names' length = ", len(tf_feature_names))
    print_top_words(lda_model, tf_feature_names, n_top_words)

    # 保存主题-单词概率字典
    # print("dict = ", topicWordsProb)
    path = pre + str(topic_number) + '-topic-words-prob-debug.npy' if debug else pre + str(topic_number) + '-topic-words-prob.npy'
    np.save(path, topicWordsProb)
    '''
    topicWordsProb_load = np.load(path, allow_pickle=True).item()
    print(topicWordsProb_load)
    topics = topicWordsProb_load.keys()
    print(topics)
    for topic in topics:
        print('after:', topic, topicWordsProb_load.get(topic).get('房型'))
    '''

    # 拟合后模型的实质
    # print("lda_model.components_ = ", lda_model.components_)
    # print(lda_model.components_.shape)

    print("end...")


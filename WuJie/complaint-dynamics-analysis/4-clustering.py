from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.externals import joblib
from sklearn.metrics import silhouette_score
import jieba, pandas as pd, datetime
from matplotlib import font_manager
import matplotlib.pylab as plt
from pylab import *
import warnings

mpl.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings("ignore")

debug = False
debugLength = 50


# 接收sentence_embeddings，训练模型并返回
def trainKMeansModel(sentence_embeddings, n_clusters):
    # 创建KMeans 对象
    model = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=-1)
    # print("sentence_embeddings = ", sentence_embeddings)
    result = model.fit(sentence_embeddings)

    return result


def jieba_tokenize(text):
    return jieba.lcut(text)


def time_fix(day):
    # print("day's type = ", type(day))
    day_time = datetime.datetime.strptime(day, "%Y/%m/%d")
    # print(day_time)
    day_change = day_time.strftime("%Y%m")
    return day_change


# 加载停用词
def load_stopwords(stopwords_path='../stopwords.txt'):
    with open(stopwords_path, encoding="utf-8") as f:
        stopwords = list(map(lambda x: x.strip(), f.readlines()))
    stopwords.extend([' ', '\t', '\n'])
    return frozenset(stopwords)
    return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))


import jieba.posseg as pseg
# 生成停用词表
stoplist = []
f = open('../stopwords.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    stoplist.append(line.strip())
# 数据处理
def data_preprocess(texts):
    # 只去停用词的语料
    corpus_nostopwords = []
    # 原始语料
    corpus_origin = []
    corpus = []
    for text in texts:
        # 词性标注过滤，保留名词、处所词、方位词、动词、代词
        words = pseg.cut(text)
        temp = ' '.join(w.word.strip() for w in words if (str(w.flag).startswith('n')))
        temp = temp.split(' ')
        # 去停用词
        temp = ''.join(word.strip() for word in temp if word not in stoplist and not word.isdigit())
        corpus.append(temp)
        # 只去停用词的语料
        words = pseg.cut(text)
        temp2 = ' '.join(w.word.strip() for w in words)
        temp2 = temp2.split(' ')
        temp2 = ''.join(word.strip() for word in temp2 if word not in stoplist and not word.isdigit())
        corpus_nostopwords.append(temp2)
        corpus_origin.append(text)
        print(text, temp2, temp)
    return corpus, corpus_nostopwords, corpus_origin


# 先验主题库


if __name__ == "__main__":
    print("start...")

    # 读取数据
    path = "data/test/complaint_merged-negative.csv"
    df = pd.read_csv(path, nrows=debugLength if debug else None)
    # print(df.head())
    # print(df["入住日期"])
    # 日期fix
    df["入住日期-fix"] = df.apply(lambda row: time_fix(row["入住日期"]), axis=1)
    # print(df["入住日期-fix"].tolist())
    df = df.loc[(df["星级"].isin([2, 3])) & (df["入住日期-fix"] == "202101")]

    short_texts = df["shortTexts-negative"].tolist()
    text_list = []
    for texts in short_texts:
        texts = texts.strip('[')
        texts = texts.strip(']')
        texts = texts.replace("'", "")
        texts = texts.replace(" ", "")
        text_list.extend(texts.split(","))

    texts1, texts2, texts3 = data_preprocess(text_list)
    # print("texts1 = ", texts1)
    # print("texts2 = ", texts2)
    # print("texts3 = ", texts3)

    print("text_list's length = ", len(text_list))
    # print("text_list:", text_list)

    # 停用词

    # 生成特征
    tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba.cut, stop_words=load_stopwords(), lowercase=False)

    tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)
    # print("tfidf_matrix = ", tfidf_matrix)
    print("tfidf_matrix's type = ", type(tfidf_matrix))
    # 训练模型
    k = 4
    SSE = []  # 根据手肘法判断结果优劣
    SSC = []  # 轮廓系数
    for i in range(2, k):
        km_cluster = KMeans(n_clusters=i, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
        # 返回各自文本的所被分配到的类索引
        # result = km_cluster.fit_predict(tfidf_matrix)
        result = km_cluster.fit(tfidf_matrix)
        print("Predicting result: ", result)
        SSC.append(silhouette_score(tfidf_matrix, result.labels_, metric="euclidean"))
        SSE.append(result.inertia_)

    # 绘制曲线
    plt.plot(range(2, k), SSE, marker="o", label=u'SSE')
    plt.xlabel("K值——簇数量", size=15)
    plt.ylabel("簇内误方差SSE", size=15)
    plt.show()
    plt.plot(range(2, k), SSC, marker="*", label=u'SSC')
    plt.xlabel("K值——簇数量", size=15)
    plt.ylabel("轮廓系数SSC", size=15)
    plt.show()

    '''
    # 将短文本分类
    result_dict = {}
    for i, text in enumerate(text_list):
        if result[i] not in result_dict.keys():
            result_dict[result[i]] = [text]
        else:
            result_dict[result[i]].append(text)

    for key, value in result_dict.items():
        print(key, ":", value)
    '''

    print("end...")


# -*- coding: utf-8 -*-

import pandas as pd, numpy as np, jieba.posseg as pseg, sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# 显示所有列
pd.set_option('display.max_columns', None)

debug = False

print("start...")

# 读取部件领域词汇,分属性保存
field_words_dict = {"comfort": [], "value": [], "power": [], "manipulate": [], "outside": [], "space": [], "energy": [], "inside": []}
path = "data/opinion_extracting.csv"
col_names = ['Attribute', 'Component']
data = pd.read_csv(path, usecols=col_names)
# print(data.head())
# print("data's length = ", len(data))
data = data.drop_duplicates(col_names)  # 只需要部件名称，因此去重
# print(data.head())
# print("data's length = ", len(data))
# 将部件名称整合
field_words = []
for key in field_words_dict.keys():
    current_words = data.loc[data['Attribute'] == key]['Component'].tolist()
    field_words_dict[key] = current_words
    field_words.extend(current_words)
field_words = set(field_words)
# print(field_words_dict)
print("field_words' length = ", len(field_words))
'''
print("*" * 100)
for word in field_words:
    print(word, "100 n")
print("*" * 100)
'''

# 读取不同车型的语料库
file_names = ["big", "jincou", "micro", "mid", "midbig", "small"]
file_name = "small"
path = "data/" + file_name + ".xlsx"
# data = pd.read_excel(path)
# data = pd.read_excel(path, usecols=["最满意评论"])
data = pd.read_excel(path, engine='openpyxl', usecols=[30, 31], nrows=50 if debug else None)
print(data.columns)
print(data.head())

# 数据预处理：分词、去停用词
# 处理为以下格式：abstract_texts = {'key': "房间,真的,非常,大,呢,酒店,的,服务,很,少,味道,差,设施,比较,完善,来看,性价比"}
import jieba  # 导入jieba模块
jieba.load_userdict("config/component_dict.txt")
# 生成停用词表
stop_words = []
f = open(file='config/stopwords.txt', mode='r', encoding='utf-8')  # 文件为123.txt
sourceInLines = f.readlines()
f.close()
for line in sourceInLines:
    temp = line.strip('\n')
    stop_words.append(temp)
# 分别处理最满意和最不满意评论，合并两类评论，计算整体重要度
content = []

satisfy = True
print("正在处理最满意评论数据。。。")
# 处理 最满意评论
# satisfy_content = []
for current_text in data['最满意评论']:
    if str(current_text).strip().isdigit():
        # print("current_text = ", current_text)
        continue
    segs = [word for word in jieba.cut(current_text) if word not in stop_words]
    content.extend(segs)
# satisfy_content = {"key": ",".join(satisfy_content)}
# print("satisfy_content = ", satisfy_content)

print("正在处理最不满意评论数据。。。")
# 处理 最不满意评论
# unsatisfy_content = []
for current_text in data['最不满意评论']:
    if str(current_text).strip().isdigit():
        # print("current_text = ", current_text)
        continue
    segs = [word for word in jieba.cut(current_text) if word not in stop_words]
    content.extend(segs)
# unsatisfy_content = {"key": ",".join(unsatisfy_content)}
# print("unsatisfy_content = ", unsatisfy_content)
# content = {"satisfy": ','.join(satisfy_content), "unsatisfy": ','.join(unsatisfy_content)}
# print(content)
# 合并最满意和最不满意评论
content = {"key": ",".join(content)}

# 计算TFIDF值，保存word-tfidf键值对，方便后面查询
wordTfidf = {'key': {}}
def transMatrix(corpus):
    # print("corpus = ", corpus)
    print("corpus' length = ", len(corpus))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print("weight's length = ", len(weight))
    print("word's length = ", len(word))
    keys = ['key']
    if len(weight) != len(keys):
        print("出错辣！！！")
        sys.exit(-3)
    for i in range(len(weight)):
        for j in range(len(word)):
            wordTfidf[keys[i]][word[j]] = weight[i][j]
    sort = np.argsort(tfidf.toarray(), axis=1)
    key_words = pd.Index(words)[sort].tolist()
    return key_words


# 从wordTfidf中检索TopK词语的tfidf值
def searchTfidf(keyWords, topK, key):
    print("searchTfidf keys = ", key)
    topKWordsTfidf = {'key': {}}
    length = len(keyWords)
    print("length = ", length)
    for i in range(length):  # 依次遍历七个主题
        topKWords = keyWords[i][-topK:]
        for word in topKWords:
            topKWordsTfidf[key[i]][word] = wordTfidf.get(key[i]).get(word)
    return topKWordsTfidf


# 从wordTfidf中检索部件词语的tfidf值
def searchComponentTfidf():
    df = pd.DataFrame(columns=('aspect', 'component', 'tfidf'))
    # 检索哦
    for aspect in field_words_dict.keys():
        components = field_words_dict.get(aspect)
        for component in components:
            t = wordTfidf.get("key").get(component, 'no_value')
            df = df.append([{'aspect': aspect, 'component': component, 'tfidf': t}], ignore_index=True)
    return df


# 检索分属性部件的TFIDF值，并保存至文件
# print("正在处理" + "satisfy" if satisfy else "unsatisfy" + "_content...")
keyWords = transMatrix(content.values())
result = searchComponentTfidf()
print(result.head())
# 保存至文件
result_path = "result/" + file_name + "_content.csv"
print("result_path = ", result_path)
result.to_csv(result_path, mode='w', encoding='utf_8_sig', index=False)

'''
topK = 50
keys = list(satisfy_content.keys())
print("keys:", keys)
topKWordsTfidf = searchTfidf(keyWords, topK, keys)

for key in keys:
    for word, tfidf in topKWordsTfidf[key].items():
        print(word, tfidf)
    print("current key is ", key)
    print("*" * 50)
    print(key, topKWordsTfidf[key])
'''

print("end...")


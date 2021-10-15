import re
import os
import pandas as pd
import jieba
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# 显示所有列
pd.set_option('display.max_columns', None)

debug = False
debugLength = 72

# 加载停用词
def load_stopwords(path='stopwords.txt'):
    with open(path, encoding="utf-8") as f:
        stopwords = list(map(lambda x: x.strip(), f.readlines()))
    stopwords.extend([' ', '\t', '\n'])
    return frozenset(stopwords)
    return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))


threshold = 0.7
tfidf_model = TfidfVectorizer(tokenizer=jieba.cut, stop_words=load_stopwords(), analyzer='char_wb')
# 使用TextRank算法提取摘要
def get_abstract(content):
    # 使用逗号替换空格
    content = re.sub(r" +", ",", str(content))
    pattern = r',|\.|;|\?|:|!|\(|\)|，|。|、|；|·|！| |…|（|）|~|～|：|；'
    docs = list(re.split(pattern, content))
    # print("docs pre = ", docs)
    docs = list(filter(lambda x: len(str(x).strip()) > 3, docs))
    # print("docs = ", docs)
    if len(docs) == 0:
        return ''
    normalized_matrix = tfidf_model.fit_transform(docs)
    normalized_matrix_array = normalized_matrix.toarray()
    similarity = nx.from_scipy_sparse_matrix(normalized_matrix * normalized_matrix.T)
    scores = nx.pagerank(similarity)
    tops = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # 所有
    indices = list(map(lambda x: x[0], tops))

    # 遍历normalized_matrix_array，依次保存符合条件的摘要
    # 参考文献：Extractive summarization using supervised and unsupervised learning
    # 计算余弦相似度矩阵
    cosine_matrix = cosine_similarity(normalized_matrix_array)
    indices_final = []
    for index in indices:
        if len(indices_final) == 0:
            indices_final.append(index)  # 先放入第一个摘要
            continue
        max_consine = 0
        for current_index in indices_final:
            if cosine_matrix[index][current_index] > max_consine:
                max_consine = cosine_matrix[index][current_index]
        if max_consine < threshold:
            indices_final.append(index)
    return list(map(lambda idx: docs[idx], indices_final))


origin_files_directory = 'data/'
star = 'three-star'
# 读取文件+生成文本摘要+写入文件
def read_write_origin_files():
    for file in os.listdir(origin_files_directory + star):
        hotelName = os.path.splitext(file)[0]
        print("current hotel is ", hotelName)
        path = origin_files_directory + star + '/' + hotelName + '.xlsx'
        originData = pd.read_excel(path, usecols=[1, 2, 4, 5, 6], nrows=debugLength if debug else None)
        # 增加两列，表示数据为哪个酒店+星级
        originData['hotelName'] = hotelName
        originData['star'] = 2 if star == 'two-star' else 3 if star == 'three-star' else 5
        # 生成本文摘要
        originData['评论文本'] = originData.apply(lambda row: get_abstract(row['评论文本']), axis=1)
        # 删除空值
        colNames = originData.columns
        originData = originData.dropna(axis=0, subset=colNames)
        targetPath = origin_files_directory + star + '.csv'
        # originData.to_csv(targetPath, encoding='utf_8_sig', index=False, header=None, mode='a')
        originData.to_csv(targetPath, mode='a')

    return originData


if __name__ == "__main__":
    print("start...")
    originData = read_write_origin_files()
    # print(originData['评论文本'])
    print(originData.columns)
    # print(originData)
    # print(originData['评论文本'])

    print("end...")


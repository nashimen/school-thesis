# 利用TextRank，提取文本摘要
import jieba
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


def cut_sentence(sentence):
    """
    分句
    :param sentence:
    :return:
    """
    # if not isinstance(sentence, unicode):
    # sentence = sentence.decode('utf-8')
    delimiters = frozenset(u'.,，?。！？\n')
    buf = []
    for ch in sentence:
        buf.append(ch)
        if delimiters.__contains__(ch):
            yield ''.join(buf)
            buf = []
    if buf:
        yield ''.join(buf)


def load_stopwords(path='stopwords.txt'):
    """
    加载停用词
    :param path:
    :return:
    """
    with open(path, encoding="utf-8") as f:
        # stopwords = filter(lambda x: x, list(map(lambda x: x.strip(), f.readlines())))
        stopwords = list(map(lambda x: x.strip(), f.readlines()))
    stopwords.extend([' ', '\t', '\n'])
    return frozenset(stopwords)
    return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))


def get_abstract(content, size=4):
    """
    利用textrank提取摘要
    :param content:
    :param size:
    :return:
    """
    docs = list(cut_sentence(content))
    tfidf_model = TfidfVectorizer(tokenizer=jieba.cut, stop_words=load_stopwords())
    tfidf_matrix = tfidf_model.fit_transform(docs)
    normalized_matrix = TfidfTransformer().fit_transform(tfidf_matrix)
    similarity = nx.from_scipy_sparse_matrix(normalized_matrix * normalized_matrix.T)
    scores = nx.pagerank(similarity)
    tops = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    size = min(size, len(docs))
    indices = list(map(lambda x: x[0], tops))[:size]
    return list(map(lambda idx: docs[idx], indices))


text = u'''
这个宾馆总体感觉还不错，无论从价格、出行、购物，还是订机票（国航就在它的右边）都很方便。只是房间没有什么电脑，也不知道它说的免费上网在什么地方可以。补充点评2007年10月12日：对了，去机场也很方便，机场大巴就在它下面，每半个小时一趟
'''
# 读取文件
# f=open(r"D:\temp.txt",encoding="utf-8")
# text=f.read()
# f.close()
for i in get_abstract(text, 12):
    print(i)


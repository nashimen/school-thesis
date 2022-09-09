import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import paddlehub as hub

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False


# 判断是否包含中文字符
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


# 提取短文本（即句子）
def get_abstract(content):
    # print("content = ", content)
    # 使用逗号替换空格
    content = re.sub(r" +", ",", str(content))
    pattern = r',|\.|;|\?|:|!|\(|\)|，|。|、|；|·|！| |…|（|）|~|～|：|；'
    docs = list(re.split(pattern, content))
    # print("docs pre = ", docs)
    docs = list(filter(lambda x: len(str(x).strip()) > 0, docs))
    # 去掉不包含中文字符的短文本
    remove_index = []
    for i in range(len(docs)):
        if not is_Chinese(docs[i]):
            remove_index.append(i)
    if len(remove_index) > 0:
        print("非中文1：", docs)

    for counter, index in enumerate(remove_index):
        index = index - counter
        docs.pop(index)
    if len(remove_index) > 0:
        print("非中文2：", docs)
        print("*" * 50)

    return docs


# 加载数据+提取短文本
def load_data(path, s_path):
    current_data = pd.read_excel(path)
    print(current_data.columns)
    print("current_data's length = ", len(current_data))

    # 提取短文本
    print("正在提取短文本")
    current_data["shortTexts"] = current_data.apply(lambda row: get_abstract(row['评论文本']), axis=1)
    print("current_data['shortTexts']'s length = ", len(current_data["shortTexts"]))
    # print(current_data.head())
    # 删除空白行
    # colNames = current_data.columns
    # current_data = current_data.dropna(axis=0, subset=colNames)
    # 保存文件
    current_data.to_excel(s_path, index=False)


def calculate_sentiment_by_row(texts):
    texts = texts.strip('[')
    texts = texts.strip(']')
    texts = texts.replace("'", "")
    texts = texts.replace(" ", "")
    texts = texts.split(',')
    input_dict = {"text": texts}
    res = senta.sentiment_classify(data=input_dict)
    positive_probs = []
    for r in res:
        # print(r["sentiment_label"], r["sentiment_key"], r['positive_probs'], r["negative_probs"], r["text"])
        positive_probs.append(r['positive_probs'])
    if len(texts) != len(positive_probs):
        print("长度不一致！！出错辣！！！")
        print(positive_probs)
        print(texts)
    return positive_probs


# 计算短文本情感
senta = hub.Module(name='senta_cnn')
def calculate_sentiment(path, s_path):
    current_data = pd.read_excel(path)
    print("正在计算情感分数。。。")
    current_data['score'] = current_data.apply(lambda row: calculate_sentiment_by_row(row['shortTexts']), axis=1)
    current_data.to_excel(s_path, index=False)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取文件+短文本提取
    print(">>>正在读取数据。。。")
    path_global = "test/data-test.xlsx" if debug else "data/data.xlsx"
    print("path_global:", path_global)
    s_path_global = "test/1-shortTexts-test.xlsx" if debug else "result/1-shortTexts.xlsx"
    # 判断文件是否存在，如果不存在则执行
    if not os.path.exists(s_path_global):
        print(s_path_global, "does not exist")
        load_data(path_global, s_path_global)
    else:
        print(s_path_global, "exists")

    # 短文本情感计算
    s_path_score_global = "test/2-score-test.xlsx" if debug else "result/2-score.xlsx"
    # 判断文件是否存在，如果不存在则执行
    if not os.path.exists(s_path_score_global):
        print(s_path_score_global, "does not exist")
        calculate_sentiment(s_path_global, s_path_score_global)
    else:
        print(s_path_score_global, "exists")


import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import xiangshi as xs

debug = False


# 读取文件，文件读取函数
def read_file(filename):
    # with open(filename, 'rb')as f:
    with open(filename, 'r', encoding='utf-8')as f:
        text = f.read()
        # 返回list类型数据
        text = text.split('\n')
    return text


#文本分句
def cut_sentence(text):
    pattern = r',|\.|;|\?|:|!|\(|\)|，|。|、|；|·|！| |…|（|）|~|～|：|；'
    sentences = list(re.split(pattern, text))
    print("sentences = ", sentences)
    sentence_list = [w for w in sentences]
    return sentence_list


#去停用词函数
def del_stopwords(words):
    # 读取停用词表
    stopwords = read_file(r"../stopwords.txt")
    # 去除停用词后的句子
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words


# 获取六种权值的词，根据要求返回list，这个函数是为了配合Django的views下的函数使用
def weighted_value(request):
    result_dict = []
    if request == "one":
        result_dict = read_file(r"lexicons\most.txt")
    elif request == "two":
        result_dict = read_file(r"lexicons\very.txt")
    elif request == "three":
        result_dict = read_file(r"lexicons\more.txt")
    elif request == "four":
        result_dict = read_file(r"lexicons\ish.txt")
    elif request == "five":
        result_dict = read_file(r"lexicons\insufficiently.txt")
    elif request == "six":
        result_dict = read_file(r"lexicons\inverse.txt")
    elif request == 'posdict':
        result_dict = read_file("lexicons\pos_all_dict.txt")
    elif request == 'negdict':
        result_dict = read_file("lexicons\\neg_all_dict.txt")
    else:
        pass
    return result_dict


print("reading sentiment dict .......")
#读取情感词典
posdict = weighted_value('posdict')
negdict = weighted_value('negdict')
# 读取程度副词词典
# 权值为2
mostdict = weighted_value('one')
# 权值为1.75
verydict = weighted_value('two')
# 权值为1.50
moredict = weighted_value('three')
# 权值为1.25
ishdict = weighted_value('four')
# 权值为0.25
insufficientdict = weighted_value('five')
# 权值为-1
inversedict = weighted_value('six')


# 程度副词处理，对不同的程度副词给予不同的权重
def match_adverb(word, sentiment_value):
    # 最高级权重为
    if word in mostdict:
        sentiment_value *= 8
    # 比较级权重
    elif word in verydict:
        sentiment_value *= 6
    # 比较级权重
    elif word in moredict:
        sentiment_value *= 4
    # 轻微程度词权重
    elif word in ishdict:
        sentiment_value *= 2
    # 相对程度词权重
    elif word in insufficientdict:
        sentiment_value *= 0.5
    # 否定词权重
    elif word in inversedict:
        sentiment_value *= -1
    else:
        sentiment_value *= 1
    return sentiment_value


# 对每一条微博打分
def single_sentiment_score(sent):
    if pd.isna(sent):
        return -2
    # 分词
    words = list(jieba.cut(sent))
    seg_words = del_stopwords(words)
    # i，s 记录情感词和程度词出现的位置
    i = 0   # 记录扫描到的词位子
    s = 0   # 记录情感词的位置
    poscount = 0  # 记录积极情感词数目
    negcount = 0  # 记录消极情感词数目

    # 逐个查找情感词
    for word in seg_words:
        # 如果为积极词汇
        if word in posdict:
            poscount += 1  # 情感词数目加1
            # 在情感词前面寻找程度副词
            for w in seg_words[s:i]:
                poscount = match_adverb(w, poscount)
            s = i+1  # 记录情感词位置
            # 如果是消极情感词
        elif word in negdict:
            negcount += 1
            for w in seg_words[s:i]:
                negcount = match_adverb(w, negcount)
            s = i+1
        # 如果结尾为感叹号或者问号，表示句子结束，并且倒序查找感叹号前的情感词，权重+4
        elif word == '!' or word == '！' or word == '?' or word == '？':
            for w2 in seg_words[::-1]:
                # 如果为积极词，poscount+2
                if w2 in posdict:
                    poscount += 4
                    break
                # 如果是消极词，negcount+2
                elif w2 in negdict:
                    negcount += 4
                    break
        i += 1  # 定位情感词的位置
    # 计算情感值
    sentiment_score = poscount - negcount

    return 1 if sentiment_score > 0 else -1 if sentiment_score < 0 else 0


# 分析test_data.txt 中的所有微博，返回一个列表，列表中元素为（分值，微博）元组
def run_score(contents):
    # 待处理数据
    scores_list = []
    for content in contents:
        if content != '':
            score = single_sentiment_score(content)  # 对每条微博调用函数求得打分
            scores_list.append((score, content))  # 形成（分数，微博）元组
    return scores_list


if __name__ == "__main__":
    print("开始执行main函数咯。。。")

    # 读取数据
    path_global = "test/shortTexts-test.xlsx" if debug else "result/shortTexts-validation.xlsx"
    data_global = pd.read_excel(path_global, engine="openpyxl")
    # print(data_global.columns)

    # 依次计算不同属性情感
    columns = data_global.columns
    data_result = pd.DataFrame()
    for col in columns:
        data_result[col] = data_global.apply(lambda row_global: single_sentiment_score(row_global[col]), axis=1)

    s_path = "test/predicted-test.xlsx" if debug else "result/predicted.xlsx"
    data_result.to_excel(s_path, index=False)

    print("main函数执行结束咯。。。")


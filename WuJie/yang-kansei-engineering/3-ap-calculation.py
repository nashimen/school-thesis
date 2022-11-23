import time, codecs, csv, math, numpy as np, random, datetime, os, gc, pandas as pd, jieba, re, sys
import paddlehub as hub
# from pylab import *

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 30


# 加载停用词
def getStopList():
    stoplist = pd.read_csv(filepath_or_buffer='../stopwords.txt').values
    return stoplist


# 判断是否包含中文字符
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


# 读取文件，文件读取函数
def read_file(filename):
    # with open(filename, 'rb')as f:
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        # 返回list类型数据
        text = text.split('\n')
    return text


# 读取所需文件
most = read_file("lexicons/most.txt")
very = read_file("lexicons/very.txt")
more = read_file("lexicons/more.txt")
ish = read_file("lexicons/ish.txt")
insufficiently = read_file("lexicons/insufficiently.txt")
inverse = read_file("lexicons/inverse.txt")

# 读取停用词表
stop_words = read_file(r"../stopwords.txt")
print('origin stop length: ' + str(len(stop_words)))

# 去掉停用词中的情感词
# 情感词与停用词有重合导致一些文本分数为0
stop_df = pd.DataFrame(stop_words)
senti_df = pd.read_excel('test/3 Kansei word sentiment lexicon-20221002.xlsx' if debug else 'data/3 Kansei word sentiment lexicon.xlsx', engine="openpyxl")
stop_df.columns = ['word']
duplicated = pd.merge(stop_df, senti_df, on='word')['word'].tolist()
stop_words = list(filter(lambda x: x not in duplicated, stop_words))
print('remove sentiment stop length: ' + str(len(stop_words)))

# 去掉停用词中的程度词
# 合并程度词
degree_word = most + very + more + ish + insufficiently + inverse
stop_words = list(filter(lambda x: x not in degree_word, stop_words))
print('remove degree stop length: ' + str(len(stop_words)))


# 读取情感词及分数
def get_senti_word():
    sentiment_dict = senti_df.set_index(keys='word')['sentiment'].to_dict()
    return sentiment_dict


# 去停用词函数
def del_stopwords(words):
    # 去除停用词后的句子
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words


# 获取六种权值的词，根据要求返回list，这个函数是为了配合Django的views下的函数使用
def weighted_value(request):
    result_dict = []
    if request == "most":
        result_dict = most
    elif request == "very":
        result_dict = very
    elif request == "more":
        result_dict = more
    elif request == "ish":
        result_dict = ish
    elif request == "insufficiently":
        result_dict = insufficiently
    elif request == "inverse":
        result_dict = inverse
    elif request == 'senti':
        result_dict = get_senti_word()
    # elif request == 'pos_dict':
    #     result_dict = get_senti_word(polar='pos')
    # elif request == 'neg_dict':
    #     result_dict = get_senti_word(polar='neg')
    else:
        pass
    return result_dict


print("reading sentiment dict .......")
# 读取情感词典
senti_dict = weighted_value('senti')

# 读取程度副词词典
# 权值为2
most_dict = weighted_value('most')
# 权值为1.75
very_dict = weighted_value('very')
# 权值为1.50
more_dict = weighted_value('more')
# 权值为1.25
ish_dict = weighted_value('ish')
# 权值为0.25
insufficient_dict = weighted_value('insufficiently')
# 权值为-1
inverse_dict = weighted_value('inverse')


# 程度副词处理，对不同的程度副词给予不同的权重
def match_adverb(word, sentiment_value):
    # 最高级权重为
    if word in most_dict:
        sentiment_value *= 2
    # 比较级权重
    elif word in very_dict:
        sentiment_value *= 1.75
    # 比较级权重
    elif word in more_dict:
        sentiment_value *= 1.5
    # 轻微程度词权重
    elif word in ish_dict:
        sentiment_value *= 1.25
    # 相对程度词权重
    elif word in insufficient_dict:
        sentiment_value *= 0.25
    # 否定词权重
    elif word in inverse_dict:
        sentiment_value *= -1
    else:
        sentiment_value *= 1
    return sentiment_value


# 每个句子打分
def single_sentiment_score(sent):
    # if debug:
    print("text:", sent)
    if pd.isna(sent):
        return -10086
    # 预处理
    words = list(jieba.cut(str(sent)))
    seg_words = del_stopwords(words)
    senti_pos = []
    score = []
    # 记录情感词位置
    for i, word in enumerate(seg_words):
        if word in senti_dict.keys():
            senti_pos.append(i)
    # 如果没有情感词，则返回-2
    if len(senti_pos) <= 0:
        return -10086

    # 计算情感分数
    for i in range(len(senti_pos)):
        pos = senti_pos[i]
        senti_word = seg_words[pos]
        word_score = senti_dict.get(senti_word)
        # 每个情感词的程度词范围为此情感词与上个情感词之间
        if i == 0:
            last_pos = 0
        else:
            last_pos = senti_pos[i - 1]

        # 程度词范围
        degree_range = seg_words[last_pos + 1: pos]
        # 对程度词范围去重，出现多个相同程度词时只计算一次
        degree_range = set(degree_range)
        for w in degree_range:
            word_score = match_adverb(w, word_score)
        score.append(word_score)

    sentiment_score = sum(score)
    return sentiment_score


# 提取短文本
def get_abstract(content):
    # print("content = ", content)
    # 使用逗号替换空格
    content = re.sub(r" +", ",", str(content))
    pattern = r',|\.|;|\?|:|!|\(|\)|，|。|、|；|·|！| |…|（|）|~|～|：|；'
    docs = list(re.split(pattern, content))
    # print("docs pre = ", docs)
    docs = list(filter(lambda x: len(str(x).strip()) > 0, docs))
    # 去掉不包含中文字符的短文本
    for doc in docs:
        if not is_Chinese(doc):
            print("非中文：", doc)
            docs.remove(doc)

    return docs


# 删除没有情感分数的文本
def remove_texts_scores(path):
    current_data = pd.read_excel(path, nrows=debugLength if debug else None, engine="openpyxl")
    print("正在删除没有情感分数的文本。。。")
    current_data[['score-fix', 'shortTexts-fix']] = current_data.apply(lambda row: remove_10086_texts(row['score'], row['shortTexts']),
                                                                       axis=1, result_type='expand')
    s_path = "test/Beijing & Shanghai + score-v4.xlsx" if debug else "data/Beijing & Shanghai + score-v4.xlsx"
    current_data.to_excel(s_path, index=False)


# 删除-10086的文本,即没有情感分数的文本
def remove_10086_texts(scores, texts):

    scores = scores.strip('[')
    scores = scores.strip(']')
    scores = scores.replace(" ", "")
    scores = scores.split(',')
    # print("scores:", scores)

    texts = texts.strip('[')
    texts = texts.strip(']')
    texts = texts.replace("'", "")
    texts = texts.replace(" ", "")
    texts = texts.split(',')

    if len(scores) != len(texts):
        print("scores texts长度不一致:", scores, texts)

    length = len(scores)
    temp = []
    for i in range(length):
        if scores[i] != '-10086':
            temp.append(i)

    # print("temp:", temp)

    scores = list(map(float, scores))
    result_scores = np.array(scores)[temp].tolist()
    result_texts = np.array(texts)[temp].tolist()

    if len(result_scores) != len(result_texts):
        print("长度不一致！！出错辣！！！")
        print(result_scores)
        print(result_texts)
    return result_scores, result_texts


# 保留中文字符，删除非中文字符
def find_Chinese(doc):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    Chinese = re.sub(pattern, "", doc)
    # print(Chinese)
    return Chinese


digits = '0123456789'
# 文本预处理,shortTexts-list type
def text_processing(shortTexts, scores):
    result = []
    result_scores = []
    result_origin = []
    for current_scores in scores:
        current_scores = current_scores.strip('[')
        current_scores = current_scores.strip(']')
        current_scores = current_scores.replace(" ", "")
        current_scores = current_scores.split(',')
        # print("current_scores:", current_scores)
        # print("current_scores' type:", type(current_scores))
        result_scores.append(list(map(float, current_scores)))

    for texts in shortTexts:
        texts = texts.strip('[')
        texts = texts.strip(']')
        texts = texts.replace("'", "")
        texts = texts.replace(" ", "")
        texts = texts.split(',')
        # print("texts:", texts)
        result_origin.append(texts)

        current_result = []
        for line in texts:
            origin_line = line
            # 去标点符号
            line = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(line))
            # 去掉数字
            line = line.translate(str.maketrans('', '', digits))
            # 去掉非中文字符
            line = find_Chinese(line)
            # 分词
            line = jieba.cut(line)
            # 去停用词
            stoplist = getStopList()
            line = [word.strip() for word in line if word not in stoplist]
            line = ' '.join(line)
            if len(line.strip()) == 0:
                print("当前短文本预处理后为空：", origin_line)
            current_result.append(line)
        result.append(current_result)
    # print("result_origin:", result_origin)
    return result, result_scores, result_origin


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # read corpus, namely online reviews
    corpus_path = "test/1 Corpus-test.xlsx" if debug else "data/1 Corpus.xlsx"
    corpus = pd.read_excel(corpus_path, engine="openpyxl")

    # 属性
    attributes = ["外观", "内饰", "空间", "动力", "操控", "能耗", "舒适性", "性价比"]

    # 依次计算每个属性的Kansei表现
    for attribute in attributes:
        corpus[attribute + "_kansei"] = corpus.apply(lambda row: single_sentiment_score(row[attribute + "评论"]), axis=1)

    # 删除评论文本，将其他结果保存至文件
    # for attribute in attributes:
    #     corpus = corpus.drop(attribute + "评论", axis=1)

    print(corpus.head())

    # 移除没有Kansei表现的评论
    # path = "test/Beijing & Shanghai + score-v3.xlsx" if debug else "data/Beijing & Shanghai + score-v3.xlsx"
    # remove_texts_scores(path)

    # 保存结果
    s_path = "test/4 Kansei sentiment calculation.xlsx" if debug else "result/3 Kansei sentiment calculation.xlsx"
    corpus.to_excel(s_path, index=False)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    sys.exit(10086)


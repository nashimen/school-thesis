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


# 情感计算，输入一行
# 初始化接口
senta = hub.Module(name='senta_bilstm')
def sentiment_calculating(row):
    if not is_Chinese(row):
        print("row:", row)
        return 0.5
    # print("row2:", row)

    # print("row:", row)
    input_dict = {"text": [row]}
    res = senta.sentiment_classify(data=input_dict)
    positive_probs = []
    for r in res:
        positive_probs.append(r['positive_probs'])

    # if debug:
    #     print(positive_probs)

    return positive_probs[0]


# 先验词典读取
def get_field_words(path):
    result = {"comfort": [], "value": [], "power": [], "manipulate": [], "outside": [], "space": [], "energy": [], "inside": []}
    col_names = ['Attribute', 'Component']
    data = pd.read_csv(path, usecols=col_names)
    data = data.drop_duplicates(col_names)  # 只需要部件名称，因此去重
    # 将部件名称整合
    for key in result.keys():
        current_words = data.loc[data['Attribute'] == key]['Component'].tolist()
        result[key] = current_words
    return result


# 生成停用词表
stop_words = []
f = open(file='../stopwords.txt', mode='r', encoding='utf-8')  # 文件为123.txt
sourceInLines = f.readlines()
f.close()
for line in sourceInLines:
    temp = line.strip('\n')
    stop_words.append(temp)


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


# 购车原因中的属性统计，返回属性和个数
def find_aspects(row):
    # print("row:", row)
    result_aspects = []
    # 去标点符号→分词→去停用词→依次遍历
    # 去标点符号
    line = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(row))
    line = line.replace("\n", "").replace("\r", "")
    words = []
    for word in jieba.cut(line):
        if word not in stop_words and not is_number(word):
            words.append(word)
    # 遍历每个词语，判断是否为某个属性
    for word in words:
        for aspect in domain_words.keys():
            if word in domain_words.get(aspect):
                result_aspects.append(aspect)
                continue
    if len(result_aspects) == 0:
        print("未明确提及属性:", row)
    result_aspects = set(result_aspects)
    # print(" ".join(result_aspects), len(result_aspects))
    return " ".join(result_aspects), len(result_aspects)


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 读取领域先验词库
    domain_words_path = "领域先验词库.csv"
    domain_words = get_field_words(domain_words_path)

    dictory = "test/data/" if debug else "data/"

    # 读取当前目录下所有文件
    file_names = os.listdir(dictory)
    file_names = [name.split(".")[0] for name in file_names]
    print("数据文件包括：", file_names)

    columns = ["空间评论", "动力评论", "操控评论", "能耗评论", "舒适性评论", "外观评论", "内饰评论", "性价比评论"]

    # 依次处理每个文件：计算属性情感、期望值、最满意&最不满意情感
    for name in file_names:
        path_global = dictory + name + ".xlsx"
        # s_path_global = "test/" + name + "_result-test.xlsx" if debug else "result/" + name + "_result.xlsx"

        # 读取文件
        print("正在读取数据:", path_global)
        data_global = pd.read_excel(path_global)

        # 最满意&最不满意评论情感计算
        # 依次判断几个属性列是否存在（计算完成），存在则跳过
        if "最满意评论情感" not in data_global.columns.tolist():
            print("正在计算最满意评论情感...")
            data_global["最满意评论情感"] = data_global.apply(lambda row_global: sentiment_calculating(str(row_global["最满意评论"])), axis=1)
            data_global.to_excel(path_global, index=False, engine='xlsxwriter')
        if "最不满意评论情感" not in data_global.columns.tolist():
            print("正在计算最不满意评论情感...")
            data_global["最不满意评论情感"] = data_global.apply(lambda row_global: sentiment_calculating(str(row_global["最不满意评论"])), axis=1)
            data_global.to_excel(path_global, index=False, engine='xlsxwriter')
        if "整体情感" not in data_global.columns.tolist():
            print("正在计算整体情感...")
            data_global["整体情感"] = data_global.apply(lambda row_global: sentiment_calculating(str(row_global["最满意评论"]) + str(row_global["最不满意评论"])), axis=1)
            data_global.to_excel(path_global, index=False, engine='xlsxwriter')

        # 属性情感计算
        for col in columns:
            if col + "情感" not in data_global.columns.tolist():
                print("正在计算属性情感：", col)
                data_global[col + "情感"] = data_global.apply(lambda row_global: sentiment_calculating(str(row_global[col])), axis=1)
                data_global.to_excel(path_global, index=False, engine='xlsxwriter')
        # 期望值计算：给出提及的属性+个数
        if "mentioned_aspects" not in data_global.columns.tolist():
            print("正在计算期望...")
            processed = data_global.apply(lambda row_global: find_aspects(str(row_global["购车原因"])), axis=1)
            data_global["mentioned_aspects"], data_global["mentioned_aspects_count"] = np.array(processed.to_list()).T
            data_global.to_excel(path_global, index=False, engine='xlsxwriter')

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


import time, numpy as np, pandas as pd, jieba, re, sys
from paddlenlp import Taskflow

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = False
debugLength = 50


# 观点提取函数
# 输入sentences，返回两个list，一个特征词list，一个观点词list
def extract_opinion(sentences, interface):
    features = []
    opinions = []
    for sentence in sentences:
        elements = interface(sentence)[0]
        if len(elements) == 0:
            print("sentence:", sentence)
            continue
        elements = elements["评价维度"]
        for element in elements:
            # print("element:", element)
            feature = element["text"]
            opinion = element['relations']
            # 没有观点词
            if len(opinion) == 0 or "观点词" not in opinion.keys():
                print("element:", element)
                continue
            # print("opinion:", opinion)
            opinion = opinion["观点词"][0]["text"]
            if len(feature) > 0 and len(opinion) > 0:
                features.append(feature)
                opinions.append(opinion)

    return features, opinions


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 初始化paddle接口
    schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']} # Define the schema for opinion extraction
    ie = Taskflow("information_extraction", schema=schema)

    # 读取文件+短文本提取
    village = "宏村"
    print(">>>正在读取数据。。。")
    path = "data/10 Aspect-level sentences for opinion extraction.xlsx"
    data = pd.read_excel(path, engine="openpyxl", nrows=debugLength if debug else None)
    # print("First ", data.head())
    data = data.loc[data["Village"] == village]
    # print("Second ", data.head())

    # 依次处理每个属性下的sentence，提取负向观点
    attributes = ["Food", "Hospitality", "Culture", "Nature", "Price"]
    for attribute in attributes:
        df = pd.DataFrame()
        print("正在处理：", attribute)
        # 获取当前属性情感为负的sentences
        data_attribute = data.loc[data[attribute + "_label"] <= 0.5][attribute]
        # print(data_attribute)
        attribute_features, attribute_opinions = extract_opinion(data_attribute, ie)
        df[str(attribute + "_entity")] = pd.Series(attribute_features)
        df[str(attribute + "_opinion")] = pd.Series(attribute_opinions)
        # print("*" * 50)
        s_path = "result/negative opinions/" + village + "_" + attribute + ".xlsx"
        df.to_excel(s_path, index=False)

    # 保存数据

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")
    sys.exit(10001)


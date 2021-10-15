import pandas as pd
from aip import AipNlp
import sys
import datetime
import traceback
import time
sys.setrecursionlimit(5000000)

debug = False
debugLength = 10

""" 你的 APPID AK SK """
APP_ID = '24520514'
API_KEY = 'NH1mjaWqSj1WWuBB74vEWBoj'
SECRET_KEY = 'gQnGvSfR2KRhD6GOhR1Wbj9lXw8YTz4n'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
options = dict()
options["type"] = 10

# 结果保存路径
result_path = 'result/opinion_extracting.csv'

# 读取数据
path = 'data/口碑爬取_中型-大于100条-debug.xlsx' if debug else 'data/口碑爬取_中型-大于100条.xlsx'
origin_data = pd.read_excel(path, usecols=[7, 8, 14, 16, 18, 20, 22, 24, 26, 28], nrows=debugLength if debug else None)
# print(origin_data.columns)
attributes = ['space', 'power', 'manipulate', 'energy', 'comfort', 'outside', 'inside', 'value']

# 统计不同车系下的车型,格式为types = {车系:[车型们]，车系2：[车型们}
types = {}
series = list(set(origin_data['车系']))
# print("series = ", series)
for current_series in series:
    # print("current_series = ", current_series)
    models = list(set(origin_data.loc[origin_data['车系'] == current_series]['车型']))
    types[current_series] = models
print("types = ", types)


def calculate(text):
    local_result_api = ''
    try:
        local_result_api = client.commentTag(text, options)
        error_code = 'error_code'
        i = 0
        while error_code in local_result_api.keys():
            if i > 50:
                print(local_result_api)
                break
            local_result_api = client.commentTag(text, options)
            i = i + 1
        if i > 10:
            print("接口访问了", i, '次:', text)
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt text = ", text)
        print("KeyboardInterrupt local_result_api = ", local_result_api)
        traceback.print_exc()
    except Exception as e:
        print("Exception text = ", text)
        print("Exception local_result_api = ", local_result_api)
        traceback.print_exc()
        # calculate(texts)
    return local_result_api

# 依次遍历车系、车型、属性，访问baidu接口进行观点抽取
# 处理完一个属性就写入一次文件
series_done = ['迈锐宝', '凯迪拉克CT4', '奥迪A4L', '沃尔沃S60', '思锐', '一汽-大众CC', '雪铁龙C5', '凯美瑞']
for current_series in series:
    if current_series in series_done:
        continue
    for model in types.get(current_series):
        if current_series == '亚洲龙' and model in ['2019款双擎2.5LLimited旗舰版国VI', '2019款双擎2.5LXLE尊贵版国VI']:
            continue
        if current_series == '阿特兹' and model in ['2020款2.5L蓝天至尊版']:
            continue
        if current_series == '名图' and model in ['2017款1.8L自动智能型GLS国V', '2017款1.6T自动智能型GLS国V']:
            continue
        if current_series == '迈锐宝XL' and model in ['2021款535T自动锐动版', '2019款535TCVT锐动版']:
            continue
        if current_series == '凯迪拉克ATS-L' and model in ['2017款28T技术型']:
            continue
        print('current series is', current_series, ", current model is", model)
        for current_attribute in attributes:
            if current_series == '亚洲龙' and model == '2019款2.5LTouring尊贵版国VI' and current_attribute in ['manipulate', 'power', 'space']:
                continue
            if current_series == '阿特兹' and model == '2020款2.5L蓝天运动版' and current_attribute in ['manipulate', 'power', 'space', 'energy', 'comfort']:
                continue
            if current_series == '马自达6' and model == '阿特兹' and current_attribute in ['space', 'power', 'manipulate', 'energy']:
                continue
            if current_series == '迈锐宝XL' and model == '2019款535TCVT锐联版' and current_attribute in ['space', 'power']:
                continue
            if current_series == '凯迪拉克ATS-L' and model == '2017款28T时尚型' and current_attribute in ['space', 'power']:
                continue
            print("current attribute is", current_attribute)
            texts = origin_data.loc[(origin_data["车系"] == current_series) & (origin_data['车型'] == model)][current_attribute].tolist()
            # print("texts = ", texts)
            component_result = {}
            for text in texts:
                text = str(text)
                if len(str(text).strip()) <= 1:
                    continue
                # print("result_api = ", result_api)
                result_api = calculate(text)
                if result_api is None or len(result_api) <= 1:
                    continue
                items = result_api.get('items')
                if items is None:
                    continue
                if len(items) == 0:
                    # print("extract nothing:", text)
                    continue
                # print("items = ", items)
                for current_dict in items:
                    # print("current_dict = ", current_dict)
                    component = current_dict.get('prop')
                    opinion = current_dict.get('adj', 'no_value')
                    if len(opinion) == 0:
                        opinion = 'no_value'
                    # print(component, opinion)
                    if component in component_result.keys():  # 字典中已有部件
                        if opinion in component_result.get(component).keys():  # 当前部件有此感知单元时
                            count = component_result[component][opinion]
                            component_result[component][opinion] = count + 1
                        else:  # 当前部件无此感知单元时
                            component_result[component][opinion] = 1
                    else:
                        component_result[component] = {opinion: 1}  # 字典中暂无部件时
            # print("component_result = ", component_result)
            # 将当前得到的component_result转为DataFrame
            data = {'series': [], 'model': [], 'attribute': [], 'component': [], 'opinion': [], 'count': []}
            for key, value in component_result.items():
                for k, v in value.items():
                    data['series'].append(current_series)
                    data['model'].append(model)
                    data['attribute'].append(current_attribute)
                    data['component'].append(key)
                    data['opinion'].append(k)
                    data['count'].append(v)
            result = pd.DataFrame(data)
            # 追加写入文件
            print("正在写入文件：", current_series, model, current_attribute)
            result.to_csv(result_path, mode='a', encoding='utf_8_sig', header=None, index=False)

print("end...")


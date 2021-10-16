import time, codecs, csv, numpy as np, sys, random
from keras.utils import to_categorical

from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from bert_serving.client import BertClient
bc = BertClient(port=5555, port_out=5556)
# print(bc.encode(['你好']))
import traceback

import pandas as pd
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

sys.setrecursionlimit(5000000)

debug = False
debugLength = 3
maxRowNumber = 20000
padding_length = 4 if debug else 512


def initData(ratio=0.7):
    path_pre = 'data/'
    file_names = ['big', 'mid', 'midbig', 'small', 'micro', 'jincou']
    attribute_names = ['空间评论', '动力评论', '操控评论', '能耗评论', '舒适性评论', '外观评论', '内饰评论', '性价比评论']
    attribute_score_names = ['空间评分', '动力评分', '操控评分', '能耗评分', '舒适性评分', '外观评分', '内饰评分', '性价比评分']
    col_names = ['空间评论', '动力评论', '操控评论', '能耗评论', '舒适性评论', '外观评论', '内饰评论', '性价比评论', '空间评分', '动力评分', '操控评分', '能耗评分', '舒适性评分', '外观评分', '内饰评分', '性价比评分']

    # 读取所有文件，合并数据
    all_data = pd.DataFrame(columns=col_names)
    for file_name in file_names:
        path = path_pre + file_name + '.csv'
        data = pd.read_csv(path, usecols=col_names, nrows=debugLength if debug else maxRowNumber)
        # print("current file is ", file_name, ", length = ", len(data))
        # print(data.head())
        all_data = all_data.append(data)

    # 将所有评论内容为空的属性标签改为0
    all_data.loc[all_data['空间评论'] == '0', ['空间评分']] = 0
    all_data.loc[all_data['动力评论'] == '0', ['动力评分']] = 0
    all_data.loc[all_data['操控评论'] == '0', ['操控评分']] = 0
    all_data.loc[all_data['能耗评论'] == '0', ['能耗评分']] = 0
    all_data.loc[all_data['舒适性评论'] == '0', ['舒适性评分']] = 0
    all_data.loc[all_data['外观评论'] == '0', ['外观评分']] = 0
    all_data.loc[all_data['内饰评论'] == '0', ['内饰评分']] = 0
    all_data.loc[all_data['性价比评论'] == '0', ['性价比评分']] = 0

    # 修改标签：：2→1,3→1,4→2,5→3，目前包括1、2、3
    for attribute_score_name in attribute_score_names:
        all_data.loc[all_data[attribute_score_name] == 2, [attribute_score_name]] = 1
        all_data.loc[all_data[attribute_score_name] == 3, [attribute_score_name]] = 1
        all_data.loc[all_data[attribute_score_name] == 4, [attribute_score_name]] = 2
        all_data.loc[all_data[attribute_score_name] == 5, [attribute_score_name]] = 3

    # 生成一个content
    all_data['content'] = all_data["空间评论"].map(str) + "。" + all_data["动力评论"].map(str) + "。" + all_data["操控评论"].map(str) + "。" + all_data["能耗评论"].map(str) + "。" + \
                          all_data["舒适性评论"].map(str) + "。" + all_data["外观评论"].map(str) + "。" + all_data["内饰评论"].map(str) + "。" + all_data["性价比评论"].map(str) + "。"

    # 删除所有属性均为空的行
    all_data = all_data.drop(index=all_data.loc[(all_data['content'] == '0。space。0。power。0。manipulation。0。consumption。0。comfort。0。outside。0。inside。0。value。')].index)

    # 对data进行shuffle
    all_data = all_data.sample(frac=1)

    # 索引重置
    all_data = all_data.reset_index(drop=True)

    # 删除无关列
    all_data = all_data.drop(attribute_names, axis=1)
    # print(all_data['content'])
    print(all_data.columns)
    print("all_data's length = ", len(all_data))

    # 训练集测试集数据划分
    length = len(all_data)
    train_length = int(length * ratio)
    data_train = all_data[: train_length]
    data_validation = all_data[train_length:]
    y_train = data_train[attribute_score_names]
    y_validation = data_validation[attribute_score_names]
    x_train = data_train['content']
    x_validation = data_validation['content']

    return x_train, y_train, x_validation, y_validation, attribute_score_names


def seq_padding(X, maxlen=padding_length, padding='0'):
    if len(X) < maxlen:
        return X.ljust(maxlen, padding)
    else:
        return X[:maxlen]


# 将text转为token
# 取长补短，查询Bert_service
def parseLineForBert(line):
    result = []
    # print("line1 = ", line)
    line = seq_padding(line)
    # print("line2 = ", line)
    line_list = []
    for l in line:
        if len(l) > 0:
            line_list.append(l)
        else:
            continue
    print("bc.encode() = ", bc.encode(line_list))
    # print("bc.encode()[0] = ", bc.encode(line_list)[0])
    result.append(bc.encode(line_list))
    return np.array(result)


# 批量生成训练数据
def generateSetForBert(X_value, Y_value, batch_size, tokenizer):
    length = len(Y_value)
    while True:
        cnt = 0  # 记录当前是否够一个batch
        X1 = []
        X2 = []
        Y = []
        i = 0  # 记录Y的遍历
        cnt_Y = 0
        for line in X_value:
            x1, x2 = parseLineForBert(str(line), tokenizer)
            X1.append(x1)
            X2.append(x2)
            i += 1
            cnt += 1
            if cnt == batch_size or i == length:
                Y = Y_value[int(cnt_Y): int(i)]
                cnt_Y += batch_size

                cnt = 0
                yield ([np.array(X1), np.array(X2)], to_categorical(Y, num_classes=4))
                X1 = []
                X2 = []
                Y = []


# 生成批量数据
def iter_minibatches(X_value, Y_value, batch_size):
    length = len(Y_value)
    cnt = 0  # 记录当前是否够一个batch
    X = []
    Y = []
    i = 0  # 记录Y的遍历
    cnt_Y = 0
    for line in X_value:
        x = bc.encode([str(line).strip()])
        # print('x.shape = ', x.shape)
        X.extend(x)
        i += 1
        cnt += 1
        if cnt == batch_size or i == length:
            Y = Y_value[int(cnt_Y): int(i)]
            cnt_Y += batch_size

            cnt = 0
            yield (np.array(X), Y)
            # yield (np.array(X), to_categorical(Y, num_classes=4))
            X = []
            Y = []


# 将X转为bert向量
def generateXBert(X_value):
    X = []
    for line in X_value:
        x = searchBert(line)
        # print("x = ", x)
        X.extend(x)
    # print("X = ", X)
    # print('X.shape = ', np.array(X).shape)
    return np.array(X)


# 查询bert词向量
def searchBert(line):
    try:
        x = bc.encode([str(line).strip()])
        num = random.randint(0, 9)
        # if num % 2 == 0:
        #     raise Exception('抛出异常哦')
    except Exception as e:
        traceback.print_exc()
        print("something is wrong when looking up bert embedding")
        searchBert(line)
    return x


# 批量产生X
def generateXSetForBert(X_value, y_length, batch_size, tokenizer):
    while True:
        # print("in generateXSetForBert...")
        cnt = 0
        X1 = []
        X2 = []
        i = 0
        for line in X_value:
            x1, x2 = parseLineForBert(str(line), tokenizer)
            X1.append(x1)
            X2.append(x2)
            i += 1
            cnt += 1
            if cnt == batch_size or i == y_length:
                cnt = 0
                yield ([np.array(X1), np.array(X2)])
                X1 = []
                X2 = []


# 训练模型
def trainModel(x_train, y_train, x_validation, y_validation, columns):
    F1_scores = 0
    print("Y_validation's length = ", len(y_validation))
    # 生成Bert词向量
    x_train = generateXBert(x_train)
    # print("x_train's length = ", len(x_train))
    # print("x_train[0]'s length = ", len(x_train[0]))

    # 生成验证集
    X_validation_ml = generateXBert(x_validation)

    for index, col in enumerate(columns):
        print("current col is:", col)
        y_train_col = y_train[col]
        y_train_col = list(y_train_col)
        y_validation_col = y_validation[col]
        y_validation_col = list(y_validation_col)

        # 训练机器学习模型
        # ml_names = ["bayes", "logicRegression"]
        ml_names = ["bayes", "ada", "svm", "randomForest", "decisionTree", "logicRegression"]
        # print("y_train_col = ", y_train_col)
        y_train_col = np.array(y_train_col)
        # print("y_train_col = ", y_train_col)
        # y_train_col = to_categorical(y_train_col, num_classes=4)
        # print("y_train_col = ", y_train_col)
        for ml in ml_names:
            # print("current ml model is ", ml)

            # 3.构建&训练模型
            # print("》》》开始构建&训练模型。。。")
            if ml.startswith("bayes"):
                current_model = GaussianNB()
            elif ml.startswith("ada"):
                current_model = AdaBoostClassifier(n_estimators=10)
            elif ml.startswith("svm"):
                current_model = Pipeline((("scaler", StandardScaler()), ("liner_svc", LinearSVC(C=1, loss="hinge")), ))
            elif ml.startswith("logic"):
                current_model = LogisticRegression(C=1e5)
            elif ml.startswith("decisionTree"):
                current_model = DecisionTreeClassifier(criterion="entropy")
            elif ml.startswith("random"):
                current_model = RandomForestClassifier()

            # 训练模型，一次性传入所有数据
            current_model.fit(x_train, y_train_col)

            '''
            for i, (X_train_ml, Y_train_ml) in enumerate(iter_minibatches(x_train, y_train_col, batch_size)):
                print("X_train_ml:")
                print(X_train_ml.shape)
                print(X_train_ml)
                print("Y_train_ml:")
                print(Y_train_ml)
                print("Y_train_ml's type = ", type(Y_train_ml))
                labels = [0, 1, 2, 3]
                current_model.partial_fit(X_train_ml, Y_train_ml, classes=labels)
            '''

            # 4.预测结果
            # print("》》》开始预测结果。。。")
            Y_predicts = current_model.predict(X_validation_ml)
            # Y_predicts = np.argmax(predicts, axis=1)
            # 5.计算各种评价指标&保存结果
            # print("Y_validation = ", Y_validation)
            # print("Y_predicts = ", Y_predicts)
            # print("Y_predicts' length = ", len(Y_predicts))
            saveResult(ml, y_validation_col, Y_predicts, col)
            '''
            '''

    print('all F1_score:', F1_scores / len(columns))


# 将预测结果保存至文件
def saveResult(ml, y_tokens_test, y_val_pred, col):
    # print("正在保存至文件...")
    # 计算各种评价指标&保存结果
    report = classification_report(y_tokens_test, y_val_pred, digits=4, output_dict=True)
    # print(report)
    accuracy = report.get("accuracy")
    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')
    weighted_avg = report.get("weighted avg")
    weighted_precision = weighted_avg.get("precision")
    weighted_recall = weighted_avg.get("recall")
    weighted_f1 = weighted_avg.get('f1-score')
    data = [ml, col, weighted_precision, weighted_recall, weighted_f1, macro_precision, macro_recall, macro_f1, accuracy]
    print(data)
    # 保存至文件
    path = 'result/ml-debug.csv' if debug else 'result/ml.csv'
    with codecs.open(path, "a", "utf_8_sig") as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in 2-sentiment analysis.py ...")

    print("》》》【1】正在读取文件", "。" * 80)
    x_train, y_train, x_validation, y_validation, columns = initData()

    print("》》》【3】设置模型参数", "。" * 80)
    epoch = 1 if debug else 3
    batch_size = 30
    batch_size_validation = 100
    times = 1
    print("training times = ", times)

    # print("》》》【4】构建模型", "。" * 100)

    print("》》》【5】训练模型", "。" * 80)
    trainModel(x_train, y_train, x_validation, y_validation, columns)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of absa_main.py...")


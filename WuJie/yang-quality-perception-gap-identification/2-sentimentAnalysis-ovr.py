import time, codecs, csv, math, numpy as np, copy
from keras.utils import to_categorical

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

from keras.layers import Input, Flatten, Dense, Dropout, GRU, Bidirectional, Conv1D, LSTM
from keras.layers import MaxPool1D
from keras import Model
from keras.optimizers import Adam

import tensorflow as tf

from keras_bert import Tokenizer, load_trained_model_from_checkpoint
import pandas as pd
pd.set_option('display.max_columns', None)

debug = False
debugLength = 10
maxRowNumber = 22000
sampling = True  # 使用OvR策略时是否下采样保证数据平衡
sampling_ratio = 0.5

# keras_bert
bert_path = "keras_bert/chinese_L-12_H-768_A-12"
bert_config_path = 'keras_bert/chinese_L-12_H-768_A-12/bert_config.json'
bert_checkpoint_path = 'keras_bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
bert_dict_path = 'keras_bert/chinese_L-12_H-768_A-12/vocab.txt'


def initData(ratio=0.8):
    path_pre = 'data/'
    file_names = ['口碑爬取_大型', '口碑爬取_中型', '口碑爬取_中大型', '口碑爬取_小型', '口碑爬取_微型', '口碑爬取_紧凑型']
    attribute_names = ['空间评论', '动力评论', '操控评论', '能耗评论', '舒适性评论', '外观评论', '内饰评论', '性价比评论']
    attribute_score_names = ['空间评分', '动力评分', '操控评分', '能耗评分', '舒适性评分', '外观评分', '内饰评分', '性价比评分']
    col_names = ['空间评论', '动力评论', '操控评论', '能耗评论', '舒适性评论', '外观评论', '内饰评论', '性价比评论', '空间评分', '动力评分', '操控评分', '能耗评分', '舒适性评分', '外观评分', '内饰评分', '性价比评分']

    # 读取所有文件，合并数据
    all_data = pd.DataFrame(columns=col_names)
    for file_name in file_names:
        path = path_pre + file_name + '.csv'
        data = pd.read_csv(path, usecols=col_names, nrows=debugLength if debug else maxRowNumber)
        print("current file is ", file_name, ", length = ", len(data))
        # print(data.head())
        # 删除存在属性为空的行
        print("data's length1 = ", len(data))
        data = data.drop(index=data.loc[(data['空间评论'] == '0')].index)
        data = data.drop(index=data.loc[(data['动力评论'] == '0')].index)
        data = data.drop(index=data.loc[(data['操控评论'] == '0')].index)
        data = data.drop(index=data.loc[(data['能耗评论'] == '0')].index)
        data = data.drop(index=data.loc[(data['舒适性评论'] == '0')].index)
        data = data.drop(index=data.loc[(data['外观评论'] == '0')].index)
        data = data.drop(index=data.loc[(data['内饰评论'] == '0')].index)
        data = data.drop(index=data.loc[(data['性价比评论'] == '0')].index)
        print("data's length2 = ", len(data))
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
    # 查看标签类别
    print('labels:', set(all_data['空间评分'].tolist()))

    # 生成一个content
    all_data['content'] = all_data["空间评论"].map(str) + "。space。" + all_data["动力评论"].map(str) + "。power。" + all_data["操控评论"].map(str) + "。manipulation。" + all_data["能耗评论"].map(str) + "。consumption。" + \
                          all_data["舒适性评论"].map(str) + "。comfort。" + all_data["外观评论"].map(str) + "。outside。" + all_data["内饰评论"].map(str) + "。inside。" + all_data["性价比评论"].map(str) + "。value。"

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

    # 生成OvR策略对应的三份数据
    length = len(all_data)
    train_length = int(length * ratio)
    data_train = all_data[: train_length]
    data_validation = all_data[train_length:]

    length1 = len(data_train)
    train_length1 = int(length1 * ratio)
    data_ovr_train = data_train[: train_length1]  # 包括X和Y
    data_ovr_validation = data_train[train_length1:]  # 包括X和Y
    # backup
    data_ovr_train_1 = copy.deepcopy(data_ovr_train)
    data_ovr_validation_1 = copy.deepcopy(data_ovr_validation)
    data_ovr_train_2 = copy.deepcopy(data_ovr_train)
    data_ovr_validation_2 = copy.deepcopy(data_ovr_validation)
    data_ovr_train_3 = copy.deepcopy(data_ovr_train)
    data_ovr_validation_3 = copy.deepcopy(data_ovr_validation)
    # print("test-before1:", data_ovr_train_1[attribute_score_names][:4])
    # print("test-before2:", data_ovr_train_2[attribute_score_names][:4])
    # print("test-before3:", data_ovr_train_3[attribute_score_names][:4])
    for attribute_score_name in attribute_score_names:
        # 生成第一份数据，标签非1置为0，标签为0和1（1为目标）
        data_ovr_train_1.loc[data_ovr_train_1[attribute_score_name] != 1, [attribute_score_name]] = 0
        data_ovr_validation_1.loc[data_ovr_validation_1[attribute_score_name] != 1, [attribute_score_name]] = 0
        # 生成第二份数据，标签非2置为0，标签为2置为1，标签为0和1（1为目标）
        data_ovr_train_2.loc[data_ovr_train_2[attribute_score_name] != 2, [attribute_score_name]] = 0
        data_ovr_validation_2.loc[data_ovr_validation_2[attribute_score_name] != 2, [attribute_score_name]] = 0
        data_ovr_train_2.loc[data_ovr_train_2[attribute_score_name] == 2, [attribute_score_name]] = 1
        data_ovr_validation_2.loc[data_ovr_validation_2[attribute_score_name] == 2, [attribute_score_name]] = 1
        # 生成第三份数据，标签非3置为0，标签为3置为1，标签为0和1（1为目标）
        data_ovr_train_3.loc[data_ovr_train_3[attribute_score_name] != 3, [attribute_score_name]] = 0
        data_ovr_validation_3.loc[data_ovr_validation_3[attribute_score_name] != 3, [attribute_score_name]] = 0
        data_ovr_train_3.loc[data_ovr_train_3[attribute_score_name] == 3, [attribute_score_name]] = 1
        data_ovr_validation_3.loc[data_ovr_validation_3[attribute_score_name] == 3, [attribute_score_name]] = 1
    # print("test-after1:", data_ovr_train_1[attribute_score_names][:4])
    # print("test-after2:", data_ovr_train_2[attribute_score_names][:4])
    # print("test-after3:", data_ovr_train_3[attribute_score_names][:4])

    # 统计三份数据的比例分布

    # 生成三份数据训练数据
    # 数据1
    x_ovr_train_1 = data_ovr_train_1['content']
    y_ovr_train_1 = data_ovr_train_1[attribute_score_names]
    x_ovr_validation_1 = data_ovr_validation_1['content']
    y_ovr_validation_1 = data_ovr_validation_1[attribute_score_names]
    # 数据2
    x_ovr_train_2 = data_ovr_train_2['content']
    y_ovr_train_2 = data_ovr_train_2[attribute_score_names]
    x_ovr_validation_2 = data_ovr_validation_2['content']
    y_ovr_validation_2 = data_ovr_validation_2[attribute_score_names]
    # 数据3
    x_ovr_train_3 = data_ovr_train_3['content']
    y_ovr_train_3 = data_ovr_train_3[attribute_score_names]
    x_ovr_validation_3 = data_ovr_validation_3['content']
    y_ovr_validation_3 = data_ovr_validation_3[attribute_score_names]
    # 最后的验证数据
    x_validation = data_validation['content']
    y_validation = data_validation[attribute_score_names]
    # print("y_validation = ", y_validation)

    return x_ovr_train_1, y_ovr_train_1, x_ovr_validation_1, y_ovr_validation_1, \
           x_ovr_train_2, y_ovr_train_2, x_ovr_validation_2, y_ovr_validation_2, \
           x_ovr_train_3, y_ovr_train_3, x_ovr_validation_3, y_ovr_validation_3, \
           x_validation, y_validation, attribute_score_names


# 加载tokenizer
def get_tokenizer():
    token_dict = {}
    with codecs.open(bert_dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)
    return tokenizer


# 构建CNN模型
def createCNNModel(cnn_filter, window_size):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])
        cnn = Conv1D(cnn_filter, window_size, name='conv')(x)
        cnn = MaxPool1D(name='max_pool')(cnn)
        flatten = Flatten()(cnn)
        x = Dense(32, activation='relu', name='dense_1')(flatten)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(2, activation='softmax', name='softmax')(x)
        model = Model([x1_in, x2_in], p)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['mae'])
    model.summary()
    return model


# 构建LSTM模型
def createLSTMModel(dim):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        lstm = LSTM(dim, return_sequences=False, name='lstm1')(x)

        x = Dense(64, activation='relu', name='dense_1')(lstm)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(2, activation='softmax', name='softmax')(x)

        model = Model(inputs=[x1_in, x2_in], outputs=p)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['mae'])
    print(model.summary())

    return model


# 将text转为token
def parseLineForBert(line, tokenizer):
    indices, segments = tokenizer.encode(first=line, max_len=512)
    return np.array(indices), np.array(segments)


# 批量生成训练数据
def generateSetForBert(X_value, Y_value, batch_size, tokenizer):
    '''
    print("X_value = ", X_value)
    print("Y_value = ", Y_value)
    print("batch_size = ", batch_size)
    print("X's length = ", len(X_value))
    print("Y's length = ", len(Y_value))
    '''
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
                yield ([np.array(X1), np.array(X2)], to_categorical(Y, num_classes=2))
                X1 = []
                X2 = []
                Y = []


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


# 下采样函数,data为属性列，DataFrame类型
def lower_sampling(x_data, data, col):
    data_0 = data[data == 0]
    data_1 = data[data == 1]

    # 获取label为0和1的下标
    indice_0 = data[data == 0].index.tolist()
    indice_1 = data[data == 1].index.tolist()
    x_data_0 = x_data[indice_0]
    x_data_1 = x_data[indice_1]
    '''
    print("pre data_0's length = ", len(data_0))
    print("pre data_1's length = ", len(data_1))
    print("pre x_data_0's length = ", len(x_data_0))
    print("pre x_data_1's length = ", len(x_data_1))
    '''

    length_0 = len(data_0)
    length_1 = len(data_1)
    length_min = min(length_0, length_1)
    print("min_length = ", length_min)

    if (length_0 * sampling_ratio) > length_min:
        index = np.random.randint(length_0, size=int(min(length_0 * sampling_ratio, length_min / sampling_ratio)))
        print("index's length = ", len(index))
        print("data_0.shape = ", data_0.shape)
        data_0 = data_0.iloc[list(index)]
        print("data_0.shape = ", data_0.shape)
        x_data_0 = x_data_0.iloc[list(index)]
    if (length_1 * sampling_ratio) > length_min:
        index = np.random.randint(length_1, size=int(min(length_1 * sampling_ratio, length_min / sampling_ratio)))
        print("index's length = ", len(index))
        print("data_1.shape = ", data_1.shape)
        data_1 = data_1.iloc[list(index)]
        print("data_1.shape = ", data_1.shape)
        x_data_1 = x_data_1.iloc[list(index)]
    '''
    print("data_0's length = ", len(data_0))
    print("data_1's length = ", len(data_1))
    print("x_data_0's length = ", len(x_data_0))
    print("x_data_1's length = ", len(x_data_1))
    '''
    final_data = pd.DataFrame()
    final_data[col] = pd.concat([data_0, data_1])
    final_data['content'] = pd.concat([x_data_0, x_data_1])
    final_data = final_data.sample(frac=1)
    print("final_data.shape = ", final_data.shape)

    return final_data[col], final_data['content']


# 训练模型
def trainModel(model1, model2, model3, tokenizer, epoch, batch_size, batch_size_validation, columns,
               x_ovr_train_1, y_ovr_train_1, x_ovr_validation_1, y_ovr_validation_1,
               x_ovr_train_2, y_ovr_train_2, x_ovr_validation_2, y_ovr_validation_2,
               x_ovr_train_3, y_ovr_train_3, x_ovr_validation_3, y_ovr_validation_3,
               x_validation, y_validation):
    length_ovr_train_1 = len(y_ovr_train_1)
    length_ovr_validation_1 = len(y_ovr_validation_1)
    length_ovr_train_2 = len(y_ovr_train_2)
    length_ovr_validation_2 = len(y_ovr_validation_2)
    length_ovr_train_3 = len(y_ovr_train_3)
    length_ovr_validation_3 = len(y_ovr_validation_3)
    length_validation = len(y_validation)
    F1_scores = 0
    for index, col in enumerate(columns):
        print("current col is:", col)
        y_validation_col = list(y_validation[col])
        y_ovr_train_col_1 = y_ovr_train_1[col]
        y_ovr_train_col_2 = y_ovr_train_2[col]
        y_ovr_train_col_3 = y_ovr_train_3[col]
        # 在此处进行下采样
        if sampling:
            y_ovr_train_col_1, x_ovr_train_col_1 = lower_sampling(x_ovr_train_1, y_ovr_train_col_1, col)
            y_ovr_train_col_2, x_ovr_train_col_2 = lower_sampling(x_ovr_train_2, y_ovr_train_col_2, col)
            y_ovr_train_col_3, x_ovr_train_col_3 = lower_sampling(x_ovr_train_3, y_ovr_train_col_3, col)

        y_ovr_validation_col_1 = list(y_ovr_validation_1[col])
        # print("y_ovr_validation_col_1 = ", y_ovr_validation_col_1)
        y_ovr_validation_col_2 = list(y_ovr_validation_2[col])
        # print("y_ovr_validation_col_2 = ", y_ovr_validation_col_2)
        y_ovr_validation_col_3 = list(y_ovr_validation_3[col])

        # 分别训练三个模型
        model1.fit(generateSetForBert(x_ovr_train_col_1, y_ovr_train_col_1, batch_size, tokenizer), steps_per_epoch=math.ceil(length_ovr_train_1 / (batch_size)),
                  epochs=epoch, batch_size=batch_size, verbose=1, validation_steps=math.ceil(length_ovr_validation_1 / (batch_size_validation)),
                  validation_data=generateSetForBert(x_ovr_validation_1, y_ovr_validation_col_1, batch_size_validation, tokenizer))
        model2.fit(generateSetForBert(x_ovr_train_col_2, y_ovr_train_col_2, batch_size, tokenizer), steps_per_epoch=math.ceil(length_ovr_train_2 / (batch_size)),
                   epochs=epoch, batch_size=batch_size, verbose=1, validation_steps=math.ceil(length_ovr_validation_2 / (batch_size_validation)),
                   validation_data=generateSetForBert(x_ovr_validation_2, y_ovr_validation_col_2, batch_size_validation, tokenizer))
        model3.fit(generateSetForBert(x_ovr_train_col_3, y_ovr_train_col_3, batch_size, tokenizer), steps_per_epoch=math.ceil(length_ovr_train_3 / (batch_size)),
                   epochs=epoch, batch_size=batch_size, verbose=1, validation_steps=math.ceil(length_ovr_validation_3 / (batch_size_validation)),
                   validation_data=generateSetForBert(x_ovr_validation_3, y_ovr_validation_col_3, batch_size_validation, tokenizer))

        # 预测验证集
        y_val_pred_1 = model1.predict(generateXSetForBert(x_validation, length_validation, batch_size_validation, tokenizer), steps=math.ceil(length_validation / (batch_size_validation)))
        # print("y_val_pred_1 = ", y_val_pred_1)
        y_val_pred_2 = model2.predict(generateXSetForBert(x_validation, length_validation, batch_size_validation, tokenizer), steps=math.ceil(length_validation / (batch_size_validation)))
        # print("y_val_pred_2 = ", y_val_pred_2)
        y_val_pred_3 = model3.predict(generateXSetForBert(x_validation, length_validation, batch_size_validation, tokenizer), steps=math.ceil(length_validation / (batch_size_validation)))
        # print("y_val_pred_3 = ", y_val_pred_3)

        # 统计三个模型预测结果
        y_val_pred_1 = y_val_pred_1[:, 1]
        # print("y_val_pred_1 = ", y_val_pred_1)
        y_val_pred_2 = y_val_pred_2[:, 1]
        # print("y_val_pred_2 = ", y_val_pred_2)
        y_val_pred_3 = y_val_pred_3[:, 1]
        # print("y_val_pred_3 = ", y_val_pred_3)
        # 取出三个模型预测结果概率最大的
        y_val_pred = []
        for i in range(len(y_val_pred_1)):
            max_value = max(y_val_pred_1[i], y_val_pred_2[i], y_val_pred_3[i])
            # print(max_value, y_val_pred_1[i], y_val_pred_2[i], y_val_pred_3[i])
            y_val_pred.append(1 if max_value == y_val_pred_1[i] else 2 if max_value == y_val_pred_2[i] else 3)
        # print("y_val_pred = ", y_val_pred)

        # 准确率：在所有预测为正的样本中，确实为正的比例
        # 召回率：本身为正的样本中，被预测为正的比例
        print("y_val[20] = ", list(y_validation_col)[:20])
        print("y_val_pred[20] = ", list(y_val_pred)[:20])
        precision, recall, fscore, support = score(y_validation_col, y_val_pred)
        print("precision = ", precision)
        print("recall = ", recall)
        print("fscore = ", fscore)
        print("support = ", support)

        report = classification_report(y_validation_col, y_val_pred, digits=4, output_dict=True)
        print(report)

        F1_score = f1_score(y_validation_col, y_val_pred, average='macro')
        F1_scores += F1_score
        print('第', index, '个属性', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_validation_col, y_val_pred))
        print("%Y-%m%d %H:%M:%S", time.localtime())

        # 保存当前属性的结果,整体的结果根据所有属性的结果来计算
        print("正在保存结果。。。")
        save_result_to_csv(report, F1_score, col)

    print('all F1_score:', F1_scores / len(columns))


# 把结果保存到csv
# report是classification_report生成的字典结果
# def save_result_to_csv(report, f1_score, experiment_id, model_name, col_name, debug):
def save_result_to_csv(report, f1_score, col_name):
    accuracy = report.get("accuracy")

    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')

    weighted_avg = report.get("weighted avg")
    weighted_precision = weighted_avg.get("precision")
    weighted_recall = weighted_avg.get("recall")
    weighted_f1 = weighted_avg.get('f1-score')
    data = [col_name, weighted_precision, weighted_recall, weighted_f1, macro_precision, macro_recall, macro_f1, f1_score, accuracy]

    if debug:
        path = "result/ovr_auto_absa_debug.csv"
    else:
        path = "result/ovr_auto_absa.csv"
    with codecs.open(path, "a", encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in 2-sentiment analysis.py ...")

    print("》》》【1】正在读取文件", "。" * 100)
    x_ovr_train_1, y_ovr_train_1, x_ovr_validation_1, y_ovr_validation_1, \
    x_ovr_train_2, y_ovr_train_2, x_ovr_validation_2, y_ovr_validation_2, \
    x_ovr_train_3, y_ovr_train_3, x_ovr_validation_3, y_ovr_validation_3, \
    x_validation, y_validation, columns = initData()

    print("》》》【2】正在加载tokenizer", "。" * 100)
    tokenizer = get_tokenizer()
    print("》》》【3】设置模型参数", "。" * 100)
    epoch = 1 if debug else 3
    batch_size = 32
    batch_size_validation = 100
    times = 1
    print("training times = ", times)

    print("》》》【4】构建模型", "。" * 100)
    model1 = createLSTMModel(64)
    model2 = createLSTMModel(64)
    model3 = createLSTMModel(64)
    # model1 = createCNNModel(64, 4)
    # model2 = createCNNModel(64, 4)
    # model3 = createCNNModel(64, 4)
    print("model is LSTM")

    print("》》》【5】训练模型", "。" * 100)
    trainModel(model1, model2, model3, tokenizer, epoch, batch_size, batch_size_validation, columns,
               x_ovr_train_1, y_ovr_train_1, x_ovr_validation_1, y_ovr_validation_1,
               x_ovr_train_2, y_ovr_train_2, x_ovr_validation_2, y_ovr_validation_2,
               x_ovr_train_3, y_ovr_train_3, x_ovr_validation_3, y_ovr_validation_3,
               x_validation, y_validation)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of absa_main.py...")


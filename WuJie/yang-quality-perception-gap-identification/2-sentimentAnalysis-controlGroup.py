import time, codecs, csv, math, numpy as np
from keras.utils import to_categorical

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

from keras.layers import Input, Flatten, Dense, Dropout, GRU, Bidirectional, Conv1D, LSTM, GlobalAveragePooling1D
from keras.layers import MaxPool1D
from keras import Model
from keras.optimizers import Adam

import tensorflow as tf

from keras_bert import Tokenizer, load_trained_model_from_checkpoint
import pandas as pd
pd.set_option('display.max_columns', None)

debug = False
debugLength = 5
maxRowNumber = 22744

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


# 加载tokenizer
def get_tokenizer():
    token_dict = {}
    with codecs.open(bert_dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)
    return tokenizer


# 构建MLP模型
def createMLPModel():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])
        x_contents = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x_contents)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)
        model = Model([x1_in, x2_in], p)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['mae'])
    model.summary()
    return model


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
        p = Dense(4, activation='softmax', name='softmax')(x)
        model = Model([x1_in, x2_in], p)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['mae'])
    model.summary()
    return model


# 构建GRU模型
def createGRUModel(dim):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        bi_gru = GRU(dim, name="gru_1")(x)

        x = Dense(64, activation='relu', name='dense_1')(bi_gru)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)

        model = Model(inputs=[x1_in, x2_in], outputs=p)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['mae'])
    print(model.summary())

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
        p = Dense(4, activation='softmax', name='softmax')(x)

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
def trainModel(model, model_name, x_train, y_train, x_validation, y_validation, columns, tokenizer, epoch, batch_size, batch_size_validation):
    length = len(y_train)
    length_validation = len(y_validation)
    F1_scores = 0
    for index, col in enumerate(columns):
        print("current col is:", col)
        y_train_col = y_train[col]
        y_train_col = list(y_train_col)
        y_validation_col = y_validation[col]
        y_validation_col = list(y_validation_col)
        history = model.fit(generateSetForBert(x_train, y_train_col, batch_size, tokenizer), steps_per_epoch=math.ceil(length / (batch_size)),
                            epochs=epoch, batch_size=batch_size, verbose=1, validation_steps=math.ceil(length_validation / (batch_size_validation)),
                            validation_data=generateSetForBert(x_validation, y_validation_col, batch_size_validation, tokenizer))

        # 预测验证集
        y_val_pred = model.predict(generateXSetForBert(x_validation, length_validation, batch_size_validation, tokenizer), steps=math.ceil(length_validation / (batch_size_validation)))
        y_val_pred = np.argmax(y_val_pred, axis=1)

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
        save_result_to_csv(model_name, report, F1_score, col)
        # save_result_to_csv(report, F1_score, experiment_name, model_name, col, debug)

    print('all F1_score:', F1_scores / len(columns))


# 把结果保存到csv
# report是classification_report生成的字典结果
# def save_result_to_csv(report, f1_score, experiment_id, model_name, col_name, debug):
def save_result_to_csv(model_name, report, f1_score, col_name):
    accuracy = report.get("accuracy")

    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')

    weighted_avg = report.get("weighted avg")
    weighted_precision = weighted_avg.get("precision")
    weighted_recall = weighted_avg.get("recall")
    weighted_f1 = weighted_avg.get('f1-score')
    data = [model_name, col_name, weighted_precision, weighted_recall, weighted_f1, macro_precision, macro_recall, macro_f1, f1_score, accuracy]

    if debug:
        path = "result/auto_absa_debug.csv"
    else:
        path = "result/auto_absa.csv"
    with codecs.open(path, "a", encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in 2-sentiment analysis.py ...")

    print("》》》【1】正在读取文件", "。" * 100)
    x_train, y_train, x_validation, y_validation, columns = initData()

    print("》》》【2】正在加载tokenizer", "。" * 100)
    tokenizer = get_tokenizer()

    print("》》》【3】设置模型参数", "。" * 100)
    epoch = 1 if debug else 3
    batch_size = 30
    batch_size_validation = 100
    times = 1
    print("training times = ", times)

    print("》》》【4】构建模型", "。" * 100)
    model_name = 'mlp'
    # model = createCNNModel(64, 4)
    # model = createLSTMModel(64)
    # model = createLSTMModel(64)
    model = createMLPModel()

    print("》》》【5】训练模型", "。" * 100)
    trainModel(model, model_name, x_train, y_train, x_validation, y_validation, columns, tokenizer, epoch, batch_size, batch_size_validation)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of absa_main.py...")


import time, codecs, csv, math, numpy as np, random, datetime, os, gc
from keras.utils import to_categorical

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from keras.layers import Input, Flatten, Dense, Dropout, GRU, Bidirectional, Conv1D, LSTM, GlobalAveragePooling1D
from keras.layers import MaxPool1D
from keras import Model, backend as K
from keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from keras_bert import Tokenizer, load_trained_model_from_checkpoint
import pandas as pd
pd.set_option('display.max_columns', None)

debug = False
debugLength = 400

# keras_bert
bert_path = "../keras_bert/chinese_L-12_H-768_A-12"
bert_config_path = '../keras_bert/chinese_L-12_H-768_A-12/bert_config.json'
bert_checkpoint_path = '../keras_bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
bert_dict_path = '../keras_bert/chinese_L-12_H-768_A-12/vocab.txt'


# 加载tokenizer
def get_tokenizer():
    token_dict = {}
    with codecs.open(bert_dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)
    return tokenizer


# 初始化数据，不拆分X和Y
def initData2():
    path_training = "data/sentiment_analysis_training_set.csv"
    path_validation = "data/sentiment_analysis_validation_set.csv"
    path_test = "data/merged_file.xlsx"

    # 加载所有列
    data_training = pd.read_csv(path_training, nrows=debugLength if debug else None)
    data_validation = pd.read_csv(path_validation, nrows=debugLength if debug else None)
    data_test = pd.read_excel(path_test, nrows=debugLength if debug else None)
    # print("before:", data_training.columns)
    # 删除id和dish_recommendation和others_overall_experience三列
    drop_columns = ["id"]
    data_training = data_training.drop(drop_columns, axis=1)
    data_validation = data_validation.drop(drop_columns, axis=1)

    return data_training, data_validation, data_test


# 下采样函数，输入col和data，返回X和Y
def sampling(data, col):
    # 进行下采样
    indice_2 = data[(data[col] == -2)].index.tolist()
    indice_others = data[(data[col] != -2)].index.tolist()
    length_others = len(indice_others)
    print("indice_others' length = ", length_others)
    indice_random = random.sample(indice_2, min(int(length_others / 3), len(indice_2)))
    print("indice_random's length = ", len(indice_random))
    indice = indice_others + indice_random
    random.shuffle(indice)
    '''
    if debug:
        print("indice_2 = ", indice_2)
        print("indice_others = ", indice_others)
        print("indice_random = ", indice_random)
        print("indice = ", indice)
    '''

    x = data['content'][indice]
    y = data[col][indice]

    return x, y


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


# 构建MLP模型
def createModel():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        x = GlobalAveragePooling1D()(x)

        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)

        model = Model(inputs=[x1_in, x2_in], outputs=p)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['mae'])
    print(model.summary())

    return model


# 构建MLP模型
def createMLPModel():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        x = GlobalAveragePooling1D()(x)

        x = Dense(128, activation='relu', name='Dense_target')(x)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)

        model = Model(inputs=[x1_in, x2_in], outputs=p)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['mae'])
    print(model.summary())

    return model


# 目的：多模型对比
# 采用交叉验证方式，合并训练集和验证集，模型不需要保存
# 输入为data_training和data_validation
cols_done_dictionary_path = 'cols_done_file-debug.npy' if debug else 'cols_done_file.npy'
def get_keys(d, value):
    return [k for k, v in d.items() if v == value]
# data_test为最终需要预测的评论文本
def trainModel(dl_model_name, data_training, data_validation, data_test, tokenizer, epoch, batch_size, batch_size_validation):
    # 获取当天日期
    now_time = datetime.datetime.now()
    day = datetime.datetime.strftime(now_time, '%Y%m%d')
    # 合并训练集和验证集
    data = data_training.append(data_validation)
    data = data.sample(frac=1)  # shuffle
    data = data.reset_index(drop=True)  # 索引重置

    # 生成X和Y
    columns = data_training.columns.tolist()
    columns.remove('content')
    print("columns = ", columns)
    print("columns' length = ", len(columns))
    F1_scores = 0
    # 查询已完成属性,json文件。加载文件，如果文件不存在或者无内容，则返回/创建一个空字典
    for index, col in enumerate(columns):
        if os.path.exists(cols_done_dictionary_path):
            cols_done_dictionary = np.load(cols_done_dictionary_path, allow_pickle=True).item()
            cols_done_1 = get_keys(cols_done_dictionary.get(dl_model_name), 1)  # 完成了一次的属性
        else:
            cols_done_dictionary = {}
            cols_done_1 = []
        if col in cols_done_1:
            continue
        print("cols_done_dictionary = ", cols_done_dictionary)
        print("cols_done_1 = ", cols_done_1)

        print("current col is:", col)
        x, y = sampling(data, col)  # 喂给模型，x只包含content
        x = x.tolist()
        print("data length is", len(x))

        # 标签+2
        y = y + 2
        y_col = list(y)

        # 不要交叉验证，训练完模型后直接预测
        if dl_model_name == "mlp":
            model = createMLPModel()
        elif dl_model_name == "bert":
            model = createModel()

        # 跑两次交叉验证，保证最终的数量为10，然后进行T检验
        # 2.交叉验证数据集
        kf = KFold(n_splits=5)
        current_k = 0
        rows = []
        for train_index, validation_index in kf.split(x):
            if current_k > 0:
                continue
            print("正在进行第", current_k, "轮交叉验证。。。")
            current_k += 1
            x_train, Y_train = np.array(x)[train_index], np.array(y_col)[train_index]
            x_validation, Y_validation = np.array(x)[validation_index], np.array(y_col)[validation_index]

            length = len(x_train)
            length_validation = len(x_validation)

            # 早停法，如果val_acc没有提高0.0001就停止
            earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=2)
            model.fit(generateSetForBert(x_train, Y_train, batch_size, tokenizer), steps_per_epoch=math.ceil(length / batch_size),
                      epochs=epoch, batch_size=batch_size, verbose=1, validation_steps=math.ceil(length_validation / batch_size_validation),
                      validation_data=generateSetForBert(x_validation, Y_validation, batch_size_validation, tokenizer), callbacks=[earlystop_callback])

            # 预测验证集
            y_val_pred = model.predict(generateXSetForBert(x_validation, length_validation, batch_size_validation, tokenizer), steps=math.ceil(length_validation / batch_size_validation))
            y_val_pred = np.argmax(y_val_pred, axis=1)

            # 准确率：在所有预测为正的样本中，确实为正的比例
            # 召回率：本身为正的样本中，被预测为正的比例
            print("y_val[20] = ", list(Y_validation)[:20])
            print("y_val_pred[20] = ", list(y_val_pred)[:20])

            # 计算MSE MAE R2 report
            mse = mean_squared_error(Y_validation, y_val_pred)
            mae = mean_absolute_error(Y_validation, y_val_pred)
            r2 = r2_score(Y_validation, y_val_pred)
            report = classification_report(Y_validation, y_val_pred, digits=4, output_dict=True)
            # print(report)

            F1_score = f1_score(Y_validation, y_val_pred, average='macro')
            F1_scores += F1_score
            print('第', index, '个属性', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(Y_validation, y_val_pred))
            print(datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m%d %H:%M:%S"))
            # print("%Y-%m%d %H:%M:%S", time.localtime())

            # 计算各种指标
            accuracy = report.get("accuracy")
            macro_avg = report.get("macro avg")
            macro_precision = macro_avg.get("precision")
            macro_recall = macro_avg.get("recall")
            macro_f1 = macro_avg.get('f1-score')

            row = [dl_model_name, col, macro_precision, macro_recall, macro_f1, accuracy, mse, mae, r2]
            print("row:", row)
            rows.append(row)

            # 预测测试集
            length_test = len(data_test)
            # 去除回车符
            data_test["content"] = data_test["content"].apply(lambda x: str(x).replace('\n', '').replace('\r', ''))
            y_test_pred = model.predict(generateXSetForBert(data_test["content"], length_test, batch_size_validation, tokenizer), steps=math.ceil(length_test / batch_size_validation))
            y_test_pred = np.argmax(y_test_pred, axis=1)
            # 保存测试集预测结果
            data_test[col] = y_test_pred
            data_test.to_csv("result/data_test_predict-debug.csv" if debug else "result/data_test_predict_v2.csv", index=False, mode="w", encoding='utf_8_sig')

            # 清理内存
            K.clear_session()
            del model
            gc.collect()

        # 保存当前属性的结果,整体的结果根据所有属性的结果来计算
        print("正在保存结果。。。")
        save_result_to_csv_cv(rows, day)

        # 保存已经跑完的属性及其频率
        # 格式为{model_name1:{col1:counter,col2:counter},model_name2:{col1:counter,col2:counter,col3:counter}}
        if dl_model_name not in cols_done_dictionary.keys():  # 如果模型不存在的话
            cols_done_dictionary[dl_model_name] = {col: 1}
            print("模型不存在，设置,", col, "为1")
        else:
            if col not in cols_done_dictionary.get(dl_model_name).keys():  # 如果模型存在，但是当前col不存在
                print("模型存在，但是,", col, "不存在，设置为1")
                cols_done_dictionary[dl_model_name][col] = 1
            else:  # 如果模型存在&当前col也存在，则counter设置为2
                print("模型存在&当前,", col, "存在，设置为2")
                cols_done_dictionary[dl_model_name][col] = 2
        np.save(cols_done_dictionary_path, cols_done_dictionary)

    print('all F1_score:', F1_scores / len(columns))


# data_test为最终需要预测的评论文本,保留正负向情感的标签&概率值
def trainModel2(dl_model_name, data_training, data_validation, data_test, tokenizer, epoch, batch_size, batch_size_validation):
    # 获取当天日期
    now_time = datetime.datetime.now()
    day = datetime.datetime.strftime(now_time, '%Y%m%d')
    # 合并训练集和验证集
    data = data_training.append(data_validation)
    data = data.sample(frac=1)  # shuffle
    data = data.reset_index(drop=True)  # 索引重置

    # 生成X和Y
    columns = data_training.columns.tolist()
    columns.remove('content')
    print("columns = ", columns)
    print("columns' length = ", len(columns))
    F1_scores = 0
    # 查询已完成属性,json文件。加载文件，如果文件不存在或者无内容，则返回/创建一个空字典
    for index, col in enumerate(columns):
        if os.path.exists(cols_done_dictionary_path):
            cols_done_dictionary = np.load(cols_done_dictionary_path, allow_pickle=True).item()
            cols_done_1 = get_keys(cols_done_dictionary.get(dl_model_name), 1)  # 完成了一次的属性
        else:
            cols_done_dictionary = {}
            cols_done_1 = []
        if col in cols_done_1:
            continue
        print("cols_done_dictionary = ", cols_done_dictionary)
        print("cols_done_1 = ", cols_done_1)

        print("current col is:", col)
        x, y = sampling(data, col)  # 喂给模型，x只包含content
        x = x.tolist()
        print("data length is", len(x))

        # 标签+2
        y = y + 2
        y_col = list(y)

        # 不要交叉验证，训练完模型后直接预测
        if dl_model_name == "mlp":
            model = createMLPModel()
        elif dl_model_name == "bert":
            model = createModel()

        # 2.交叉验证数据集
        kf = KFold(n_splits=5)
        current_k = 0
        rows = []
        for train_index, validation_index in kf.split(x):
            if current_k > 0:
                continue
            print("正在进行第", current_k, "轮交叉验证。。。")
            current_k += 1
            x_train, Y_train = np.array(x)[train_index], np.array(y_col)[train_index]
            x_validation, Y_validation = np.array(x)[validation_index], np.array(y_col)[validation_index]

            length = len(x_train)
            length_validation = len(x_validation)

            # 早停法，如果val_acc没有提高0.0001就停止
            earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=2)
            model.fit(generateSetForBert(x_train, Y_train, batch_size, tokenizer), steps_per_epoch=math.ceil(length / batch_size),
                      epochs=epoch, batch_size=batch_size, verbose=1, validation_steps=math.ceil(length_validation / batch_size_validation),
                      validation_data=generateSetForBert(x_validation, Y_validation, batch_size_validation, tokenizer), callbacks=[earlystop_callback])

            # 预测验证集
            y_val_pred = model.predict(generateXSetForBert(x_validation, length_validation, batch_size_validation, tokenizer), steps=math.ceil(length_validation / batch_size_validation))
            y_val_pred = np.argmax(y_val_pred, axis=1)

            # 准确率：在所有预测为正的样本中，确实为正的比例
            # 召回率：本身为正的样本中，被预测为正的比例
            print("y_val[20] = ", list(Y_validation)[:20])
            print("y_val_pred[20] = ", list(y_val_pred)[:20])

            # 计算MSE MAE R2 report
            mse = mean_squared_error(Y_validation, y_val_pred)
            mae = mean_absolute_error(Y_validation, y_val_pred)
            r2 = r2_score(Y_validation, y_val_pred)
            report = classification_report(Y_validation, y_val_pred, digits=4, output_dict=True)
            # print(report)

            F1_score = f1_score(Y_validation, y_val_pred, average='macro')
            F1_scores += F1_score
            print('第', index, '个属性', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(Y_validation, y_val_pred))
            print(datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m%d %H:%M:%S"))
            # print("%Y-%m%d %H:%M:%S", time.localtime())

            # 计算各种指标
            accuracy = report.get("accuracy")
            macro_avg = report.get("macro avg")
            macro_precision = macro_avg.get("precision")
            macro_recall = macro_avg.get("recall")
            macro_f1 = macro_avg.get('f1-score')

            row = [dl_model_name, col, macro_precision, macro_recall, macro_f1, accuracy, mse, mae, r2]
            print("row:", row)
            rows.append(row)

            # 预测测试集
            length_test = len(data_test)
            # 去除回车符
            data_test["content"] = data_test["content"].apply(lambda x: str(x).replace('\n', '').replace('\r', ''))
            y_test_pred = model.predict(generateXSetForBert(data_test["content"], length_test, batch_size_validation, tokenizer), steps=math.ceil(length_test / batch_size_validation))
            y_test_pred_index = np.argmax(y_test_pred, axis=1)  # 获取最大值对应索引
            y_test_pred_prob = np.max(y_test_pred, axis=1)  # 获取最大值
            # 保存测试集预测结果
            data_test[col] = y_test_pred_index
            data_test[col + "-prob"] = y_test_pred_prob
            data_test.to_csv("result/" + str(day) + "-data_test_predict-debug.csv" if debug else "result/" + str(day) + "-data_test_predict_v2.csv", index=False, mode="w", encoding='utf_8_sig')

            # 清理内存
            K.clear_session()
            del model
            gc.collect()

        # 保存当前属性的结果,整体的结果根据所有属性的结果来计算
        print("正在保存结果。。。")
        save_result_to_csv_cv(rows, day)

        # 保存已经跑完的属性及其频率
        # 格式为{model_name1:{col1:counter,col2:counter},model_name2:{col1:counter,col2:counter,col3:counter}}
        if dl_model_name not in cols_done_dictionary.keys():  # 如果模型不存在的话
            cols_done_dictionary[dl_model_name] = {col: 1}
            print("模型不存在，设置,", col, "为1")
        else:
            if col not in cols_done_dictionary.get(dl_model_name).keys():  # 如果模型存在，但是当前col不存在
                print("模型存在，但是,", col, "不存在，设置为1")
                cols_done_dictionary[dl_model_name][col] = 1
            else:  # 如果模型存在&当前col也存在，则counter设置为2
                print("模型存在&当前,", col, "存在，设置为2")
                cols_done_dictionary[dl_model_name][col] = 2
        np.save(cols_done_dictionary_path, cols_done_dictionary)

    print('all F1_score:', F1_scores / len(columns))


# 把结果保存到csv
# report是classification_report生成的字典结果
def save_result_to_csv_cv(rows, day):
    path = 'result/' + str(day) + '_bert_dl-debug.csv' if debug else 'result/' + str(day) + '_bert_dl.csv'
    with codecs.open(path, "a", encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        f.close()


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in 1-SA-allAttr-dl-ml.py ...")

    print(">>>【1】正在读取文件", "。" * 100)
    data_training, data_validation, data_test = initData2()

    print("》》》【2】正在加载tokenizer", "。" * 100)
    tokenizer = get_tokenizer()

    print("》》》【3】设置模型参数", "。" * 100)
    epoch = 1 if debug else 3
    batch_size = 32
    batch_size_validation = 128

    print("》》》【4】构建模型", "。" * 100)
    dl_model_name = 'bert'
    print("model name is ", dl_model_name)

    print("》》》【5】训练模型", "。" * 100)
    # 多模型对比
    # times = 2
    # for i in range(times):
    trainModel2(dl_model_name, data_training, data_validation, data_test, tokenizer, epoch, batch_size, batch_size_validation)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


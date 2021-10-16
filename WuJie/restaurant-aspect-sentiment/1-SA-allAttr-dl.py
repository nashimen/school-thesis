# 方面级情感分类
# 针对所有属性
# 深度学习

import time, codecs, csv, math, numpy as np, random
from keras.utils import to_categorical

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
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
from keras import Model
from keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from keras_bert import Tokenizer, load_trained_model_from_checkpoint
import pandas as pd
pd.set_option('display.max_columns', None)

debug = False
debugLength = 500

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


# 构建CNN模型
def createCNNModel(cnn_filter, window_size, node):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])
        cnn = Conv1D(cnn_filter, window_size, name='conv')(x)
        cnn = MaxPool1D(name='max_pool')(cnn)
        flatten = Flatten()(cnn)
        x = Dense(node, activation='relu', name='Dense_target')(flatten)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)
        model = Model([x1_in, x2_in], p)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['mae'])
    model.summary()
    return model


# 构建GRU模型
def createGRUModel(dim, node):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        bi_gru = GRU(dim, name="gru_1")(x)

        x = Dense(node, activation='relu', name='Dense_target')(bi_gru)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)

        model = Model(inputs=[x1_in, x2_in], outputs=p)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['mae'])
    print(model.summary())

    return model


# 构建LSTM模型
def createLSTMModel(dim, node):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        lstm = LSTM(dim, return_sequences=False, name='lstm1')(x)

        x = Dense(node, activation='relu', name='Dense_target')(lstm)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)

        model = Model(inputs=[x1_in, x2_in], outputs=p)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['mae'])
    print(model.summary())

    return model


# 构建MLP模型
def createMLPModel(node):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        x = GlobalAveragePooling1D()(x)

        x = Dense(node, activation='relu', name='Dense_target')(x)
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

    if debug:
        print("indice_2 = ", indice_2)
        print("indice_others = ", indice_others)
        print("indice_random = ", indice_random)
        print("indice = ", indice)

    x = data['content'][indice]
    y = data[col][indice]

    return x, y


# 训练模型，输入为data_training和data_validation
# 采用交叉验证方式，合并训练集和验证集
def trainModel(dl_model_name, node, data_training, data_validation, tokenizer, epoch, batch_size, batch_size_validation):
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
    for index, col in enumerate(columns):
    # index = -1
    # columns = ['service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed']
    # for col in reversed(columns):
    #     index += 1
        # if col in ["location_traffic_convenience", "location_distance_from_business_district", "location_easy_to_find",
        #            "service_wait_time", "service_serving_speed", "service_parking_convenience", "service_waiters_attitude",
        #            "price_level", "others_willing_to_consume_again"]:
        #     continue
        print("current col is:", col)
        x, y = sampling(data, col)  # 喂给模型
        x = x.tolist()
        print("data length is", len(x))

        # 标签+2
        y = y + 2
        y_col = list(y)

        # 2.交叉验证数据集
        kf = KFold(n_splits=5)
        current_k = 0
        for train_index, validation_index in kf.split(x):
            # print("正在进行第", current_k, "轮交叉验证。。。")
            current_k += 1
            x_train, Y_train = np.array(x)[train_index], np.array(y_col)[train_index]
            x_validation, Y_validation = np.array(x)[validation_index], np.array(y_col)[validation_index]

            length = len(x_train)
            length_validation = len(x_validation)

            if dl_model_name == "mlp":
                model = createMLPModel(node)
            elif dl_model_name == "cnn":
                model = createCNNModel(128, 4, node)
            elif dl_model_name == "lstm":
                model = createLSTMModel(64, node)
            elif dl_model_name == "gru":
                model = createGRUModel(64, node)

            # 早停法，如果val_acc没有提高0.0001就停止
            earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=3)
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
            precision, recall, fscore, support = score(Y_validation, y_val_pred)
            print("precision = ", precision)
            print("recall = ", recall)
            print("fscore = ", fscore)
            print("support = ", support)

            report = classification_report(Y_validation, y_val_pred, digits=4, output_dict=True)
            print(report)

            F1_score = f1_score(Y_validation, y_val_pred, average='macro')
            F1_scores += F1_score
            print('第', index, '个属性', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(Y_validation, y_val_pred))
            print("%Y-%m%d %H:%M:%S", time.localtime())

            # 保存当前属性的结果,整体的结果根据所有属性的结果来计算
            print("正在保存结果。。。")
            save_result_to_csv(dl_model_name, node, report, col)

        # 提取特征，喂给机器学习模型
        # 已有的model在load权重过后
        # 取某一层的输出为输出新建为model，采用函数模型
        dense1_layer_model = Model(inputs=model.input,
                                   outputs=model.get_layer('Dense_target').output)

        length_x = len(x)
        # 以这个model的预测值作为输出
        cnn_feature = dense1_layer_model.predict(generateXSetForBert(x, length_x, batch_size, tokenizer), steps=math.ceil(length_x / batch_size))
        print("cnn_feature = ", cnn_feature.shape)

        # 训练机器学习模型
        ml_names = ["bayes", "ada", "svm", "randomForest", "decisionTree", "logicRegression"]
        y_col = np.array(y_col)
        for ml in ml_names:
            print("current ml model is ", ml)
            # 2.交叉验证数据集
            kf = KFold(n_splits=5)
            current_k = 0
            for train_index, validation_index in kf.split(x):
                # print("正在进行第", current_k, "轮交叉验证。。。")
                current_k += 1
                X_train_ml, Y_train_ml = cnn_feature[train_index], y_col[train_index]
                X_validation_ml, Y_validation_ml = cnn_feature[validation_index], y_col[validation_index]

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

                current_model.fit(X_train_ml, Y_train_ml)

                # 4.预测结果
                Y_predicts = current_model.predict(X_validation_ml)
                saveResult(dl_model_name, ml, node, Y_validation_ml, Y_predicts, col)

    print('all F1_score:', F1_scores / len(columns))


# 把结果保存到csv
# report是classification_report生成的字典结果
def save_result_to_csv(dl_model_name, node, report, col_name):
    accuracy = report.get("accuracy")

    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')

    data = [dl_model_name, node, col_name, macro_precision, macro_recall, macro_f1, accuracy]
    print("dl data:", data)
    path = 'result/allAttr_only_dl_cv-debug.csv' if debug else 'result/one_process_allAttr_cv_dl.csv'
    with codecs.open(path, "a", encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


# 将预测结果保存至文件
def saveResult(dl_model_name, ml, node, y_tokens_test, y_val_pred, col):
    # 计算各种评价指标&保存结果
    report = classification_report(y_tokens_test, y_val_pred, digits=4, output_dict=True)
    # print(report)
    accuracy = report.get("accuracy")
    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')
    data = [dl_model_name, ml, node, col, macro_precision, macro_recall, macro_f1, accuracy]
    print(data)
    # 保存至文件
    path = 'result/allAttr_dl_ml_sampling-debug.csv' if debug else 'result/one_process_allAttr_cv_dl_ml.csv'
    with codecs.open(path, "a", "utf_8_sig") as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


# 初始化数据，不拆分X和Y
def initData2():
    path_training = "data/sentiment_analysis_training_set.csv"
    path_validation = "data/sentiment_analysis_validation_set.csv"

    # 加载所有列
    data_training = pd.read_csv(path_training, nrows=debugLength if debug else None)
    data_validation = pd.read_csv(path_validation, nrows=debugLength if debug else None)
    # print("before:", data_training.columns)
    # 删除id和dish_recommendation和others_overall_experience三列
    drop_columns = ["dish_recommendation", "others_overall_experience", "id"]
    data_training = data_training.drop(drop_columns, axis=1)
    data_validation = data_validation.drop(drop_columns, axis=1)

    return data_training, data_validation


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in 1-SA-allAttr-dl-ml.py ...")

    print("》》》【1】正在读取文件", "。" * 100)
    data_training, data_validation = initData2()

    print("》》》【2】正在加载tokenizer", "。" * 100)
    tokenizer = get_tokenizer()

    print("》》》【3】设置模型参数", "。" * 100)
    epoch = 1 if debug else 3
    batch_size = 16
    batch_size_validation = 64

    print("》》》【4】构建模型", "。" * 100)
    dl_model_name = 'mlp'
    print("model name is ", dl_model_name)

    node = 128

    print("》》》【5】训练模型", "。" * 100)
    trainModel(dl_model_name, node, data_training, data_validation, tokenizer, epoch, batch_size, batch_size_validation)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of 1-SA-allAttr-dl-ml.py...")


# 方面级情感分类
# 针对所有属性
# 深度学习

import time, codecs, csv, math, numpy as np, random, datetime, os, jieba, re
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
from gensim.models import KeyedVectors

from keras.layers import Input, Flatten, Dense, Dropout, GRU, Bidirectional, Conv1D, LSTM, GlobalAveragePooling1D, Embedding
from keras.layers import MaxPool1D
from keras import Model, backend as K
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence

from keras_bert import load_trained_model_from_checkpoint
import pandas as pd
pd.set_option('display.max_columns', None)

# 一些超参数设置
debug = False
debugLength = 15000
dict_length = 50000 if debug else 3000000  # 词典长度
maxlen = 300  # padding的长度
ratio = 0.8  # 训练集划分比例

# keras_bert
bert_path = "../keras_bert/chinese_L-12_H-768_A-12"
bert_config_path = '../keras_bert/chinese_L-12_H-768_A-12/bert_config.json'
bert_checkpoint_path = '../keras_bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
bert_dict_path = '../keras_bert/chinese_L-12_H-768_A-12/vocab.txt'


# 构建GRU模型
def createGRUModel(dim):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, trainable=True)

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        bi_gru = GRU(dim, name="gru_1")(x)

        # x = Dense(node, activation='relu', name='Dense_target')(bi_gru)
        x = Dropout(0.4, name='dropout')(bi_gru)
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

        # x = Dense(node, activation='relu', name='Dense_target')(lstm)
        x = Dropout(0.4, name='dropout')(lstm)
        p = Dense(4, activation='softmax', name='softmax')(x)

        model = Model(inputs=[x1_in, x2_in], outputs=p)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['mae'])
    print(model.summary())

    return model


# 构建CNN模型
def createCNNModel(cnn_filter, window_size):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = Input(shape=(maxlen,), name="inputs")
        x = Embedding(input_dim=dict_length, output_dim=200, name='embedding_mlp', weights=[embedding_matrix], trainable=True)(inputs)
        cnn = Conv1D(cnn_filter, window_size, name='conv')(x)
        cnn = MaxPool1D(name='max_pool')(cnn)
        flatten = Flatten()(cnn)  # 可以测试一下Pooling
        # x = Dense(node, activation='relu', name='Dense_target')(flatten)
        x = Dropout(0.4, name='dropout')(flatten)
        p = Dense(4, activation='softmax', name='softmax')(x)
        model = Model(inputs, p)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['mae'])
    model.summary()
    return model


# 构建MLP模型
def createMLPModel():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = Input(shape=(maxlen,), name="inputs")
        x = Embedding(input_dim=dict_length, output_dim=200, name='embedding_mlp', weights=[embedding_matrix], trainable=True)(inputs)
        x = GlobalAveragePooling1D()(x)

        x = Dense(128, activation='relu', name='Dense_target')(x)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)

        model = Model(inputs=inputs, outputs=p)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['mae'])
    print(model.summary())

    return model


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


# 下采样函数，输入col和data，返回X和Y
def sampling(dealed_data, data, col):
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

    print("dealed_data's type = ", type(dealed_data))
    x = dealed_data[indice]
    y = data[col][indice]

    return x, y


# 目的：多模型对比
# 采用交叉验证方式，合并训练集和验证集，模型不需要保存
# 输入为data_training和data_validation
cols_done_dictionary_path = 'cols_done_file_w2v-debug.npy' if debug else 'cols_done_file_w2v.npy'
def get_keys(d, value):
    return [k for k, v in d.items() if v == value]
def trainModel(dl_model_name, dealed_data, data, epoch, batch_size):
    # 获取当天日期
    now_time = datetime.datetime.now()
    day = datetime.datetime.strftime(now_time, '%Y%m%d')

    # 生成X和Y
    columns = data.columns.tolist()
    columns.remove('content')
    columns.remove('words')
    print("columns = ", columns)
    print("columns' length = ", len(columns))
    F1_scores = 0
    # 查询已完成属性,json文件。加载文件，如果文件不存在或者无内容，则返回/创建一个空字典
    for index, col in enumerate(columns):
        if os.path.exists(cols_done_dictionary_path):
            cols_done_dictionary = np.load(cols_done_dictionary_path, allow_pickle=True).item()
            cols_done = get_keys(cols_done_dictionary.get(dl_model_name), 2)  # 已完成的属性
            cols_done_1 = get_keys(cols_done_dictionary.get(dl_model_name), 1)  # 完成了一次的属性
        else:
            cols_done_dictionary = {}
            cols_done = []
            cols_done_1 = []
        if col in cols_done:
            continue
        print("cols_done_dictionary = ", cols_done_dictionary)
        print("cols_done = ", cols_done)
        print("cols_done_1 = ", cols_done_1)

        print("current col is:", col)
        x, y = sampling(dealed_data, data, col)  # 喂给模型
        x = x.tolist()
        print("data length is", len(x))

        # 标签+2
        y = y + 2
        y_col = list(y)

        # 跑两次交叉验证，保证最终的数量为10，然后进行T检验
        times = 1 if col in cols_done_1 else 2
        for i in range(times):
            print("正在进行第", i, "次训练")
            # 2.交叉验证数据集
            kf = KFold(n_splits=5)
            current_k = 0
            rows = []
            for train_index, validation_index in kf.split(x):
                print("正在进行第", current_k, "轮交叉验证。。。")
                current_k += 1
                x_train, Y_train = np.array(x)[train_index], np.array(y_col)[train_index]
                x_validation, Y_validation = np.array(x)[validation_index], np.array(y_col)[validation_index]

                # one-hot
                Y_train_onehot = to_categorical(Y_train)
                Y_validation_onehot = to_categorical(Y_validation)

                if dl_model_name == "mlp":
                    model = createMLPModel()
                elif dl_model_name == "cnn":
                    model = createCNNModel(128, 4)
                elif dl_model_name == "lstm":
                    model = createLSTMModel(64)
                elif dl_model_name == "gru":
                    model = createGRUModel(64)
                elif dl_model_name == "bert":
                    model = createModel()

                # 早停法，如果val_acc没有提高0.0001就停止
                earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=3)
                model.fit(x_train, Y_train_onehot, epochs=epoch, batch_size=batch_size, verbose=1, callbacks=[earlystop_callback], validation_data=(x_validation, Y_validation_onehot))

                # 预测验证集
                y_val_pred = model.predict(x_validation)
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

                row = [dl_model_name, col, i, macro_precision, macro_recall, macro_f1, accuracy, mse, mae, r2]
                print("row:", row)
                rows.append(row)

                # 释放内存
                K.clear_session()

            # 保存当前属性的结果,整体的结果根据所有属性的结果来计算
            print("正在保存结果。。。")
            save_result_to_csv_cv(rows, day)

            # 保存已经跑完的属性及其频率
            # 格式为{model_name1:{col1:counter,col2:counter},model_name2:{col1:counter,col2:counter,col3:counter}}
            if dl_model_name not in cols_done_dictionary.keys():  # 如果模型不存在的话
                cols_done_dictionary[dl_model_name] = {col: 1}
                print("模型不存在，设置col为1")
            else:
                if col not in cols_done_dictionary.get(dl_model_name).keys():  # 如果模型存在，但是当前col不存在
                    print("模型存在，但是当前col不存在，设置col为1")
                    cols_done_dictionary[dl_model_name][col] = 1
                else:  # 如果模型存在&当前col也存在，则counter设置为2
                    print("模型存在&当前col存在，设置col为2")
                    cols_done_dictionary[dl_model_name][col] = 2
            np.save(cols_done_dictionary_path, cols_done_dictionary)

    print('all F1_score:', F1_scores / len(columns))


# 把结果保存到csv
# report是classification_report生成的字典结果
def save_result_to_csv_cv(rows, day):
    path = 'result/' + str(day) + '_w2v_dl-debug.csv' if debug else 'result/' + str(day) + '_w2v_dl.csv'
    with codecs.open(path, "a", encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        f.close()


# 初始化数据，不拆分X和Y
stoplist = pd.read_csv('../stopwords.txt').values
def initData():
    # 只要training数据集
    path_training = "data/sentiment_analysis_training_set.csv"
    # path_validation = "data/sentiment_analysis_validation_set.csv"

    # 加载所有列
    origin_data = pd.read_csv(path_training, nrows=debugLength if debug else None)
    # 删除id和dish_recommendation和others_overall_experience三列
    drop_columns = ["id"]
    origin_data = origin_data.drop(drop_columns, axis=1)

    print("all_data's length = ", len(origin_data))

    all_texts = preprocessing(origin_data, stoplist)

    # 利用keras的Tokenizer进行onehot，并调整未等长数组
    tokenizer = Tokenizer(num_words=dict_length)
    tokenizer.fit_on_texts(all_texts)

    word_index = tokenizer.word_index

    data_w = tokenizer.texts_to_sequences(all_texts)
    data_T = sequence.pad_sequences(data_w, maxlen=maxlen)

    # 数据划分，重新划分为训练集，测试集和验证集
    data_length = data_T.shape[0]
    print("data_length = ", data_length)

    return data_T, origin_data, word_index


# 数据预处理：去标点符号、分词、去停用词、转为tokenizer
def preprocessing(data, stoplist):
    # 去标点符号
    data['words'] = data['content'].apply(lambda x: re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？、~@#￥%……&*（）．；：】【|]+", "", str(x)))
    # 分词
    data['words'] = data['words'].apply(lambda x: list(jieba.cut(x)))

    words_dict = []
    texts = []

    # 去掉停用词
    # print(">>>去停用词ing in DataProcess.py...")
    for index, row in data.iterrows():
        line = [word.strip() for word in list(row['words']) if word not in stoplist]
        # print("line = ", line)

        words_dict.extend([word for word in line])
        texts.append(line)

    # print("words_dict's length = ", len(words_dict))
    # print("data[words] = ", data["words"])

    print(">>>end of processDataToTexts in dataProcess.py...")

    return texts


# 生成词向量矩阵matrix
w2v_path = "../embeddings/Tencent_AILab_ChineseEmbedding_300w.bin" if debug else "../embbedings/Tencent_AILab_ChineseEmbedding_300w.bin"
def load_w2v(word_index):
    print("word_index's lengh = ", len(word_index))
    t1 = time.time()
    wv_from_text = KeyedVectors.load(w2v_path)
    wv_from_text.init_sims(replace=True)  # 神奇，很省内存，可以运算most_similar
    print("加载w2v文件耗时：", (time.time() - t1) / 60.0, "minutes")
    w2v = wv_from_text.wv

    # 创建词向量索引字典
    embeddings_index = {}
    # 遍历得到word对应的embedding
    for word in w2v.vocab.keys():
        embeddings_index[word] = wv_from_text[word]

    # 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
    # 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+1，词向量维度）
    matrix = np.zeros((len(word_index) + 1, 200))
    # 遍历词汇表中的每一项
    for word, i in word_index.items():
        # 在词向量索引字典中查询单词word的词向量
        embedding_vector = embeddings_index.get(word)
        # print("embedding_vector = ", embedding_vector)
        # 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
        # print(word, ":", embedding_vector)
        # print("vector's length = ", len(embedding_vector))
        if embedding_vector is not None:
            matrix[i] = embedding_vector

    return matrix


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in 1-SA-allAttr-dl-ml.py ...")

    print("》》》【1】正在读取文件", "。" * 100)
    dealed_train, train, word_index = initData()

    print("》》》【2】正在加载w2v", "。" * 100)
    if debug:
        embedding_matrix = np.zeros((len(word_index) + 1, 200))
    else:
        embedding_matrix = load_w2v(word_index)

    # 语料库中的单词数量可能少于预训练词典的单词数量（例如debug的时候只传入少量数据)
    dict_length = min(dict_length, len(word_index) + 1)
    print("dict_length = ", dict_length)

    print("》》》【3】设置模型参数", "。" * 100)
    epoch = 1 if debug else 30
    batch_size = 512
    batch_size_validation = 512

    print("》》》【4】构建模型", "。" * 100)
    dl_model_name = 'mlp'
    print("model name is ", dl_model_name)

    print("》》》【5】训练模型", "。" * 100)
    # 多模型对比
    trainModel(dl_model_name, dealed_train, train, epoch, batch_size)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of 1-SA-allAttr-dl-ml.py...")


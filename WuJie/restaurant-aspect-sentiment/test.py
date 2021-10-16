import time, codecs, csv, math, numpy as np, pandas as pd, random, datetime, os


def sampling():
    # 构建DataFrame
    data_training = {"content": ["aaabbbccc", "bbbcccaaa", "aaabbbddd", "cccbbbaaa", "bbbccceee", "dddeeeaaa"],
                     "col1": [1, 1, 1, -1, 1, 1], "col2": [1, 1, -1, 1, 1, 1], "col3": [1, -1, 1, -1, 1, 1]}
    data_training = pd.DataFrame(data_training)
    print(data_training)
    indice = [0, 2, 4]
    print(data_training['content'][indice])
    print(data_training['content'][indice] + 1)

    for col in data_training.columns:
        if col == "content":
            continue
        indice_1 = data_training[(data_training[col] == 1)].index.tolist()
        print("indice_1 = ", indice_1)
        indice_others = data_training[(data_training[col] != 1)].index.tolist()
        print("indice_others = ", indice_others)
        length_others = len(indice_others)
        indice_random = random.sample(indice_1, length_others)
        print(indice_random)


def cv_test():
    data_training = {"content": ["aaabbbccc", "bbbcccaaa", "aaabbbddd", "cccbbbaaa", "bbbccceee", "dddeeeaaa"],
                     "col1": [1, 1, 1, -1, 1, 1], "col2": [1, 1, -1, 1, 1, 1], "col3": [1, -1, 1, -1, 1, 1]}
    data_training = pd.DataFrame(data_training)
    print(data_training)
    arr = [0, 1, 2, 3, 4]
    print("arr's type = ", type(arr))
    content = data_training["content"][arr]
    print(type(content))
    indice = [0, 2, 4]
    print(content["content"][indice])


def test():
    x = [1, 2, 3, 4, 5]
    print("x = ", x)
    # print(random.shuffle(x))
    # random.shuffle(x)
    print("x = ", x)
    y = [31, 31, 11]
    print(x + y)
    print(type(x + y))
    print(min(len(x), len(y)))

    df1 = pd.DataFrame(np.ones((4, 4))*1, columns=list('DCBA'), index=list('4321'))
    df2 = pd.DataFrame(np.ones((4, 4))*2, columns=list('DCBA'), index=list('4321'))
    print(df1)
    print(df2)
    df3 = df1.append(df2)
    print(df3)
    # 索引重置
    df3 = df3.reset_index(drop=True)
    print(df3)


def fetchTime():
    now_time = datetime.datetime.now()
    day = datetime.datetime.strftime(now_time, '%Y%m%d')
    print(day)


def get_keys(d, value):
    return [k for k, v in d.items() if v == value]


def saveDicts():
    dictionary = {'mlp': {'col1': 1, 'col2': 2, 'col3': 1, 'col4': 1}, 'cnn': {'col1': 2, 'col2': 1}}
    print("he")
    dictionary['lstm'] = {'col1': 1}
    dictionary['lstm']['col1'] = 2
    save_path = 'temp/cols_done_file.npy'
    np.save(save_path, dictionary)

    print(os.path.exists(save_path))
    read_dictionary = np.load(save_path, allow_pickle=True).item()
    print(read_dictionary)
    # 获取mlp中value为1的所有key
    print(dictionary.get('mlp').keys())
    print(dictionary.get('mlp').values())
    print(get_keys(dictionary.get('mlp'), 1))
    print(get_keys(dictionary.get('mlp'), 3))


from gensim.models import KeyedVectors
w2v_path = "G:\Coding\preWordEmbedding\Tencent_AILab_ChineseEmbedding.tar\Tencent_AILab_ChineseEmbedding.txt"
w2v_bin_path = "G:\Coding\preWordEmbedding\Tencent_AILab_ChineseEmbedding.tar\Tencent_AILab_ChineseEmbedding_5w.bin"
def loadTencentEmbedding():
    t1 = time.time()
    # wv_from_text = KeyedVectors.load_word2vec_format(w2v_path, binary=False, limit=50000)
    wv_from_text = KeyedVectors.load(w2v_bin_path)
    wv_from_text.init_sims(replace=True)  # 神奇，很省内存，可以运算most_similar
    print("加载文件耗时：", (time.time() - t1) / 60.0, "minutes")
    w2v = wv_from_text.wv
    print(w2v.vocab.keys())
    print(list(w2v.vocab.values())[0])
    print(wv_from_text[list(w2v.vocab.keys())[0]])

    # wv_from_text.save(w2v_bin_path)

    # print(wv_from_text['的'])


if __name__ == "__main__":
    print("start...")
    loadTencentEmbedding()
    # sampling()

    print("end...")
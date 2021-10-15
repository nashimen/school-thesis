import pandas as pd, numpy as np, time, re, os, jieba, networkx as nx, json, requests, traceback, ast, paddlehub as hub, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 显示所有列
pd.set_option('display.max_columns', None)


debug = False
debugLength = 20
current_file = "上海4星"
negative_score_threshold = 0.5


# 加载停用词
def load_stopwords(path='../stopwords.txt'):
    with open(path, encoding="utf-8") as f:
        stopwords = list(map(lambda x: x.strip(), f.readlines()))
    stopwords.extend([' ', '\t', '\n'])
    return frozenset(stopwords)
    return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))


threshold = 0.8
tfidf_model = TfidfVectorizer(tokenizer=jieba.cut, stop_words=load_stopwords(), analyzer='char_wb')
# 使用TextRank算法提取摘要
def get_abstract(content):
    # 使用逗号替换空格
    content = re.sub(r" +", ",", str(content))
    pattern = r',|\.|;|\?|:|!|\(|\)|，|。|、|；|·|！| |…|（|）|~|～|：|；'
    docs = list(re.split(pattern, content))
    # print("docs pre = ", docs)
    docs = list(filter(lambda x: len(str(x).strip()) > 1, docs))
    # print("docs = ", docs)
    if len(docs) == 0:
        return ''
    normalized_matrix = tfidf_model.fit_transform(docs)
    normalized_matrix_array = normalized_matrix.toarray()
    similarity = nx.from_scipy_sparse_matrix(normalized_matrix * normalized_matrix.T)
    scores = nx.pagerank(similarity)
    tops = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # 所有
    indices = list(map(lambda x: x[0], tops))

    # 遍历normalized_matrix_array，依次保存符合条件的摘要
    # 参考文献：Extractive summarization using supervised and unsupervised learning
    # 计算余弦相似度矩阵
    cosine_matrix = cosine_similarity(normalized_matrix_array)
    indices_final = []
    for index in indices:
        if len(indices_final) == 0:
            indices_final.append(index)  # 先放入第一个摘要
            continue
        max_consine = 0
        for current_index in indices_final:
            if cosine_matrix[index][current_index] > max_consine:
                max_consine = cosine_matrix[index][current_index]
        if max_consine < threshold:
            indices_final.append(index)
    return list(map(lambda idx: docs[idx], indices_final))


def readExtractText(path):
    # 读取文件
    current_data = pd.read_excel(path, nrows=debugLength if debug else None)
    # print(current_data.head())

    # 删除指定列
    delCols = ["作者", "房型", "作者点评数", "点赞数", "酒店回复", "发布日期"]
    current_data = current_data.drop(delCols, axis=1)

    # 提取短文本
    print("正在提取短文本...")
    current_data["shortTexts"] = current_data.apply(lambda row: get_abstract(row['评论文本']), axis=1)
    print(current_data.head())
    # 删除空白行
    colNames = current_data.columns
    current_data = current_data.dropna(axis=0, subset=colNames)
    # 保存文件
    target_path = "data/test/short_text_" + current_file + ".csv" if debug else "data/shortTexts/brief/brief_" + current_file + ".csv"
    current_data.to_csv(target_path, mode="w", encoding='utf_8_sig', index=False)


# 判断是否包含中文字符
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


# token_yang152 = '24.74e1ddeedadc2ceaa9150298307e9ea3.2592000.1628400979.282335-24520602'
# token_wu133 = '24.ec198262ebd83384d9240f8356480162.2592000.1628402149.282335-24520757'
token = '24.d6dbdf9d8505b274c48b683bdeae94d0.2592000.1634040737.282335-24838368'
print("token = ", token)
url = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token={}'.format(token)
def calculate_sentiment_single(texts):
    try:
        if type(texts) == str:
            texts = texts.strip('[')
            texts = texts.strip(']')
            texts = texts.replace("'", "")
            texts = texts.replace(" ", "")
            texts = texts.split(',')
        negative_probs_res = []
        for text in texts:
            if not is_Chinese(text):
                continue
            data = {
                'text': text
            }
            data = json.dumps(data)
            res = requests.post(url, data=data).text
            # res = json.loads(res)
            # print('1,', text, res)
            res = ast.literal_eval(res)

            loop_max = 0
            error_code = 'error_code'
            while error_code in res.keys():
                if loop_max >= 30:
                    break
                res = requests.post(url, data=data).text
                # res = json.loads(res)
                res = ast.literal_eval(res)
                loop_max += 1
                # print('2,', text, res)
            if loop_max > 15:
                print('3,', loop_max, text, res)
            if error_code in res.keys():
                negative_probs_res.append(0)
            else:
                negative_probs_res.append(res['items'][0]['negative_prob'])
                # print(text, res)
        if len(texts) != len(negative_probs_res):
            print("长度不一致！！出错辣！！！")
    except Exception as e:
        traceback.print_exc()
        calculate_sentiment_single(texts)
    return negative_probs_res


# 过滤掉正面摘要
def texts_filter(texts, score):
    texts = texts.strip('[')
    texts = texts.strip(']')
    texts = texts.replace("'", "")
    texts = texts.replace(" ", "")
    texts = texts.split(',')
    negative_texts = []
    for i in range(len(score)):
        if score[i] >= negative_score_threshold:
            negative_texts.append(texts[i])

    return negative_texts


# 过滤掉正面分数
def score_filter(score):
    negative_score = []
    for i in range(len(score)):
        if score[i] >= negative_score_threshold:
            negative_score.append(score[i])
    return negative_score


# 情感计算
hotel_done_dictionary_path = current_file + "_hotel_done_file-debug.npy" if debug else current_file + "_hotel_done_file.npy"
def calculateSentiment(path):
    df = pd.read_csv(path, nrows=debugLength if debug else None)
    # 删除短文本为空的行
    df = df.dropna(subset=["shortTexts"])
    # 5星评论不进行判断
    df = df.loc[df["评分"] != 5]
    print(df.head())

    # 依次取出每个酒店的数据进行计算（接口访问词数有限）
    hotel_names = set(df["名称"].tolist())
    print(current_file, ":", hotel_names)
    for current_hotel_name in hotel_names:
        # 查询当前酒店是否已经跑完
        if os.path.exists(hotel_done_dictionary_path):
            hotel_done_dictionary = (np.load(hotel_done_dictionary_path, allow_pickle=True)).tolist()
        else:
            hotel_done_dictionary = []
        print("current_hotel_name = ", current_hotel_name)
        if current_hotel_name in hotel_done_dictionary:
            continue
        print("已完成店铺：", hotel_done_dictionary)
        current_hotel = df.loc[df["名称"] == current_hotel_name]

        print("正在计算情感分:", current_hotel_name)
        current_hotel['score'] = current_hotel.apply(lambda row: calculate_sentiment_single(row['shortTexts']), axis=1)

        print("正在过滤正面摘要文本。。。")
        current_hotel['shortTexts-negative'] = current_hotel.apply(lambda row: texts_filter(row['shortTexts'], row['score']), axis=1)
        print("正在过滤正向分数。。。")
        current_hotel['score-negative'] = current_hotel.apply(lambda row: score_filter(row['score']), axis=1)

        # print(current_hotel.head())
        save_path_negative = "data/test/complaint_" + current_file + ".csv" if debug else "data/complaintContent/complaint_" + current_file + ".csv"
        current_hotel.to_csv(save_path_negative, encoding='utf_8_sig', index=False, mode='a')

        # 保存已经跑完的酒店名称
        hotel_done_dictionary.append(current_hotel_name)
        np.save(hotel_done_dictionary_path, hotel_done_dictionary)


if __name__ == "__main__":
    print("start...")

    # 数据预处理
    # 读取文件+短文本提取
    data_path = "data/test/" + current_file + ".xlsx" if debug else "data/original/" + current_file + ".xlsx"
    print("data_path = ", data_path)
    readExtractText(data_path)

    # 情感计算
    short_text_path = "data/test/short_text_评论-" + current_file + ".csv" if debug else "data/shortTexts/brief/brief_" + current_file + ".csv"
    calculateSentiment(short_text_path)

    print("end...")


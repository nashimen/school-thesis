from gensim.parsing.preprocessing import STOPWORDS
import gensim
import numpy as np

class CONFIG(object):
    NO_BELOW = 2
    NO_ABOVE = 0.99
    EXTEND_STOPWORDS = '../data/stopwords.txt'
    LDA_MODEL_PATH = './model/model.lda'
    TOPIC_NUM = 20


stopwords = STOPWORDS

def tokenize(text, extend_stopwords):
    words = [w for w in text.split(" ") if w not in stopwords]
    extend_stopwords_list = []
    es = open(extend_stopwords, "r")
    for line in es:
        extend_stopwords_list.append(line.strip())
    words = [w for w in words if w not in extend_stopwords_list]
    return words

def read_document(file_path):
    documents = []
    try:
        f = open(file_path)
        for line in f:
            documents.append(line.strip())
        return documents
    except:
        return []

def get_corpus_train(file_path, extend_stopwords, no_below=CONFIG.NO_BELOW, no_above=CONFIG.NO_ABOVE):
    documents = read_document(file_path)
    processed_docs = [tokenize(doc, extend_stopwords) for doc in documents]
    # for doc in processed_docs:
    #     print len(doc)
    #obtain: (word_id:word)
    word_count_dict = gensim.corpora.Dictionary(processed_docs)
    word_count_dict.filter_extremes(no_below=no_below, no_above=no_above)
    # word must appear >5 times, and no more than 20% documents
    bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]
    return word_count_dict, bag_of_words_corpus

def load_id2word(model_path):
    return gensim.corpora.Dictionary.load(model_path)

def get_corpus_test(file_path, word_count_dict):
    documents = read_document(file_path)
    processed_docs = [tokenize(doc) for doc in documents]
    bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]
    return bag_of_words_corpus

def gen_document_topic_distrubtion(file_path, topic_num=CONFIG.TOPIC_NUM, lda_model_path=CONFIG.LDA_MODEL_PATH,
                                   extend_stopwords=CONFIG.EXTEND_STOPWORDS,
                                   no_below=CONFIG.NO_BELOW, no_above=CONFIG.NO_ABOVE):
    data = open(file_path, 'r').readlines()
    word_count_dict, bag_of_words_corpus = get_corpus_train(file_path, extend_stopwords, no_below, no_above)
    lda_model = gensim.models.LdaModel.load(lda_model_path)
    doc_vec = []
    for corpus in bag_of_words_corpus:
        _v = lda_model.get_document_topics(corpus, minimum_probability=0)
        v = [0] * topic_num
        for idx, prob in _v:
            v[idx] = prob
        v = np.array(v)
        if np.sum((v - np.mean(v)) == 0) == topic_num:
            v[0] += 0.0001
        v = list(v)
        doc_vec.append(v)
    return data, lda_model, doc_vec, word_count_dict, bag_of_words_corpus


def cal_sen(pos, neg, docs):
    result = []
    for doc in docs:
        doc_sen = 0
        for word in doc:
            doc_sen += pos.get(word, 0)
            doc_sen += neg.get(word, 0)
        result.append(0 if doc_sen <0 else 1)
    return result

# def cal_f1(result, target):
#     assert len(result) == len(target)
#     n_pos = sum(result)
#     n_pos_t = sum(target)
#     n_neg = len(result) - sum(result)
#     n_neg_t = len(target) - sum(target)
#     tmp = np.array(result) + np.array(target)
#     n_cor_pos = sum(tmp == 4)
#     n_cor_neg = sum(tmp == 0)
#     precision_pos = (n_cor_pos/float(n_pos))
#     recall_pos = (n_cor_pos / float(n_pos_t))
#     precision_neg = (n_cor_neg/float(n_neg))
#     recall_neg = (n_cor_neg / float(n_neg_t))
#     print "precision_pos(4):", precision_pos
#     print "recall_pos:",recall_pos
#     print "precision_neg(0):",precision_neg
#     print "recall_neg:", recall_neg
    # precision = (n_cor_pos / float(n_pos) + n_cor_neg / float(n_neg)) / 2
    # recall = (n_cor_pos / float(n_pos_t) + n_cor_neg / float(n_neg_t)) / 2
    # F1 = 2*precision*recall/(precision + recall)
    # return precision, recall, F1

# def cal_f1(y_t,y_p):
#     _all_ = [0, 1, 2]
#     _f1_ = []
#     _p_ = []
#     _r_ = []
#     _n_t_ = []
#     _n_c_ = []
#     _n_p_ = []

#     _weight_ = []
#     __idx = np.array(y_p)==np.array(y_t)
#     count = float(__idx.sum())
#     _p = count/len(y_t)
#     for tgt in _all_:
#         _idx = np.array(y_t) == tgt
#         _n_t = float((np.array(y_t) == tgt).sum())
#         _n_c =  float(((np.array(y_t) == tgt) & (np.array(y_p)==np.array(y_t))).sum())
#         _n_p = float(( np.array(y_p) == tgt).sum())
#         _p = _n_c / _n_t
#         _r = _n_c / _n_p
#         _f1 = 2 * _p * _r / (_p + _r)
#         _p_.append(_p)
#         _r_.append(_r)
#         _f1_.append(_f1)
#         _weight_.append(_n_t)
#         _n_t_.append(_n_t)
#         _n_c_.append(_n_c)
#         _n_p_.append(_n_p)
# 	return np.array(_f1_).dot(np.array(_weight_))/np.sum(np.array(_weight_))

def cal_f1(y_t, y_p):
        _all_ = [0, 2]
        _f1_ = []
        _p_ = []
        _r_ = []
        _n_t_ = []
        _n_c_ = []
        _n_p_ = []
        _weight_ = []
        __idx = np.array(y_p)==np.array(y_t)
        count = float(__idx.sum())
        _p = count/len(y_t)
        for tgt in _all_:
            _idx = np.array(y_t) == tgt
            _n_t = float((np.array(y_t) == tgt).sum())
            _n_c =  float(((np.array(y_t) == tgt) & (np.array(y_t)==np.array(y_p))).sum())
            _n_p = float(( np.array(y_p) == tgt).sum())
            _p = _n_c / _n_t
            _r = _n_c / _n_p
            _f1 = 2 * _p * _r / (_p + _r)
            _p_.append(_p)
            _r_.append(_r)
            _f1_.append(_f1)
            _weight_.append(_n_t)
            _n_t_.append(_n_t)
            _n_c_.append(_n_c)
            _n_p_.append(_n_p)

        print ("total p:")
        p_total = np.array(_p_).dot(np.array(_weight_))/np.sum(np.array(_weight_))
        print (np.array(_p_).dot(np.array(_weight_))/np.sum(np.array(_weight_)))

        print ("total R:")
        r_total = np.array(_r_).dot(np.array(_weight_))/np.sum(np.array(_weight_))
        print (np.array(_r_).dot(np.array(_weight_))/np.sum(np.array(_weight_)))
        
        print ("total f1:")
        f1_total = np.array(_f1_).dot(np.array(_weight_))/np.sum(np.array(_weight_))
        print (np.array(_f1_).dot(np.array(_weight_))/np.sum(np.array(_weight_)))

def read_docs(path, spt, pos):
    result = []
    target = []
    with open(path, 'r') as ipt:
        for line in ipt.readlines():
            items = line.strip().split(spt)
            result.append(items)
            target.append(1 if pos else 0)
    assert len(result) == len(target)
    return result, target

def gen_seed_dic(path):
    result = {}
    with open(path, 'r') as ipt:
        for line in ipt.readlines():
            items = line.strip()
            result[items] = 1
    return result

def gen_batch_data(docs, target, n=3):
    permut = np.random.permutation(len(docs))
    docs = np.array(docs)
    target = np.array(target)
    batch = []
    for i in xrange(0, n):
        idx = permut[i * len(permut) / n: (i+1) * len(permut) / n]
        batch.append((list(docs[idx]), list(target[idx])))
    return batch


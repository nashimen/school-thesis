import io
import numpy as np
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
#import lightgbm as gbm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

class sen_model(object):

    def __init__(self, method, docs, target, test, seed_pos, seed_neg, gen_feat_method, n_feat, freq, **kwargs):
        self.method = method
        self.docs = docs
        self.target = target
        self.test = test
        self.seed_pos = seed_pos
        self.seed_neg = seed_neg
        self.gen_feat_method = gen_feat_method
        self.freq = freq
        self.n_feat = n_feat
        self.kwargs = kwargs
        self.train, self.test = self.gen_feat()
        self.model = self.fit()


    def gen_feat(self):
        train = []
        test = []
        if 'unigram' in self.gen_feat_method:
            self.uf_train, self.uf_test = self.unigram_feat()
            train.append(self.uf_train)
            test.append(self.uf_test)
        if 'bigram' in self.gen_feat_method:
            self.bf_train, self.bf_test = self.bigram_feat()
            train.append(self.bf_train)
            test.append(self.bf_test)
        if 'trigram' in self.gen_feat_method:
            self.tf_train,self.tf_test = self.trigram_feat()
            train.append(self.tf_train)
            test.append(self.tf_test)
        if 'seed' in self.gen_feat_method:
            self.sf_train, self.sf_test = self.seed_feat()
            train.append(self.sf_train)
            test.append(self.sf_test)
        train = np.concatenate(train, axis=1)
        test = np.concatenate(test, axis=1)
        return train, test

    def unigram_feat(self):
        print ('start init...')
        print ('start gen unigram feat..')
        word_dic = {}
        word2id = {}
        train = []
        test = []

        # 统计每个词的个数
        for doc in self.docs:
            for word in doc:
                tmp = word_dic.get(word, 0)
                tmp += 1
                word_dic[word] = tmp
        

        idx = 0
        # print(len(word_dic))
        for k, v in sorted(word_dic.items(), key=lambda item:item[1], reverse=True):
            word2id[k] = idx
            idx += 1
            if idx >= self.n_feat:
                break
        for doc in self.docs:
            feat = [0] * self.n_feat
            assert len(feat) == len(word2id)
            for word in doc:
                tmp = word2id.get(word, -1)
                if tmp != -1:
                    if self.freq:
                        feat[tmp] += 1
                    else:
                        feat[tmp] = 1
            train.append(feat)
        for doc in self.test:
            feat = [0] * self.n_feat
            assert len(feat) == len(word2id)
            for word in doc:
                tmp = word2id.get(word, -1)
                if tmp != -1:
                    if self.freq:
                        feat[tmp] += 1
                    else:
                        feat[tmp] = 1
            test.append(feat)
        print ('gen unigram feat successed..')
        return train, test


    def seed_feat(self):  # 提取情感词特征  有无正向，有无负向，正向个数，负向个数，正向-负向个数
        print ('start init...')
        print ('start gen seed feat..')
        train = []
        test = []
        for doc in self.docs:
            pos = 0
            neg = 0
            for word in doc:
                if word in self.seed_pos:
                    pos += 1
                if word in self.seed_neg:
                    neg += 1
            feat = [pos, neg, 1 if(pos > 0) else 0,1 if(neg > 0) else 0,  pos - neg, 1 if (pos - neg >=0) else 0]
            train.append(feat)

        for doc in self.test:
            pos = 0
            neg = 0
            for word in doc:
                if word in self.seed_pos:
                    pos += 1
                if word in self.seed_neg:
                    neg += 1
            feat = [pos, neg, 1 if(pos > 0) else 0,1 if(neg > 0) else 0, pos - neg, 1 if (pos - neg >= 0) else 0]
            test.append(feat)
        return train, test

    def bigram_feat(self):
        print ('start init...')
        print ('start gen bigram feat..')
        word_dic = {}
        word2id = {}
        train = []
        test = []
        for doc in self.docs:
            n = len(doc)
            for i in range(0, n-1):
                #bigarm
                tmp = word_dic.get((doc[i], doc[i+1]), 0)
                tmp += 1
                word_dic[(doc[i], doc[i+1])] = tmp
        # print(len(word_dic))
        idx = 0
        for k, v in sorted(word_dic.items(), key=lambda item:item[1], reverse=True):
            word2id[k] = idx
            idx += 1
            if idx >= self.n_feat:
                break
        for doc in self.docs:
            n = len(doc)
            feat = [0] * self.n_feat
            for i in range(0, n - 1):
                tmp = word2id.get((doc[i], doc[i+1]), -1)
                if tmp != -1:
                    if self.freq:
                        feat[tmp] += 1
                    else:
                        feat[tmp] = 1
            train.append(feat)
        for doc in self.test:
            n = len(doc)
            feat = [0] * self.n_feat
            assert len(feat) == len(word2id)
            for i in range(0, n - 1):
                tmp = word2id.get((doc[i], doc[i+1]), -1)
                if tmp != -1:
                    if self.freq:
                        feat[tmp] += 1
                    else:
                        feat[tmp] = 1
            test.append(feat)
        print ('gen bigram feat successed..')
        return train, test

    def trigram_feat(self):
        print ('start init...')
        print ('start gen trigram feat..')
        word_dic = {}
        word2id = {}
        train = []
        test = []
        for doc in self.docs:
            for i in range(len(doc)):
                word = doc[i - 2] + doc[i - 1] + doc[i]
                tmp = word_dic.get(word, 0)
                tmp += 1
                word_dic[word] = tmp
        idx = 0
        print(len(word_dic))
        for k, v in sorted(word_dic.items(), key=lambda item:item[1], reverse=True):
            word2id[k] = idx
            idx += 1
            if idx >= len(word_dic):
                break
        for doc in self.docs:
            feat = [0] * len(word_dic)
            assert len(feat) == len(word2id)
            for i in range(len(doc)):
                word = doc[i - 2] + doc[i - 1] + doc[i]
                tmp = word2id.get(word, -1)
                if tmp != -1:
                    if not self.freq:
                        feat[tmp] = 1
                    else:
                        feat[tmp] += 1
            train.append(feat)
        for doc in self.test:
            feat = [0] * len(word_dic)
            assert len(feat) == len(word2id)
            for i in range(len(doc)):
                word = doc[i - 2] + doc[i - 1] + doc[i]
                tmp = word2id.get(word, -1)
                if tmp != -1:
                    if not self.freq:
                        feat[tmp] = 1
                    else:
                        feat[tmp] += 1
            test.append(feat)
        print ('gen trigram feat successed..')
        return train, test

    def fit(self):
        print ('start training model..')
        if self.method == 'svm':
            svc = SVC()
            # model = OneVsRestClassifier(svm.SVC(kernel='linear'))
			# clf = model.fit(x_train, y_train)
            return model.fit(self.train, self.target)
            # return svc.fit(self.train, self.target)
        elif self.method == 'lr':
            lr = LogisticRegression()
            return lr.fit(self.train, self.target)
        elif self.method == 'rf':
            rf = RandomForestClassifier(n_estimators=30)
            return rf.fit(self.train, self.target)
        elif self.method == 'gbdt':
            gbdt = GradientBoostingClassifier()
            return gbdt.fit(self.train, self.target)
        # elif self.method == 'gbm':
        #     train_x, test_x, train_y, test_y = train_test_split(self.train,
        #                                                         self.target,
        #                                                         test_size=0.3,
        #                                                         random_state=0)
        #     lgb_train = gbm.Dataset(train_x, train_y)
        #     lgb_eval = gbm.Dataset(test_x, test_y)
        #     estimators = gbm.train(self.kwargs['param'],
        #                             lgb_train,
        #                             num_boost_round=3000,
        #                             valid_sets=lgb_eval,
        #                             early_stopping_rounds=20)
            return estimators
        else:
            raise 'wrong train method'
    #
    def predict(self, test):
        print ('start predict...')
        # if self.method == 'gbm':
        #     return self.model.predict(test)
        # else:
        return self.model.predict_proba(test)   # 输出三种极性的概率

    # def predict(self, test):   #直接输出所属类别
    #     print 'start predict...'
    #     return self.model.predict(test)
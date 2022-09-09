#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle 
from time import time

from sklearn.utils import shuffle # shuffle打乱样本的顺序，它只会打乱样本的顺序，每个样本的数据维持不变。
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def save_obj(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file ):
    with open(file, 'rb') as f:
        return pickle.load(f)


# In[2]:


# 评价指标函数
def f1_score_get(precision, recall):
    # tf.keras.backend.epsilon() 的值为 1e-07
    # 1.0e-7 == tf.keras.backend.epsilon() 得到 True
    f1 = 2 * precision * recall/( precision + recall + 1.0e-7 )
    return f1


# valid_y:真实标签
# predict_y:预测标签
def eval_p_r_f1(valid_y, predict_y):
    # precision 0 1 2 3
    # setting labels=[pos_label] and average != 'binary' will report scores for that label only.
    accuracy = accuracy_score(
                    y_true = valid_y,
                    y_pred = predict_y
                              )
    
    precision_0 = precision_score(
                    y_true = valid_y,
                    y_pred = predict_y,
                    labels = [0],
                    pos_label = 0,
                    average = 'micro'
                    )

    precision_1 = precision_score(
                    y_true = valid_y,
                    y_pred = predict_y,
                    labels = [1],
                    pos_label = 1,
                    average = 'micro'
                    )

    precision_2 = precision_score(
                    y_true = valid_y,
                    y_pred = predict_y,
                    labels = [2],
                    pos_label = 2,
                    average = 'micro'
                    )

    precision_3 = precision_score(
                    y_true = valid_y,
                    y_pred = predict_y,
                    labels = [3],
                    pos_label = 3,
                    average = 'micro'
                    )
    # recall 0 1 2 3
    recall_0 = recall_score(
                    y_true = valid_y,
                    y_pred = predict_y,
                    labels = [0],
                    pos_label = 0,
                    average = 'micro'
                    )

    recall_1 = recall_score(
                    y_true = valid_y,
                    y_pred = predict_y,
                    labels = [1],
                    pos_label = 1,
                    average = 'micro'
                    )

    recall_2 = recall_score(
                    y_true = valid_y,
                    y_pred = predict_y,
                    labels = [2],
                    pos_label = 2,
                    average = 'micro'
                    )

    recall_3 = recall_score(
                    y_true = valid_y,
                    y_pred = predict_y,
                    labels = [3],
                    pos_label = 3,
                    average = 'micro'
                    )
    
    # f1_score 0 1 2 3
    f1_score_0 = f1_score_get( precision_0, recall_0 )
    f1_score_1 = f1_score_get( precision_1, recall_1 )
    f1_score_2 = f1_score_get( precision_2, recall_2 )
    f1_score_3 = f1_score_get( precision_3, recall_3 )
    
    #由y_true 计算各标签权重
    num_0 = np.sum(valid_y==0)
    num_1 = np.sum(valid_y==1)
    num_2 = np.sum(valid_y==2)
    num_3 = np.sum(valid_y==3)
    
    total = num_0 + num_1 + num_2 + num_3
    p_0 = num_0/total
    p_1 = num_1/total
    p_2 = num_2/total
    p_3 = num_3/total
    
    precision_avg = p_0 * precision_0 + p_1 * precision_1 + p_2 * precision_2 + p_3 * precision_3
    recall_avg = p_0 * recall_0 + p_1 * recall_1 + p_2 * recall_2 + p_3 * recall_3
    f1_score_avg = p_0 * f1_score_0 + p_1 * f1_score_1 + p_2 * f1_score_2 + p_3 * f1_score_3
    
    mse= mean_squared_error(valid_y, predict_y)
    mae=mean_absolute_error(valid_y, predict_y)
    r2=r2_score(valid_y, predict_y)
    return accuracy, precision_avg, recall_avg, f1_score_avg,mse,mae,r2
     


# In[3]:


train_fenci = pd.read_csv( '/testcbd021_zhangjunming/wu_shiyan/train_dish.csv', usecols=['content'])
valid_fenci = pd.read_csv( '/testcbd021_zhangjunming/wu_shiyan/valid_environment.csv', usecols=['content'])


# In[4]:


#这里边的文件应该是合并的  只有label才是合并出来的

lists=['location_traffic_convenience','location_distance_from_business_district','location_easy_to_find','service_wait_time','service_waiters_attitude','service_parking_convenience','service_serving_speed','price_level','price_cost_effective','price_discount','environment_decoration','environment_noise','environment_space','environment_cleaness','dish_portion','dish_taste','dish_look','others_willing_to_consume_again']


# In[ ]:


dict={}
for ele in lists:
    train = pd.read_csv( '/testcbd021_zhangjunming/dataset/meituan/sentiment_analysis_trainingset.csv',usecols=['content',ele])#此处可以修改dish_taste进行一个替换
    valid = pd.read_csv( '/testcbd021_zhangjunming/dataset/meituan/sentiment_analysis_validationset.csv',usecols=['content',ele])
    train_x=train_fenci.content.values.tolist()
    valid_x=valid_fenci.content.values.tolist()
    train_y=train[ele].values+2
    valid_y=valid[ele].values+2
    n_features = 1000
    tfidf_vectorizer = TfidfVectorizer(
                                        max_df=0.95, 
                                        min_df=2,
                                        max_features=n_features,
                                   )
    corpus = train_x + valid_x
    #print(len(corpus))

    t0 = time()
    tfidf = tfidf_vectorizer.fit(corpus)
    print("done in %0.3fs." % (time() - t0))
    train_x = tfidf.transform(train_x)
    valid_x = tfidf.transform(valid_x)
    # 标准化
    scaler = StandardScaler(with_mean=False)
    train_x = scaler.fit_transform(train_x)
    valid_x = scaler.fit_transform(valid_x)
    
    n_estimators_range = [5000]
    max_depth_range = [50]
    min_samples_split_range = [50]
    min_samples_leaf_range = [1]
    
    for n_estimator in n_estimators_range:
        for max_depth in max_depth_range:
            for min_samples_split in min_samples_split_range:
                for min_samples_leaf in min_samples_leaf_range:
                    # 设置这个条件才合理
                    if min_samples_split > min_samples_leaf:
                        print('n_estimator = %d' % n_estimator)
                        print('max_depth = %d' % max_depth)
                        print('min_samples_split = %d' % min_samples_split)
                        print('min_samples_leaf = %d' % min_samples_leaf)
                        clf = RandomForestClassifier(
                                                        n_estimators= n_estimator,
                                                        criterion='gini',
                                                        max_depth= max_depth,
                                                        min_samples_split = min_samples_split,
                                                        min_samples_leaf = min_samples_leaf,
                                                        max_features= 'sqrt',
                                                        bootstrap=True,
                                                        random_state= 2020,
                                                        verbose= 1,
                                                        class_weight= 'balanced',
                                                        n_jobs = -1
                                                    )
                        t0 = time()
                        clf.fit(train_x, train_y)
                        print("done in %0.3fs." % (time() - t0))
                        predict_y = clf.predict(valid_x)
                        # predict_prob_y = clf.predict_proba(valid_x)
                        # 评价指标值
                        accuracy, precision_avg, recall_avg, f1_score_avg,mse,mae,r2 = eval_p_r_f1(valid_y, predict_y)
                        dict[ele]=[accuracy, precision_avg, recall_avg, f1_score_avg,mse,mae,r2]
                
                


# In[ ]:


jieguodata=pd.DataFrame(dict)


# In[ ]:


jieguodata.to_excel('randomforestjieguo.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





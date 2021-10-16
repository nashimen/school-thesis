#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

sentiment_dictionary = "sentiVocab\DLUT.xlsx"
sentiment_dictionary_csv = "sentiVocab\DLUT.csv"

pre_word_embedding = "config/preEmbeddings/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"

# 百度点石数据集
baidu_sentiment = "datasets/baidu/data_train.csv"
baidu_sentiment_test = "datasets/baidu/data_test.csv"

synonym_txt = "sentiVocab/dict_synonym.txt"
synonym_xlsx = "sentiVocab/dict_synonym.xlsx"

# 停用词路径
stopwords_path = "stopwords.txt"


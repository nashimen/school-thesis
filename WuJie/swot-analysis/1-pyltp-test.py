# -*- coding: utf-8 -*-
import os
LTP_DATA_DIR = r'C:\Softwares\coding\Ltp\ltp_data\ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`


from pyltp import Segmentor
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型
words = segmentor.segment('大明王很喜欢一个人')  # 分词
print ('\t'.join(words))
segmentor.release()  # 释放模型


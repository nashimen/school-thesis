import time, codecs, csv, math, numpy as np, random, datetime, os, gc, jieba, re, sys
import paddlehub as hub
from paddlenlp import Taskflow
import jieba.posseg as pseg
# from pylab import *

sys.setrecrsionlimit(10)

import warnings
warnings.filterwarnings("ignore", category=Warning)

pd.set_option('display.max_columns', None)

debug = True
debugLength = 30


def ie_test(passed_ie):
    line = "店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"
    print(ie(line))


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 初始化paddle模块
    schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']} # Define the schema for opinion extraction
    ie = Taskflow("information_extraction", schema=schema)

    # 测试ie
    ie_test(ie)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")


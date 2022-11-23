import sys

import os
import pandas as pd
file_list = list()
# csv文件所在目录
files = os.listdir('data')
for file in files:
    print("file:", file)
    df = pd.read_csv("data/" + file)
    df['filename'] = file  # filename为新增的列名，可以根据需求自己设置
    file_list.append(df)
all_VA = pd.concat(file_list,axis=0,ignore_index=True)
all_VA.to_csv('all_VA.csv')

sys.exit(10086)
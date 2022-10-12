import time, codecs, csv, math, numpy as np, random, datetime, os, jieba, re, sys, string, pandas as pd
import paddlehub as hub
from lexical_diversity import lex_div as ld

text = "太阳当空照花儿对我笑"
flt = ld.flemmatize(text)
print(flt)

print(">>>>>>>>>>>end")


from sklearn.linear_model import LogisticRegression
import time, codecs, csv, numpy as np
from keras.utils import to_categorical


x = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]
y = [11, 12, 13, 11, 12, 13, 11, 12, 13]
# y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)

model = LogisticRegression(C=1e5)
model.fit(x, y)

x_validation = [[10, 10, 10], [11, 11, 11], [12, 12, 12]]
y_pred = model.predict(x_validation)
print(y_pred)

print("end...")


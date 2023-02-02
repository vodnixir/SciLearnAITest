from sklearn import metrics
from sklearn import tree
# from sklearn.datasets import load_svmlight_file
import sklearn.datasets
import numpy as np



X, y = sklearn.datasets.fetch_openml(data_id=40594, return_X_y=True, as_frame=False)
print(y)
print(X)
y[y == "TRUE"] = 1
y[y == "FALSE"] = 0
y = y.astype(int)
print(y)
exit()

X_train, y_train = load_svmlight_file("d:\\tmp\\a1a.txt")
print(f"X_train = {X_train}, y_train = {y_train}")
exit()

curCnt = 0
maxCnt = 100
l_test = []
x_test = []
l_work = []
x_work = []

for i in range(10000):
    if i < 10:
        if curCnt < maxCnt // 2:
            x_work.append([i])
            l_work.append(0)
        else:
            x_test.append([i])
            l_test.append(0)
        curCnt += 1
        if curCnt >= maxCnt:
            curCnt = 0
    else:
        d = str(i)
        d = int(d[-2])
        if curCnt < maxCnt // 2:
            x_work.append([i])
            l_work.append(d % 2)
        else:
            x_test.append([i])
            l_test.append(d % 2)
        curCnt += 1
        if curCnt >= maxCnt:
            curCnt = 0

features = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x_work, l_work)

y_pred = clf.predict(x_test)

accuracy = metrics.accuracy_score(l_test, y_pred)
# print(f"x_work = {x_work}, l_work = {l_work}")
# print(f"x_test = {x_test}, l_test = {l_test}")
# print(f"y_pred = {y_pred}, l_work = {l_work}")

print(f"accuracy = {accuracy}")

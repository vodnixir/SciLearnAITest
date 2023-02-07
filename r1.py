from sklearn import metrics
from sklearn import tree
import sklearn.datasets
import numpy as np


def f(x):
    if (x >= 1000 and x <= 1500) or (x >= 6000 and x <= 6500):
        return "light"
    return "dark"

def genData(beg, end):
    x = []
    y = []
    for i in range(beg, end, 2):
        x.append([i])
        y.append(f(i))
    return x, y

def main():
    beg = 0
    end = 4000
    wx, wy = genData(beg, end)
    tx, ty = genData(-10001, 10001)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(wx, wy)
    y_pred = clf.predict(tx)
    accuracy = metrics.accuracy_score(ty, y_pred)
    print(f"accuracy = {accuracy}")


main()













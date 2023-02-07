from sklearn import metrics
from sklearn import tree
import sklearn.datasets
import numpy as np
import random


def func(x):
    return 0.5 * x + 2.0

def genDataLight(beg, end, min, max):
    f = []
    l = []
    for x in range(beg, end, 2):
        y = func(x)
        y = random.uniform(y + min, y + max)
        f.append([x, y])
        l.append("Light")
    return f, l

def genDataDark(beg, end, min, max):
    f = []
    l = []
    for x in range(beg, end, 2):
        y = func(x)
        y = random.uniform(y - min, y - max)
        f.append([x, y])
        l.append("Dark")
    return f, l



def main():
    beg = -10000
    end = 10000
    min = -20.0
    max = 100.0
    delta = 100
    wx1, wy1 = genDataLight(beg, end, min, max)
    wx2, wy2 = genDataDark(beg, end, min, max)
    tx1, ty1 = genDataLight(beg + delta + 1, end + delta + 1, min, max)
    tx2, ty2 = genDataDark(beg + delta + 1, end + delta + 1, min, max)
    clf = tree.DecisionTreeClassifier()
    x = [*wx1, *wx2]
    y = [*wy1, *wy2]
    clf = clf.fit(x, y)
    y_pred = clf.predict([*tx1, *tx2])
    accuracy = metrics.accuracy_score([*ty1, *ty2], y_pred)
    print(f"accuracy = {accuracy}")


main()









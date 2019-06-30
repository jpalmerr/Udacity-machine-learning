import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("Data/train.csv").values
# print(data)

clf = DecisionTreeClassifier()

# training dataset
xtrain = data[0:21000, 1:]
train_label = data[0:21000, 0]

clf.fit(xtrain, train_label)

# testing data
xtest = data[21000:, 1:]
actual_label = data[21000:, 0]

# a test sample

var = input("Please enter a number for which image to try, between 0 and 21000: ")
value = int(var)

print("You entered: " + var)
print("predicting...")
d = xtest[value]
d.shape=(28,28)
pt.imshow(255-d, cmap='gray')
print(clf.predict( [xtest[value]]))
pt.show()

# test results

# p = clf.predict(xtest)
#
# count = 0
# for i in range(0, 21000):
#     count += 1 if p[i] == actual_label[i] else 0
# print("Accuracy = ", (count/21000)*100)

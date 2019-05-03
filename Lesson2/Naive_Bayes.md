## Gaussian NB Deployment quiz

[Code](https://github.com/jpalmerr/Udacity-machine-learning/blob/master/public/classifyNBsklearn.png)
for `classifyNB.py` file.

[Graph](https://github.com/jpalmerr/Udacity-machine-learning/blob/master/public/classifyNBsklearnGraph.png)

**Accuracy**

```
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# now print the accuracy

from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)
# returns a percentage
```
Accuracy is defined as the number of test points that are classified correctly divided by the total number of test points.

`print clf.score(features_test, labels_test)`
also works.

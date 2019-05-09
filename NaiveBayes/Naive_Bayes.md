## Gaussian NB Deployment quiz

![Code](/public/classifyNBsklearn.png)
for `classifyNB.py` file.

![Graph](/public/classifyNBsklearnGraph.png)

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


## So why is Naive Bayes 'naive'?

It returns a 'ratio' on whether a `label` is more or less likely etc.
It is called naive because it ignores one thing:
```
Word Order
```

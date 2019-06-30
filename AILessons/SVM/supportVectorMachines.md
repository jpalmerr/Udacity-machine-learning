 # SVM

 Support Vector Machine

 - They find a separating line
 - called a hyper plane
 - from the data of two different classes

We want to "maximise" the distance to the nearest point of the two data sets. This is called the `margin`.

The nearest points are called the `support vectors` and the separating line is called the `hyperplane`

Think of: "This is the widest road that seperates the two groups".

The aim is to maximise the `robustness` of the result.

This is a constrained optimisation problem. It's achieved using Lagrange multipliers (throwback!).

## Advantages

- effective in high dimensional spaces
- still effective in cases where number of dimensions is greater than the number of samples
- uses a subset of training points in the decision function => also memory efficient
- versatile: different kernel functions can be specified for he decision function

## Disadvantages

- if the number of features is much greater than he number of samples, avoid over-fitting in choosing Kernel functions and regularisation term is crucial.
- SVMs do not directly provide probability estimates

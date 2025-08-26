import numpy as np

## Penalizes larger errors more due to squaring
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

## Less sensitive to outliers
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

## Measures the proportion of the variance in the target that is predictable from the features.
## Also called the goodness-of-fit score.
## can you for the accuracy of the model 
def r2_score(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

## Square root of MSE → brings error back to original units
## More interpretable than MSE when unit of measurement matters
## When you want an error metric in the same scale as the target variable
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

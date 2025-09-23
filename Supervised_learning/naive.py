# Naive Bayes is a probabilistic machine learning model used for classification tasks.
# It is based on Bayes' theorem with the "naive" assumption of conditional independence between every pair of features given the class.
# P(A∣B)=P(B∣A)⋅P(A)​/P(B)

# P(A∣B): Posterior probability
# → The probability of event A (the hypothesis) given that B (the evidence) has occurred.
# (probability of the class given the data)

# P(B∣A): Likelihood
# → The probability of observing B given that A is true.
# (probability of the data given the class)

# P(A): Prior probability
# → The initial probability of A before observing B.
# (probability of the class)

# P(B): Marginal probability
# → The total probability of observing B under all possible hypotheses.

# assumptions :- features are independent of each other
# Our goal is to find the class y that has the highest posterior probability

# for categorical data we can just count the number of times a feature appears in a class and divide it by the total number of times the feature appears in all classes
# but for continuous data we need to use gaussian probability density function (pdf) to calculate the likelihood
import numpy as np

class NaiveBayes:
    def __init__(self):
        n_classes = None
        self._mean = None
        self._var = None
        self._priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._means = np.zeros((n_classes, n_features), dtype=np.float64)
        self._variances = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            # All our calculations (mean, variance, prior) are conditional on the class.
            # We need to isolate the data points belonging only to the current class to
            # calculate these conditional statistics.

            self._means[idx, :] = X_c.mean(axis=0)
            self._variances[idx, :] = X_c.var(axis=0)
            
            # This is where we summarize the feature distributions for the class 'c'.
            # We assume a Gaussian distribution for P(x_i | y=c). A Gaussian distribution is
            # completely defined by its mean and variance. By calculating these, we are
            # "learning" the shape of the bell curve for each feature, given that it belongs
            # to class 'c'.

            # The prior is the ratio of samples of class 'c' to the total number of samples.
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            
            # This calculates P(y), the prior probability. It represents our belief about how
            # frequent this class is in the dataset before we've seen any features.
            # A class that appears 80% of the time is inherently more likely than one
            # that appears only 20% of the time.

    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        posteriors = []

        # Calculate the posterior probability for each class
        for idx, c in enumerate(self._classes):
            # Get the stored prior for the current class
            prior = np.log(self._priors[idx])
            
            # Why we use log:
            # We are going to multiply many small probabilities together. This can lead to
            # "floating point underflow" where the result is so small it becomes zero.
            # By using logarithms, we can turn the multiplication into a sum:
            # log(a * b) = log(a) + log(b). This is numerically much more stable.
            # The class with the highest log probability will also have the highest probability.

            # Calculate the class conditional probability (Likelihood)
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            
            # This calculates the log of the Likelihood: log(P(X | y)).
            # Because of the "naive" assumption, this is Σ log(P(x_i | y)). We calculate the
            # probability of each feature value given the class using our Gaussian PDF helper
            # and then sum the logs.

            # Calculate the full posterior (in log form)
            posterior = prior + class_conditional
            posteriors.append(posterior)
            # This is the final Naive Bayes calculation: Posterior ∝ Prior * Likelihood.
            # In log form, this becomes: log(Posterior) = log(Prior) + log(Likelihood).

        # Return the class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # This function implements the formula for a Gaussian PDF. It's how we calculate
        # P(x_i | y) for our numerical features. It tells us how "likely" it is to see a
        # particular feature value `x_i` given the mean and variance we learned for that
        # class. A small epsilon is added to the variance to prevent division by zero.
  
        mean = self._means[class_idx]
        var = self._variances[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var + 1e-9))
        denominator = np.sqrt(2 * np.pi * var + 1e-9)
        return numerator / denominator
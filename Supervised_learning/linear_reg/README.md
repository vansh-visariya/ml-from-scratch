Linear Regression and Regularization: A Mathematical Guide
This document provides a detailed explanation of Linear Regression and the three primary regularization techniques used to improve it: Lasso, Ridge, and Elastic Net. The main focus is on the mathematical equations that define these models and explain their distinct behaviors.

1. Standard Linear Regression
Linear Regression is a fundamental algorithm in machine learning and statistics used for predictive analysis. The goal is to model the linear relationship between a dependent variable (y) and one or more independent variables (X).

The Hypothesis Function
For a dataset with n features, the predicted value (ŷ) is a linear combination of the input features. This is our hypothesis function:

**ŷ = w₁x₁ + w₂x₂ + ⋯ + wₙxₙ + b**

In a more compact vector form, this can be written as:

**ŷ = xw + b**

ŷ: The predicted output value.

w: The vector of model weights or coefficients. w_j represents the weight for the j-th feature.

x: The input feature vector.

b: The bias term (or y-intercept).

The Cost Function: Mean Squared Error (MSE)
To find the optimal values for the weights (w) and bias (b), we need to minimize a cost function. The most common cost function for linear regression is the Mean Squared Error (MSE), which measures the average of the squared differences between the actual values (y) and the predicted values (ŷ).

**J(w, b) = (1/m) * Σᵢ (yᵢ - ŷᵢ)²**
 
m: The number of training examples.

yᵢ: The actual value for the i-th training example.

ŷᵢ: The predicted value for the i-th training example.

The model "learns" by finding the w and b that make this cost J as small as possible. However, this approach can lead to problems like overfitting, especially when dealing with a large number of features.

2. The Need for Regularization
Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function. This penalty discourages the model from learning overly complex patterns by keeping the weights small.

The general form of a regularized cost function is:

***Regularized Cost=Original Cost (MSE)+Penalty Term***

The key difference between Lasso, Ridge, and Elastic Net lies in the mathematical form of their penalty terms.

3. Lasso Regression (L1 Regularization)
Lasso (Least Absolute Shrinkage and Selection Operator) Regression adds a penalty equal to the absolute value of the magnitude of the coefficients. Its most important characteristic is its ability to perform automatic feature selection.

Mathematical Equation
The cost function for Lasso Regression is:

**J_Lasso(w, b) = (1/m) * Σᵢ (yᵢ - ŷᵢ)² + λ * Σⱼ |wⱼ|**

λ (Lambda): The regularization parameter that controls the strength of the penalty.

Σ|wⱼ|: The L1 Norm of the weight vector. This is the sum of the absolute values of the weights.

How it Works & Why it's Different
The use of the absolute value |wⱼ| is what makes Lasso unique. The derivative of |wⱼ| is sign(wⱼ), which is a constant +1 or -1 (for non-zero wⱼ). This creates a constant "pull" towards zero during optimization.

If a feature's contribution to reducing the MSE is less than this constant pull λ, the optimization algorithm will shrink its coefficient exactly to zero. This effectively removes the feature from the model.

When to Use Lasso
When you need feature selection. If you have a dataset with many features and you suspect that only a subset of them are actually useful, Lasso is an excellent choice.

To create simpler, more interpretable models. By reducing the number of features, the final model is easier to understand and explain.

4. Ridge Regression (L2 Regularization)
Ridge Regression adds a penalty equal to the square of the magnitude of the coefficients. It is particularly effective at handling multicollinearity (when features are highly correlated).

Mathematical Equation
The cost function for Ridge Regression is:

**J_Ridge(w, b) = (1/m) * Σᵢ (yᵢ - ŷᵢ)² + λ * Σⱼ wⱼ²**
 
λ (Lambda): The regularization parameter.

Σwⱼ²: The L2 Norm (squared) of the weight vector. This is the sum of the squared values of the weights.

How it Works & Why it's Different
The penalty term wⱼ² has a derivative of 2wⱼ. This means the penalty's effect is proportional to the size of the weight. Large weights are penalized more heavily than small weights.

Unlike Lasso's constant pull, this proportional penalty means that as a weight gets closer to zero, the penalty's push to make it smaller also diminishes. As a result, Ridge can make coefficients very close to zero, but it will never make them exactly zero. It shrinks the coefficients of correlated predictors together.

When to Use Ridge
When you believe most of your features are useful. Ridge will retain all features in the final model, just with reduced weights.

When you are dealing with multicollinearity. Ridge is more stable than Lasso when you have highly correlated features.

5. Elastic Net Regression (L1 + L2 Regularization)
Elastic Net is a hybrid model that combines the penalties of both Lasso and Ridge. It aims to get the best of both worlds: performing feature selection like Lasso while handling correlated features like Ridge.

Mathematical Equation
The cost function for Elastic Net is:

**J_ElasticNet(w, b) = (1/m) * Σᵢ (yᵢ - ŷᵢ)² + λ₁ * Σⱼ |wⱼ| + λ₂ * Σⱼ wⱼ²**
 
This is often written using a mixing parameter α (alpha) that controls the blend between L1 and L2 regularization:

**J_ElasticNet(w, b) = MSE + λ * (α * Σⱼ |wⱼ| + (1 - α) * Σⱼ wⱼ²)**

λ: The overall strength of the penalty.

α: The mixing parameter, where 0 ≤ α ≤ 1.

If α = 1, Elastic Net becomes Lasso Regression.

If α = 0, Elastic Net becomes Ridge Regression.

How it Works & Why it's Different
Elastic Net has the ability to shrink some coefficients to zero (thanks to the L1 part) and also shrink groups of correlated features together (thanks to the L2 part). It overcomes a limitation of Lasso, which tends to arbitrarily select only one feature from a group of highly correlated features.

When to Use Elastic Net
When you have highly correlated features. Elastic Net is often preferred over Lasso in this scenario because it can select the entire group of correlated features instead of just one.

When you are unsure whether to use Lasso or Ridge. Elastic Net provides a good middle ground and can often perform better than either one on its own.

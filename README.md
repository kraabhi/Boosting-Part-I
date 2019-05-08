                                                            Boosting-Part-I

# Ensemble: 

During prediction of target variable by any machine learning method, we get different output in comparision to actual.The main cause of difference is noice, variance and bias.Ensemble helps to reduce these factors except for the noice which is not under our control.Ensemble is collection of different models whose individual results are aggregated to get final result.There are two types of ensemble methods.

1. Bagging :
We randomly takes various samples from the train data (with replacement)and fit different models to different samples and after getting their individual predictions we combine their results.Samples are taken in such a way that each model differs from other.
Because this technique takes many uncorrelated learners to make a final model, it reduces error by reducing variance. Example of bagging ensemble is Random Forest.

2. Boosting :
Here the predictors are not made independently, but sequentially.
This technique employs the logic in which the subsequent predictors learn from the mistakes of the previous predictors.The predictors can be chosen from a range of models like decision trees, regressors, classifiers etc. Because new predictors are learning from mistakes committed by previous predictors, it takes less time/iterations to reach close to actual predictions. But we have to choose the stopping criteria carefully or it could lead to overfitting on training data. Gradient Boosting is an example of boosting algorithm.

In case of classification task a weak learner has an error rate that is slightly lesser than 0.5 in classifying the object and a strong learner has an error rate closer to 0.To convert a weak learner into strong learner, we take a family of weak learners, combine their results for example by voting. This turns this family of weak learners into strong learners.But the important thing is that the weak learners must not be correlated.

# Different Boosting algorithms

# 1. AdaBoost(Adaptive Boosting):

The weak learners in AdaBoost are decision trees with a single split, called decision stumps.It works by putting more weight on difficult to classify instances and less on those already handled well and can be used for both classification and regression problem.

It fits a sequence of weak learners on different weighted training data. It starts by predicting original data set and gives equal weight to each observation. If prediction is incorrect using the first learner, then it gives higher weight to observation which have been predicted incorrectly. Being an iterative process, it continues to add learner(s) until a limit is reached in the number of models or accuracy.

# In simple terms:
After training a classifier at any level, ada-boost assigns weight to each training item. Misclassified item is assigned higher weight so that it appears in the training subset of next classifier with higher probability. After each classifier is trained, the weight is assigned to the classifier as well based on accuracy.So the idea is to set weights to both classifiers and data points (samples) in a way that forces classifiers to concentrate on observations that are difficult to correctly classify. This process is done sequentially in that the two weights are adjusted at each step as iterations of the algorithm proceed. This is why Adaboost is referred to as a sequential ensemble method 
The more accurate classifier is assigned higher weight so that it will have more impact in the final outcome. A classifier with 50% accuracy is given a weight of zero, and a classifier with less than 50% accuracy is given negative weight.

Example :
predicted: 1 1 -1 1
actual: -1 1 -1 1
weights: 0.5 0.3 0.1 0.01
misclassification rate / error = (0.5*1 + 0.3*0 + 0.1*0 + 0.01*0) / (0.5 + 0.3 + 0.1 + 0.01)

Here 0.5*1 or 0.3*0 says if predicted is equal to actual then we multiply the corresponding weight by 0 else by 1.

error = 0.549450549450
Next, we choose weight for the classifier 
α=  1/2 * ln(1- error / error)
α = -0.08
We can see error rate is more then 50 % so we are getting -ve weight for that classifier.If for classifier we get error rate less then 50% we get positive weight for that classifier and  classifier with exact 50% error weight comes out to be 0.

Now we will update the weights of samples such that missclassified samples get more weight and will be given more weightage further.

# Suggest you to refer : https://towardsdatascience.com/adaboost-for-dummies-breaking-down-the-math-and-its-equations-into-simple-terms-87f439757dcf for deeper mathematics behind adaboost.

# 2. Gradient Boosting Algorithm :

The main difference between adaboost and GBM lies in the fact that how it identify the shortcomings of weak learners (eg. decision trees). While the AdaBoost model identifies the shortcomings by using high weight data points, gradient boosting performs the same by using gradients in the loss function.

The loss function is a measure indicating how good are model’s coefficients are at fitting the underlying data.Selection of loss function depends on the problem statement. For example if we are doing any regression problem of predicting sales then loss function will be error between true and predicted sales.

We define a loss function which is MSE (mean squared error).We want our predictions, such that our loss function (MSE) is minimum. By using gradient descent and updating our predictions based on a learning rate, we can find the values where MSE is minimum.
So, we are basically updating the predictions such that predicted values are close to actual values.

Fit Gradient Boosting model :

Predictions=ypi
Loss=L(yi,ypi)
Loss=MSE=∑(yi−ypi)2
ypi=ypi+α∗δL(yi,ypi)/δypi
ypi=ypi+α∗δ∑(yi−ypi)2/δypi
ypi=ypi−α∗2∗∑(yi−ypi)

So when are fitting new model on residuals we are actually adjusting our predictions using the fit on residuals.

# Steps:

1. Fit the base model and find predictions y_predicted1 
2. Find the difference between actual and predicted y -y_predicted1  names it as e1, so e1 = y-y_predicted1 
3. Now fit the new model on e1 with same inputs and again calculate predictions as e1_predicted
4. Add the predictions to previous predictions y_predicted2 = y_predicted1 + e1_predicted
5. Fit another model on the residuals that left e2 = y - y_predicted2 and repeat the process untill the sum of residuals become constant.
# Balancing the unbalanced data and applying tunned XGB on it
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Import any train and test data
train = pd.read_csv("C:/Users/admin/Desktop/practice_problems/loan_prediction_upload/train_cax_loan_prediction.csv"")
test = pd.read_csv("C:/Users/admin/Desktop/practice_problems/loan_prediction_upload/test_cax_loan_prediction.csv")

# We ave train data and test data with target varibale as "RESULT". We make the target variable to 0/1

train['Loan_Status'].replace( 'Y', 1, inplace = True)
train['Loan_Status'].replace('N', 0, inplace = True)
test['Loan_Status'].replace('Y', 0, inplace = True)
test['Loan_Status'].replace('Y', 1 , inplace = True)

# After Preprocessing we perform over sampling to increase the minorities

from sklearn.utils import resample

# separate minority and majority classes
NOT_FUNDED = train[train.Loan_Status==0]
FUNDED = train[train.Loan_Status==1]

# upsample minority
NOT_FUNDED_upsampled = resample(NOT_FUNDED,
                          replace=True, # sample with replacement
                          n_samples=len(FUNDED), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([FUNDED, NOT_FUNDED_upsampled])
# check new class counts
upsampled.Loan_Status.value_counts()

#seperating target varable from train and test data

y = upsampled.Loan_Status
X = upsampled.drop('Loan_Status', axis=1)
X_test_final = test.drop('Loan_Status', axis=1)
y_test_final = test.Loan_Status

# one hot encoding to convert categorical data to numeric data

X=pd.get_dummies(X) 
X_test_final=pd.get_dummies(X_test_final)

#splitting train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# fitting the model whose parameters are tuned 
xgb3 = XGBClassifier(
 learning_rate =0.2,
 n_estimators=4000,
 max_depth=6,
 min_child_weight=8,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.9,
 reg_alpha=0.1,
 objective= 'reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgb3.fit(X_train , y_train)


# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Diabetes.csv')
dataset.info()
dataset.head()

# Split Data
X = dataset.iloc[:,0:8]
y = dataset['Outcome']

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)



# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Fit Model with train data
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

# Make predictions for test data
y_pred1 = clf.predict(X_test)
crosstab1 = pd.crosstab(y_test, y_pred1, rownames=['Actual Result'], colnames=['Predicted Result'])

# evaluate predictions
accuracy1 = accuracy_score(y_test, y_pred1)

print("Random Forest Model:")
print(crosstab1)
print("Accuracy: %.2f%%" % (accuracy1 * 100.0))


###########################################################################################################################

# Import XGBoost model
from numpy import loadtxt
from xgboost import XGBClassifier
from numpy import nan
from sklearn.metrics import accuracy_score

# Fit Model with train data
XGB = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
XGB.fit(X_train, y_train)
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

# Make predictions for test data
y_pred2 = XGB.predict(X_test)
crosstab2 = pd.crosstab(y_test, y_pred2, rownames=['Actual Result'], colnames=['Predicted Result'])

# evaluate predictions
accuracy2 = accuracy_score(y_test, y_pred2)

print("")
print("XGBoost model:")
print(crosstab2)
print("Accuracy: %.2f%%" % (accuracy2 * 100.0))

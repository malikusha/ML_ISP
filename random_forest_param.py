import pandas as pd
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import parallel_backend
from sklearn.model_selection import StratifiedShuffleSplit

rfc = RandomForestClassifier()
param_grid = {
    'n_estimators': [1000],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 4, 7, 10],
    'max_depth': [25, 30, 37, 50]
}
# max_depth = 10, 30, 50
CV_rfc_letters = GridSearchCV(estimator=rfc, param_grid=param_grid)
# max_depth = 25, 30, 37, 50
CV_rfc_bank = GridSearchCV(estimator=rfc, param_grid=param_grid)
# max_depth = 25, 30, 37, 50
CV_rfc_mat = GridSearchCV(estimator=rfc, param_grid=param_grid)
# max_depth = 25, 30, 37, 50
CV_rfc_party = GridSearchCV(estimator=rfc, param_grid=param_grid)

# Letter Recognition Dataset
X_letters = pd.read_csv('datasets/X_letters.csv', sep=',',header=None)
y_letters = pd.read_csv('datasets/Y_letters.csv', sep=',',header=None)

# Bank clients Dataset
X_bank = pd.read_csv('datasets/X_bank.csv', sep=',',header=None)
y_bank = pd.read_csv('datasets/y_bank.csv', sep=',',header=None)

for column_X in X_bank.columns:
    if X_bank[column_X].dtype == type(object):
        le = preprocessing.LabelEncoder()
        X_bank[column_X] = le.fit_transform(X_bank[column_X])

# Mat Dataset
X_mat = pd.read_csv('datasets/X.csv', sep=',',header=None)
y_mat = pd.read_csv('datasets/y.csv', sep=',',header=None)

# Congressional Dataset
X_party = pd.read_csv('datasets/X_party.csv', sep=',',header=None)
y_party = pd.read_csv('datasets/y_party.csv', sep=',',header=None)

for column_X in X_party.columns:
    if X_party[column_X].dtype == type(object):
        le = preprocessing.LabelEncoder()
        X_party[column_X] = le.fit_transform(X_party[column_X])

# CV_rfc_letters.fit(X_letters, y_letters.values.ravel())
# print("Letter Dataset:")
# print("The best parameters are %s with a score of %0.2f"
#       % (CV_rfc_letters.best_params_, CV_rfc_letters.best_score_))

CV_rfc_bank.fit(X_bank, y_bank.values.ravel())
print("Bank Dataset:")
print("The best parameters are %s with a score of %0.2f"
     % (CV_rfc_bank.best_params_, CV_rfc_bank.best_score_))

# CV_rfc_mat.fit(X_mat, y_mat.values.ravel())
# print("Mat Dataset:")
# print("The best parameters are %s with a score of %0.2f"
#      % (CV_rfc_mat.best_params_, CV_rfc_mat.best_score_))

# CV_rfc_party.fit(X_party, y_party.values.ravel())
# print("Congressional Dataset:")
# print("The best parameters are %s with a score of %0.2f"
#      % (CV_rfc_party.best_params_, CV_rfc_party.best_score_))




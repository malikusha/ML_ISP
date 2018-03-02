import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm, preprocessing
from sklearn.externals.joblib import parallel_backend

# Bank clients Dataset
X_bank = pd.read_csv('datasets/X_bank.csv', sep=',',header=None)
y_bank = pd.read_csv('datasets/y_bank.csv', sep=',',header=None)

for column_X in X_bank.columns:
    if X_bank[column_X].dtype == type(object):
        le = preprocessing.LabelEncoder()
        X_bank[column_X] = le.fit_transform(X_bank[column_X])

# preprocessing.scale(X_bank)
# scaler = StandardScaler()
# X = scaler.fit_transform(X_bank)
# preprocessing.scale(X)

####
C_range = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = dict(gamma=gammas, C=C_range)
####

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
with parallel_backend('threading'):
    grid.fit(X_bank, y_bank.values.ravel())

print("Bank clients Dataset")
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# Letter Recognition Dataset
X_letters = pd.read_csv('datasets/X_letters.csv', sep=',',header=None)
y_letters = pd.read_csv('datasets/Y_letters.csv', sep=',',header=None)

for column_X in X_letters.columns:
    if X_letters[column_X].dtype == type(object):
        le = preprocessing.LabelEncoder()
        X_letters[column_X] = le.fit_transform(X_letters[column_X])

# preprocessing.scale(X_letters)
# scaler_letters = StandardScaler()
# X_letters_x = scaler_letters.fit_transform(X_letters)
# preprocessing.scale(X_letters_x)

cv_letters = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid_letters = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv_letters)
with parallel_backend('threading'):
    grid_letters.fit(X_letters, y_letters.values.ravel())

print("Letter Recognition Dataset")
print("The best parameters are %s with a score of %0.2f"
      % (grid_letters.best_params_, grid_letters.best_score_))


# Mat Dataset
X_mat = pd.read_csv('datasets/X.csv', sep=',',header=None)
y_mat = pd.read_csv('datasets/y.csv', sep=',',header=None)

# preprocessing.scale(X_mat)
# scaler_mat = StandardScaler()
# X_mat_x = scaler_letters.fit_transform(X_mat)
# preprocessing.scale(X_mat_x)

cv_mat = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid_mat = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv_mat)
with parallel_backend('threading'):
    grid_mat.fit(X_mat, y_mat.values.ravel())

print("Mat Dataset")
print("The best parameters are %s with a score of %0.2f"
      % (grid_mat.best_params_, grid_mat.best_score_))


# Congressional Dataset
X_party = pd.read_csv('datasets/X_party.csv', sep=',',header=None)
y_party = pd.read_csv('datasets/y_party.csv', sep=',',header=None)

for column_X in X_party.columns:
    if X_party[column_X].dtype == type(object):
        le = preprocessing.LabelEncoder()
        X_party[column_X] = le.fit_transform(X_party[column_X])

preprocessing.scale(X_party)
scaler_party = StandardScaler()
X_party_x = scaler_party.fit_transform(X_party)
preprocessing.scale(X_party_x)

cv_party = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid_party = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv_party)
with parallel_backend('threading'):
    grid_party.fit(X_party_x, y_party.values.ravel())

print("Congressional Dataset")
print("The best parameters are %s with a score of %0.2f"
      % (grid_party.best_params_, grid_party.best_score_))

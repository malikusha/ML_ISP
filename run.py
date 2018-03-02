import scipy.io
import numpy as np
import pandas as pd
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from multiscorer import MultiScorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

clf = svm.SVC(kernel='linear', C=1)
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
scoring = ['precision_macro', 'recall_macro']

# Mat Dataset
X = pd.read_csv('datasets/X.csv', sep=',',header=None)
y = pd.read_csv('datasets/y.csv', sep=',',header=None)

# Random forest
scorer = MultiScorer({
    'Accuracy' : (accuracy_score, {}),
    'Precision' : (precision_score, {'pos_label': 3, 'average':'macro'}),
    'Recall' : (recall_score, {'pos_label': 3, 'average':'macro'}),
    'F-score': (f1_score, {'pos_label': 3, 'average':'macro'}),
    'MCC': (matthews_corrcoef, {})
})

cross_val_score(rf, X, y.values.ravel(), scoring=scorer, cv=10)
results = scorer.get_results()

print('Random Forest Scores - Mat Dataset')
for metric_name in results.keys():
    average_score = np.average(results[metric_name])
    print('%s : %f' % (metric_name, average_score))

# Letter Recognition Dataset
X_letters = pd.read_csv('datasets/X_letters.csv', sep=',',header=None)
y_letters = pd.read_csv('datasets/Y_letters.csv', sep=',',header=None)

# Random forest
scorer_letters = MultiScorer({
    'Accuracy' : (accuracy_score, {}),
    'Precision' : (precision_score, {'pos_label': 3, 'average':'macro'}),
    'Recall' : (recall_score, {'pos_label': 3, 'average':'macro'}),
    'F-score': (f1_score, {'pos_label': 3, 'average':'macro'}),
    'MCC': (matthews_corrcoef, {})
})

cross_val_score(rf, X_letters, y_letters.values.ravel(), scoring=scorer_letters, cv=10)
results_letters = scorer_letters.get_results()

print('Random Forest Scores - Letter recognition dataset')
for metric_name in results_letters.keys():
    average_score = np.average(results_letters[metric_name])
    print('%s : %f' % (metric_name, average_score))

# Bank Clients Dataset
X_bank = pd.read_csv('datasets/X_bank.csv', sep=',',header=None)
y_bank = pd.read_csv('datasets/y_bank.csv', sep=',',header=None)

for column_X in X_bank.columns:
    if X_bank[column_X].dtype == type(object):
        le = preprocessing.LabelEncoder()
        X_bank[column_X] = le.fit_transform(X_bank[column_X])

# Random forest
scorer_bank = MultiScorer({
    'Accuracy' : (accuracy_score, {}),
    'Precision' : (precision_score, {'pos_label': 3, 'average':'macro'}),
    'Recall' : (recall_score, {'pos_label': 3, 'average':'macro'}),
    'F-score': (f1_score, {'pos_label': 3, 'average':'macro'}),
    'MCC': (matthews_corrcoef, {})
})

cross_val_score(rf, X_bank, y_bank.values.ravel(), scoring=scorer_bank, cv=10)
results_bank = scorer_bank.get_results()

print('Random Forest Scores - Bank clients dataset')
for metric_name in results_bank.keys():
    average_score = np.average(results_bank[metric_name])
    print('%s : %f' % (metric_name, average_score))


# Congressional Dataset
X_party = pd.read_csv('datasets/X_party.csv', sep=',',header=None)
y_party = pd.read_csv('datasets/y_party.csv', sep=',',header=None)

for column_X in X_party.columns:
    if X_party[column_X].dtype == type(object):
        le = preprocessing.LabelEncoder()
        X_party[column_X] = le.fit_transform(X_party[column_X])

# Random forest
scorer_party = MultiScorer({
    'Accuracy' : (accuracy_score, {}),
    'Precision' : (precision_score, {'pos_label': 3, 'average':'macro'}),
    'Recall' : (recall_score, {'pos_label': 3, 'average':'macro'}),
    'F-score': (f1_score, {'pos_label': 3, 'average':'macro'}),
    'MCC': (matthews_corrcoef, {})
})

cross_val_score(rf, X_party, y_party.values.ravel(), scoring=scorer_party, cv=10)
results_party = scorer_party.get_results()

print('Random Forest Scores - Congressional dataset')
for metric_name in results_party.keys():
    average_score = np.average(results_party[metric_name])
    print('%s : %f' % (metric_name, average_score))










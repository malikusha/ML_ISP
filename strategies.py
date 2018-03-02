import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# Convert mat to csv
# data = scipy.io.loadmat("g50c.mat")
# for i in data:
# 	if '__' not in i and 'readme' not in i:
# 		np.savetxt(("matToCSV/"+i+".csv"),data[i],delimiter=',')

X = pd.read_csv('matToCSV/X.csv', sep=',',header=None)
y = pd.read_csv('matToCSV/y.csv', sep=',',header=None)

# # Random Forest Scores
# scores_rf = cross_val_score(rf, X, y.values.ravel(), cv = 10, scoring='precision')
# recall_score_rf = cross_validate(rf, X, y.values.ravel(), scoring= scoring, cv = 10, return_train_score=False)
# print('Random Forest Scores - Mat Dataset')
# print('Mean score : ', np.mean(scores_rf))
# print('Score variance : ', np.var(scores_rf))
# print('Accuracy: %0.2f (+/- %0.2f)' % (scores_rf.mean(), scores_rf.std() * 2))
# rc = np.mean(recall_score_rf['test_recall_macro'])
# print("Recall score : ", np.mean(recall_score_rf['test_recall_macro']))
# print("F-score : ", 2 * (np.mean(scores_rf) * np.mean(recall_score_rf['test_recall_macro'])) / (np.mean(scores_rf) + np.mean(recall_score_rf['test_recall_macro'])))
#
# # svc scores
# scores_clf = cross_val_score(clf, X, y.values.ravel(), cv = 10, scoring='precision')
# recall_score_clf = cross_validate(clf, X, y.values.ravel(), scoring= scoring, cv = 10, return_train_score=False)
# print('SVC - Mat Dataset')
# print('Mean score : ', np.mean(scores_clf))
# print('Score variance : ', np.var(scores_clf))
# print('Accuracy: %0.2f (+/- %0.2f)' % (scores_clf.mean(), scores_clf.std() * 2))
# print("Recall score : ", np.mean(recall_score_clf['test_recall_macro']))
# print("F-score : ", 2 * (np.mean(scores_clf) * np.mean(recall_score_clf['test_recall_macro'])) / (np.mean(scores_clf) + np.mean(recall_score_clf['test_recall_macro'])) )

# 10-k fold cross validation

# kf = KFold(n_splits=10) #n_splits indicate the number of folds
# kf.get_n_splits(X)
#
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     X_train.shape, y_train.shape
#     X_test.shape, y_test.shape
#     clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train.values.ravel())
#     print(clf.score(X_test, y_test))


# Making just one split

# X.shape, y.shape
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#
# X_train.shape, y_train.shape
# X_test.shape, y_test.shape
# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train.values.ravel())
# print(clf.score(X_test, y_test))


clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y.values.ravel(), cv = 10, scoring='precision')

scoring = ['precision_macro', 'recall_macro']
recall_score = cross_validate(clf, X, y.values.ravel(), scoring= scoring, cv = 10, return_train_score=False)
print('Mean score : ', np.mean(scores))
print('Score variance : ', np.var(scores))
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print("Recall score : ", np.mean(recall_score['test_recall_macro']))










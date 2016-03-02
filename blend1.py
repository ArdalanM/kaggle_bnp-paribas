__author__ = 'Ardalan'

CODE_FOLDER = "/home/ardalan/Documents/kaggle/bnp/"
# CODE_FOLDER = "/home/arda/Documents/kaggle/bnp/"

import os, sys, time, re, zipfile, pickle, operator
if os.getcwd() != CODE_FOLDER: os.chdir(CODE_FOLDER)
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import svm

def reshapePrediction(ypredproba):
    result = None
    if len(ypredproba.shape) > 1:
        if ypredproba.shape[1] == 1: result = ypredproba[:, 0]
        if ypredproba.shape[1] == 2: result = ypredproba[:, 1]
    else:
        result = ypredproba.ravel()
    return result
def eval_func(ytrue, ypredproba):
    return metrics.log_loss(ytrue, reshapePrediction(ypredproba))
def loadFileinZipFile(zip_filename, filename, dtypes=None, parsedate = None, password=None, **kvargs):
    """
    Load file to dataframe.
    """
    with zipfile.ZipFile(zip_filename, 'r') as myzip:
        if password:
            myzip.setpassword(password)

        if parsedate:
            return pd.read_csv(myzip.open(filename), sep=',', parse_dates=parsedate, dtype=dtypes, **kvargs)
        else:
            return pd.read_csv(myzip.open(filename), sep=',', dtype=dtypes, **kvargs)
def printResults(dic_logs):
        l_train_logloss = dic_logs['train_error']
        l_val_logloss = dic_logs['val_error']

        string = ("logTRVAL_{0:.4f}|{1:.4f}".format(np.mean(l_train_logloss), np.mean(l_val_logloss)))
        return string
def KerasClassWeight(Y_vec):
    # Find the weight of each class as present in y.
    # inversely proportional to the number of samples in the class
    recip_freq = 1. / np.bincount(Y_vec)
    weight = recip_freq / np.mean(recip_freq)
    dic_w = {index:weight_value for index, weight_value in enumerate(weight)}
    return dic_w
def saveDicLogs(dic_logs, filename):

    folder2save = CODE_FOLDER + 'diclogs/'
    if os.path.exists(folder2save + filename):
        print('file exist !')
        raise BrokenPipeError
    else:
        # proof_code = open(CODE_FOLDER + 'code/main_CV.py', 'rb').read()
        # open(folder2save + filename + '.py', 'wb').write(proof_code)
        pickle.dump(dic_logs, open(folder2save + filename, 'wb'))
    return


folder = CODE_FOLDER + 'diclogs/'
import glob
l_filenames = glob.glob1(folder, '*[0-9].p') ; print(l_filenames)


l_X = []
l_X_test=[]
l_Y = []
for filename in l_filenames:
    filename = filename[:-2]
    print(filename)

    dic_log = pickle.load(open(folder + filename + '.p','rb'))

    pd_temp = pd.read_csv(folder + filename + '.csv')
    test_idx = pd_temp['ID'].values.astype(int)

    l_X.append(np.hstack(dic_log['ypredproba']))
    l_X_test.append(pd_temp['PredictedProb'].values)
    l_Y.append(np.hstack(dic_log['yval']))

X = np.array(l_X).T
X_test = np.array(l_X_test).T
Y = np.array(l_Y).T.mean(1).astype(int)



#params10len(folds)
n_folds = 10
test_size = .2
nthread = 8
seed = 123





# clf = linear_model.SGDClassifier(loss='log', penalty='l2')
# clf = linear_model.LogisticRegression(C=.1, penalty='l2')
# clf = linear_model.LinearRegression(normalize=True)
# clf = linear_model.LassoCV(eps=.05, n_alphas=200)
clf = ensemble.RandomForestRegressor(n_estimators=50, max_features=1., max_depth=5, n_jobs=nthread)
# clf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=nthread)

skf = cross_validation.StratifiedShuffleSplit(Y, n_iter=n_folds, test_size=test_size, random_state=seed)

for fold_indice, (tr_idx, te_idx) in enumerate(skf):

    xtrain = X[tr_idx]
    ytrain = Y[tr_idx]
    xval = X[te_idx]
    yval = Y[te_idx]

    clf.fit(xtrain, ytrain)

    valpred = reshapePrediction(clf.predict(xval))
    trainpred = reshapePrediction(clf.predict(xtrain))


    train_err = eval_func(ytrain, trainpred)
    val_err = eval_func(yval, valpred)

    print(train_err, val_err)

clf.fit(X,Y)
testpred = clf.predict(X_test)

pd_submission = pd.DataFrame({'ID':test_idx, 'PredictedProb':testpred})
pd_submission.to_csv(CODE_FOLDER + 'diclogs/' + 'test' + '.csv', index=False)



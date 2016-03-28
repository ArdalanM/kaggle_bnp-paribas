__author__ = 'Ardalan'

CODE_FOLDER = "/home/ardalan/Documents/kaggle/bnp/"
# CODE_FOLDER = "/home/arda/Documents/kaggle/bnp/"

import os, sys, time, re, zipfile, pickle, operator
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import svm
from sklearn import calibration

def clipProba(ypredproba):
    """
    Taking list of proba and returning a list of clipped proba
    :param ypredproba:
    :return: ypredproba clipped
    """""

    ypredproba = np.where(ypredproba <= 0., 1e-5 , ypredproba)
    ypredproba = np.where(ypredproba >= 1., 1.-1e-5, ypredproba)

    return ypredproba

def reshapePrediction(ypredproba):
    result = None
    if len(ypredproba.shape) > 1:
        if ypredproba.shape[1] == 1: result = ypredproba[:, 0]
        if ypredproba.shape[1] == 2: result = ypredproba[:, 1]
    else:
        result = ypredproba.ravel()

    result = clipProba(result)

    return result

def eval_func(ytrue, ypredproba):

    return metrics.log_loss(ytrue, ypredproba)

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

def LoadParseData(filename):

    cont_var = ['ID', 'target', 'v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13',
                'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v23', 'v25', 'v26', 'v27', 'v28',
                'v29', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v40', 'v41', 'v42', 'v43',
                'v44', 'v45', 'v46', 'v48', 'v49', 'v50', 'v51', 'v53', 'v54', 'v55', 'v57', 'v58', 'v59',
                'v60', 'v61', 'v62', 'v63', 'v64', 'v65', 'v67', 'v68', 'v69', 'v70', 'v72', 'v73', 'v76',
                'v77', 'v78', 'v80', 'v81', 'v82', 'v83', 'v84', 'v85', 'v86', 'v87', 'v88', 'v89', 'v90',
                'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98', 'v99', 'v100', 'v101', 'v102', 'v103', 'v104',
                'v105', 'v106', 'v108', 'v109', 'v111', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v120',
                'v121', 'v122', 'v123', 'v124', 'v126', 'v127', 'v128', 'v129', 'v130', 'v131']



    data_name = filename.split('_')[0]
    pd_data = pd.read_hdf(CODE_FOLDER + "data/" + filename)


    pd_train = pd_data[pd_data.target >= 0]
    pd_test = pd_data[pd_data.target == -1]

    Y = pd_train['target'].values.astype(int)
    test_idx = pd_test['ID'].values.astype(int)

    X = np.array(pd_train.drop(['ID', 'target'],1))
    X_test = np.array(pd_test.drop(['ID','target'], 1))

    return X, Y, X_test, test_idx, pd_data, data_name

def models():
    params = {'n_jobs':nthread,'random_state':seed,'class_weight':None}

    # extra = ensemble.ExtraTreesClassifier(n_estimators=1000,max_features='auto',criterion= 'entropy',min_samples_split= 2, max_depth= None, min_samples_leaf= 1, **params)
    # extra1 = ensemble.ExtraTreesClassifier(n_estimators=1000,max_features=60,criterion= 'gini',min_samples_split= 4, max_depth= 40, min_samples_leaf= 2, **params)

    # rf = ensemble.RandomForestClassifier(n_estimators=1000,max_features= 'auto',criterion= 'gini',min_samples_split= 2, max_depth= None, min_samples_leaf= 1, **params)
    # rf1 = ensemble.RandomForestClassifier(n_estimators=1000,max_features=60,criterion= 'entropy',min_samples_split= 4, max_depth= 40, min_samples_leaf= 2, **params)

    clfs = (
        (D1, calibration.CalibratedClassifierCV(ensemble.RandomForestClassifier(n_jobs=nthread),cv=3)),
        (D1, ensemble.RandomForestClassifier(n_jobs=nthread))
)
    for clf in clfs:
        yield clf

def printResults(dic_logs):
        l_train_logloss = dic_logs['train_error']
        l_val_logloss = dic_logs['val_error']

        string = ("logTRVAL_{0:.4f}|{1:.4f}".format(np.mean(l_train_logloss), np.mean(l_val_logloss)))
        return string

def saveDicLogs(dic_logs, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(dic_logs, f, protocol=pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        pass

# Params
#General params
CV = 'strat'
STORE = False
DOTEST = False
n_folds = 5
test_size = 0.20
nthread = 12
seed = 123



X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D1_[LE-cat]_[NAmean].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D1 = (X, Y, X_test,test_idx, data_name)

# X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D2_[LE-cat]_[NA-999].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D2 = (X, Y, X_test,test_idx, data_name)
#
X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D3_[OH30]_[NAmean].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D3 = (X, Y, X_test,test_idx, data_name)
#
X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D4_[OH30]_[NA-999].p')
D4 = (X, Y, X_test,test_idx, data_name)
#
X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D5_[OnlyCont]_[NAmean].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D5 = (X, Y, X_test,test_idx, data_name)
#
X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D6_[OnlyCatLE].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D6 = (X, Y, X_test,test_idx, data_name)
#
X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D7_[OnlyCatOH].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D7 = (X, Y, X_test,test_idx, data_name)
#
# X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D8_[ColsRemoved]_[Namean]_[OH].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D8 = (X, Y, X_test,test_idx, data_name)
#
X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D9_[ColsRemoved]_[NA-999]_[LE-cat].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D9 = (X, Y, X_test,test_idx, data_name)
#
X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D10_[ColsRemoved]_[NA-999]_[OH].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D10 = (X, Y, X_test,test_idx, data_name)


if CV == 'strat': skf = cross_validation.StratifiedShuffleSplit(Y, n_iter=n_folds, test_size=test_size, random_state=seed)
if CV == 'random': skf = cross_validation.ShuffleSplit(len(Y), n_iter=n_folds, test_size=test_size, random_state=seed)

clfs = models()

##################################################################################
# CV
##################################################################################



# clf =  calibration.CalibratedClassifierCV(ensemble.RandomForestClassifier(),cv=skf)




for clf_indice, data_clf in enumerate(clfs):

    print('-' * 50)
    print("Classifier [%i]" % clf_indice)
    X = data_clf[0][0] ; print(X.shape)
    Y = data_clf[0][1]
    X_test = data_clf[0][2]
    test_idx = data_clf[0][3]

    clf = data_clf[1]
    print(clf)
    clf_class_name = clf.__class__.__name__
    clf_name = clf_class_name[:3]
    data_name = data_clf[0][4]

    dic_logs = {'name':clf_name, 'fold':[],'ypredproba':[],'yval':[],
                'params':None,'prepro':None,'best_epoch':[],'best_val_metric':[],
                'train_error': [], 'val_error':[]}


    filename = '{}_{}_{}f_CV{}'.format(clf_name, data_name,X.shape[1],n_folds)

    for fold_indice, (tr_idx, te_idx) in enumerate(skf):
        print("Fold [%i]" % fold_indice)
        xtrain = X[tr_idx]
        ytrain = Y[tr_idx]
        xval = X[te_idx]
        yval = Y[te_idx]

        if clf_class_name == 'XGBClassifier':
            dic_logs['params'] = clf.get_params()
            added_params = ["_{}".format('-'.join(list(map(lambda x: x[:3] ,clf.objective.split(':'))))),
                            "_d{}".format(clf.max_depth),
                            "_lr{}".format(clf.learning_rate)
                            ]

            # clf.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric=xgb_accuracy, early_stopping_rounds=100, verbose=True)

            dic_logs['best_epoch'].append(clf.best_iteration)
            dic_logs['best_val_metric'].append(clf.best_score)

        elif clf_class_name == 'NN':
            logs = clf.fit(xtrain, ytrain, eval_set=(xval, yval), show_accuracy=True)

            dic_logs['params'] = clf.model.get_config()
            dic_logs['best_epoch'].append(np.argmin(logs.history['val_loss']))
            dic_logs['best_val_metric'].append(np.min(logs.history['val_loss']))

        else:
            dic_logs['params'] = clf.get_params()
            clf.fit(xtrain, ytrain)

        #This is a list (not matrix or else)
        train_pred = clipProba(reshapePrediction(clf.predict_proba(xtrain)))
        val_pred = clipProba(reshapePrediction(clf.predict_proba(xval)))

        #metrics
        train_error = eval_func(ytrain, train_pred)
        val_error = eval_func(yval, val_pred)

        print("train/val error: [{0:.4f}|{1:.4f}]".format(train_error, val_error))
        print(metrics.confusion_matrix(yval, val_pred.round()))

        dic_logs['fold'].append(fold_indice)
        dic_logs['ypredproba'].append(val_pred)
        dic_logs['yval'].append(yval)
        dic_logs['train_error'].append(train_error)
        dic_logs['val_error'].append(val_error)


    string_result = printResults(dic_logs)
    # filename += "{}_{}".format(''.join(added_params),string_result)

    print(string_result)
    print(filename)

    if STORE: saveDicLogs(dic_logs, CODE_FOLDER + 'diclogs/' + filename + '.p')

    if DOTEST:
        print('Test prediction...')
        if clf_class_name == 'XGBClassifier':
            clf.n_estimators = np.mean(dic_logs['best_epoch']).astype(int)
            print("Best n_estimators set to: ", clf.n_estimators)
            clf.fit(X, Y)
        elif clf_class_name == 'NN':
            clf.fit(X, Y)
        else:
            clf.fit(X, Y)

        ypredproba = clipProba(reshapePrediction(clf.predict_proba(X_test)))

        pd_submission = pd.DataFrame({'ID':test_idx, 'PredictedProb':ypredproba})
        pd_submission.to_csv(CODE_FOLDER + 'diclogs/' + filename + '.csv', index=False)


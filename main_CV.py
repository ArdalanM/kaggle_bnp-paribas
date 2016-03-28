__author__ = 'Ardalan'

CODE_FOLDER = "/home/ardalan/Documents/kaggle/bnp/"
# CODE_FOLDER = "/home/arda/Documents/kaggle/bnp/"

import os, sys, time, re, zipfile, pickle, operator
import pandas as pd
import numpy as np

from xgboost import XGBClassifier, XGBRegressor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import svm
from sklearn import calibration

from keras.preprocessing import text, sequence
from keras.optimizers import *
from keras.models import Sequential
from keras.utils import np_utils

from keras.layers import core, embeddings, recurrent, advanced_activations, normalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

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

def CreateDataFrameFeatureImportance(model, pd_data):
    dic_fi = model.get_fscore()
    df = pd.DataFrame(dic_fi.items(), columns=['feature', 'fscore'])
    df['col_indice'] = df['feature'].apply(lambda r: r.replace('f','')).astype(int)
    df['feat_name'] = df['col_indice'].apply(lambda r: pd_data.columns[r])
    return df.sort('fscore', ascending=False)

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

def xgb_accuracy(ypred, dtrain):
        ytrue = dtrain.get_label().astype(int)

        ypred = np.where(ypred <= 0., 1e-5 , ypred)
        ypred = np.where(ypred >= 1., 1.-1e-5, ypred)


        return 'logloss', metrics.log_loss(ytrue, ypred)

class NN():

    def __init__(self, input_dim, output_dim,
                 hidden_units=(512, 256),
                 activation=('tanh', 'tanh'),
                 dropout=(0.3, 0.3), l2_regularizer=(1e-4, 1e-4),
                 loss="categorical_crossentropy",
                 optimizer=RMSprop(lr=0.001, tho=0.9),
                 batch_size=512,
                 nb_epoch=3, early_stopping_epoch=None,
                 verbose=1, class_mode='binary'):

        self.name = 'NN'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.l2_regularizer = l2_regularizer

        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.class_mode = class_mode

        self.model = None
        self.esr = early_stopping_epoch

        self.params = {'input_dim': self.input_dim,
                       'output_dim': self.output_dim,
                       'hidden_units': self.hidden_units,
                       'activation': self.activation,
                       'dropout': self.dropout,
                       'l2_regularizer': self.l2_regularizer,
                       'loss': self.loss,
                       'optimizer': self.optimizer,
                       'batch_size': self.batch_size,
                       'nb_epoch': self.nb_epoch,
                       'esr': self.esr,
                       'verbose': self.verbose}

    def _build_model(self, input_dim, output_dim, hidden_units, activation, dropout, loss="binary_crossentropy", optimizer=RMSprop(), class_mode='binary'):

        model = Sequential()

        model.add(core.Dense(hidden_units[0], input_dim=input_dim, W_regularizer=None, ))
        model.add(activation[0])
        model.add(normalization.BatchNormalization())
        # model.add(core.Dropout(dropout[0]))

        for i in range(1, len(activation) - 1):
            model.add(core.Dense(hidden_units[i], input_dim=hidden_units[i-1], W_regularizer=None))
            model.add(activation[i])
            model.add(normalization.BatchNormalization())
            # model.add(core.Dropout(dropout[i]))

        model.add(core.Dense(output_dim, input_dim=hidden_units[-1], W_regularizer=None))
        model.add(core.Activation(activation[-1]))
        model.compile(loss=loss, optimizer=optimizer, class_mode=class_mode)

        return model


    def fit(self, X, y, eval_set=None, class_weight=None, show_accuracy=True):


        if self.loss == 'categorical_crossentropy':
            y = np_utils.to_categorical(y)

        if eval_set != None and self.loss == 'categorical_crossentropy':
            eval_set = (eval_set[0], np_utils.to_categorical(eval_set[1]))

        self.model = self._build_model(self.input_dim,self.output_dim,self.hidden_units,self.activation,
                                       self.dropout, self.loss, self.optimizer, self.class_mode)

        if eval_set !=None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=self.esr, verbose=1, mode='min')
            logs = self.model.fit(X, y, self.batch_size, self.nb_epoch, self.verbose, validation_data=eval_set, callbacks=[early_stopping], show_accuracy=True, shuffle=True)
        else:
            logs = self.model.fit(X, y, self.batch_size, self.nb_epoch, self.verbose, show_accuracy=True, shuffle=True)

        return logs

    def predict_proba(self, X):

        prediction = self.model.predict_proba(X, verbose=0)

        return prediction

def models():
    params = {'n_jobs':nthread,'random_state':seed,'class_weight':None}

    # extra = ensemble.ExtraTreesClassifier(n_estimators=1000,max_features='auto',criterion= 'entropy',min_samples_split= 2, max_depth= None, min_samples_leaf= 1, **params)
    # extra1 = ensemble.ExtraTreesClassifier(n_estimators=1000,max_features=60,criterion= 'gini',min_samples_split= 4, max_depth= 40, min_samples_leaf= 2, **params)

    # rf = ensemble.RandomForestClassifier(n_estimators=1000,max_features= 'auto',criterion= 'gini',min_samples_split= 2, max_depth= None, min_samples_leaf= 1, **params)
    # rf1 = ensemble.RandomForestClassifier(n_estimators=1000,max_features=60,criterion= 'entropy',min_samples_split= 4, max_depth= 40, min_samples_leaf= 2, **params)

    # xgb_binlog = XGBClassifier(objective="binary:logistic" ,max_depth=10, learning_rate=0.01, n_estimators=5,nthread=nthread, seed=seed)
    # xgb_reglog = XGBClassifier(objective="reg:logistic", max_depth=10, learning_rate=0.01, n_estimators=5,nthread=nthread, seed=seed)
    # xgb_poi = XGBClassifier(objective="count:poisson", max_depth=10, learning_rate=0.01, n_estimators=5,nthread=nthread, seed=seed)
    # xgb_reglin = XGBClassifier(objective="reg:linear", max_depth=10, learning_rate=0.01, n_estimators=5,nthread=nthread, seed=seed)

    rf_params = {'n_estimators':850,'max_features':60,'criterion':'entropy','min_samples_split': 4,'max_depth': 40, 'min_samples_leaf': 2, 'n_jobs': -1}

    # #NN params
    # nb_epoch = 3
    # batch_size = 128
    # esr = 402
    #
    # param1 = {
    #     'hidden_units': (1024, 1024),
    #     'activation': (advanced_activations.PReLU(),advanced_activations.PReLU(),core.activations.sigmoid),
    #     'dropout': (0., 0.), 'optimizer': RMSprop(), 'nb_epoch': nb_epoch,
    # }
    clfs = [
    # (D5, NN(input_dim=D5[0].shape[1], output_dim=1, batch_size=batch_size, early_stopping_epoch=esr, verbose=2, loss='binary_crossentropy', class_mode='binary', **param1))]

    # [D1, linear_model.LogisticRegression(class_weight='balanced', random_state=seed, n_jobs=nthread)],
    # [D3, linear_model.LogisticRegression(class_weight='balanced', random_state=seed, n_jobs=nthread)],
    # [D4, linear_model.LogisticRegression(class_weight='balanced', random_state=seed, n_jobs=nthread)],
    # [D5, linear_model.LogisticRegression(class_weight='balanced', random_state=seed, n_jobs=nthread)],
    # [D6, linear_model.LogisticRegression(class_weight='balanced', random_state=seed, n_jobs=nthread)],
    # [D7, linear_model.LogisticRegression(class_weight='balanced', random_state=seed, n_jobs=nthread)],
    # [D9, linear_model.LogisticRegression(class_weight='balanced', random_state=seed, n_jobs=nthread)],
    # [D10, linear_model.LogisticRegression(class_weight='balanced', random_state=seed, n_jobs=nthread)],


    # (D1, ensemble.ExtraTreesRegressor(n_estimators=1000, criterion='mse', max_features=100, max_depth=50, random_state=seed, n_jobs=nthread)),
    # (D3, ensemble.ExtraTreesRegressor(n_estimators=1000, criterion='mse', max_features=100, max_depth=50, random_state=seed, n_jobs=nthread)),
    # (D4, ensemble.ExtraTreesRegressor(n_estimators=1000, criterion='mse', max_features=100, max_depth=50, random_state=seed, n_jobs=nthread)),
    # (D5, ensemble.ExtraTreesRegressor(n_estimators=1000, criterion='mse', max_features=100, max_depth=50, random_state=seed, n_jobs=nthread)),
    # (D6, ensemble.RandomForestRegressor(n_estimators=1000, criterion='mse', max_features=100, max_depth=50, random_state=seed, n_jobs=nthread)),
    # (D7, ensemble.ExtraTreesRegressor(n_estimators=1000, criterion='mse', max_features=100, max_depth=50, random_state=seed, n_jobs=nthread)),
    # (D9, ensemble.ExtraTreesRegressor(n_estimators=1000, criterion='mse', max_features=100, max_depth=50, random_state=seed, n_jobs=nthread)),
    # (D10, ensemble.ExtraTreesRegressor(n_estimators=1000, criterion='mse', max_features=100, max_depth=50, random_state=seed, n_jobs=nthread)),


#    (D1, XGBRegressor(objective="reg:linear", max_depth=10, learning_rate=0.05, n_estimators=10000,nthread=nthread, seed=seed)),
#    (D3, XGBRegressor(objective="reg:linear", max_depth=10, learning_rate=0.05, n_estimators=10000,nthread=nthread, seed=seed)),
    (D4, XGBRegressor(objective="reg:linear", max_depth=10, learning_rate=0.05, n_estimators=10000,nthread=nthread, seed=seed)),
    (D5, XGBRegressor(objective="reg:linear", max_depth=10, learning_rate=0.05, n_estimators=10000,nthread=nthread, seed=seed)),
    (D6, XGBRegressor(objective="reg:linear", max_depth=10, learning_rate=0.05, n_estimators=10000,nthread=nthread, seed=seed)),
    (D7, XGBRegressor(objective="reg:linear", max_depth=10, learning_rate=0.05, n_estimators=10000,nthread=nthread, seed=seed)),
    (D9, XGBRegressor(objective="reg:linear", max_depth=10, learning_rate=0.05, n_estimators=10000,nthread=nthread, seed=seed)),
    (D10, XGBRegressor(objective="reg:linear", max_depth=10, learning_rate=0.05, n_estimators=10000,nthread=nthread, seed=seed)),

    # (D1, XGBClassifier(objective="count:poisson", max_depth=10, learning_rate=0.01, n_estimators=10000,nthread=nthread, seed=seed)),
    # (D3, XGBClassifier(objective="count:poisson", max_depth=10, learning_rate=0.01, n_estimators=10000,nthread=nthread, seed=seed)),
    # (D4, XGBClassifier(objective="count:poisson", max_depth=10, learning_rate=0.01, n_estimators=10000,nthread=nthread, seed=seed)),
    # (D5, XGBClassifier(objective="count:poisson", max_depth=10, learning_rate=0.01, n_estimators=10000,nthread=nthread, seed=seed)),
    # (D6, XGBClassifier(objective="count:poisson", max_depth=10, learning_rate=0.01, n_estimators=10000,nthread=nthread, seed=seed)),
    # (D7, XGBClassifier(objective="count:poisson", max_depth=10, learning_rate=0.01, n_estimators=10000,nthread=nthread, seed=seed)),
    # (D9, XGBClassifier(objective="count:poisson", max_depth=10, learning_rate=0.01, n_estimators=10000,nthread=nthread, seed=seed)),
    # (D10, XGBClassifier(objective="count:poisson", max_depth=10, learning_rate=0.01, n_estimators=10000,nthread=nthread, seed=seed)),

    # (D1, XGBClassifier(objective="reg:linear", max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D3, XGBClassifier(objective="reg:linear", max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D4, XGBClassifier(objective="reg:linear", max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D5, XGBClassifier(objective="reg:linear", max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D6, XGBClassifier(objective="reg:linear", max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D7, XGBClassifier(objective="reg:linear", max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D9, XGBClassifier(objective="reg:linear", max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D10, XGBClassifier(objective="reg:linear", max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    #
    # (D1, XGBClassifier(objective="binary:logistic" ,max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D3, XGBClassifier(objective="binary:logistic" ,max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D4, XGBClassifier(objective="binary:logistic" ,max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D5, XGBClassifier(objective="binary:logistic" ,max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D6, XGBClassifier(objective="binary:logistic" ,max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D7, XGBClassifier(objective="binary:logistic" ,max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D9, XGBClassifier(objective="binary:logistic" ,max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    # (D10, XGBClassifier(objective="binary:logistic" ,max_depth=10, learning_rate=0.01, n_estimators=8,nthread=nthread, seed=seed)),
    ]
    for clf in clfs:
        yield clf

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
    try:
        with open(filename, 'wb') as f:
            pickle.dump(dic_logs, f, protocol=pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        pass

# Params
#General params
CV = 'strat'
STORE = True
DOTEST = True
n_folds = 10
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
X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D3_[OH300]_[NAmean].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D3 = (X, Y, X_test,test_idx, data_name)
#
X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D4_[OH300]_[NA-999].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
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
X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D9 = (X, Y, X_test,test_idx, data_name)
#
X, Y, X_test, test_idx, pd_data, data_name = LoadParseData('D10_[ColsRemoved]_[NA-999]_[OH].p')
X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D10 = (X, Y, X_test,test_idx, data_name)


if CV == 'strat': skf = cross_validation.StratifiedShuffleSplit(Y, n_iter=n_folds, test_size=test_size, random_state=seed)
if CV == 'random': skf = cross_validation.ShuffleSplit(len(Y), n_iter=n_folds, test_size=test_size, random_state=seed)

clfs = models()

##################################################################################
# CV
##################################################################################
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

        if clf_class_name == 'XGBClassifier' or 'XGBRegressor':
            dic_logs['params'] = clf.get_params()
            added_params = ["_{}".format('-'.join(list(map(lambda x: x[:3] ,clf.objective.split(':'))))),
                            "_d{}".format(clf.max_depth),
                            "_lr{}".format(clf.learning_rate)
                            ]

            clf.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric=xgb_accuracy, early_stopping_rounds=200, verbose=True)

            dic_logs['best_epoch'].append(clf.best_iteration)
            dic_logs['best_val_metric'].append(clf.best_score)

        elif clf_class_name == 'NN':
            logs = clf.fit(xtrain, ytrain, eval_set=(xval, yval), show_accuracy=True)

            dic_logs['params'] = clf.model.get_config()
            dic_logs['best_epoch'].append(np.argmin(logs.history['val_loss']))
            dic_logs['best_val_metric'].append(np.min(logs.history['val_loss']))

        elif clf_class_name == "CalibratedClassifierCV":
            if clf.base_estimator.__class__.__name__ == "RandomForestClassifier" or clf.base_estimator.__class__.__name__ == "ExtraTreesClassifier" :
                dic_logs['params'] = clf.get_params()
                added_params = ["_{}".format(clf.base_estimator.__class__.__name__[:3]),
                    "_{}".format(clf.method[:3]),
                    "_{}".format(clf.base_estimator.criterion[:3]),
                    "_md{}".format(clf.base_estimator.max_depth)]
                clf.fit(xtrain,ytrain)

        elif clf_class_name == "RandomForestRegressor" or "ExtraTreesRegressor":
            dic_logs['params'] = clf.get_params()
            added_params = ["_{}".format(clf.criterion[:3]),
                            "_md{}".format(clf.max_depth)]
            clf.fit(xtrain,ytrain)

        else:
            dic_logs['params'] = clf.get_params()
            added_params = [""]
            clf.fit(xtrain, ytrain)

        if hasattr(clf, 'predict_proba'):
            #This is a list (not matrix or else)
            train_pred = clipProba(reshapePrediction(clf.predict_proba(xtrain)))
            val_pred = clipProba(reshapePrediction(clf.predict_proba(xval)))
        elif hasattr(clf, 'predict'):
            #This is a list (not matrix or else)
            train_pred = clipProba(reshapePrediction(clf.predict(xtrain)))
            val_pred = clipProba(reshapePrediction(clf.predict(xval)))

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
    filename += "{}_{}".format(''.join(added_params),string_result)

    print(string_result)
    print(filename)

    if STORE: saveDicLogs(dic_logs, CODE_FOLDER + 'diclogs/' + filename + '.p')

    if DOTEST:

        print('Test prediction...')
        if clf_class_name == 'XGBClassifier' or 'XGBRegressor':
            clf.n_estimators = np.mean(dic_logs['best_epoch']).astype(int)
            print("Best n_estimators set to: ", clf.n_estimators)
            clf.fit(X, Y)

        elif clf_class_name == 'NN':
            clf.fit(X, Y)

        else:
            clf.fit(X, Y)

        if hasattr(clf, 'predict_proba'):
            ypredproba = clipProba(reshapePrediction(clf.predict_proba(X_test)))
        elif hasattr(clf, 'predict'):
            ypredproba = clipProba(reshapePrediction(clf.predict(X_test)))

        output_filename = CODE_FOLDER + 'diclogs/' + filename + '.csv'
        np.savetxt(output_filename, np.vstack((test_idx,ypredproba)).T, delimiter=',',
                   fmt='%i,%.10f'  ,header='ID,PredictedProb', comments="")
    del data_clf

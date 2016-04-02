__author__ = 'Ardalan'

CODE_FOLDER = "/home/ardalan/Documents/kaggle/bnp/"
# CODE_FOLDER = "/home/arda/Documents/kaggle/bnp/"
SAVE_FOLDER = CODE_FOLDER + "/diclogs/stage1/"

import theano.sandbox.cuda
theano.sandbox.cuda.use("cpu")

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


def str_feat_importance(model, col_features):

    dic_fi = model.get_fscore()

    #getting somithing like [(f0, score), (f1, score)]
    importance = [(col_features[int(key[1:])], dic_fi[key]) for key in dic_fi]

    #same but sorted by score
    importance = sorted(importance, key=operator.itemgetter(1), reverse=True)

    sum_importance = np.sum([score for feat, score in importance])

    l=[]
    for rank, (feat, score) in enumerate(importance):
        l.append("{}, {:.3f}, {}\n".format(rank, score / sum_importance, feat))
    return "".join(l)

def LoadParseData(filename):

    data_name = filename.split('_')[0]
    pd_data = pd.read_hdf(CODE_FOLDER + "data/" + filename)
    cols_features = pd_data.drop(['ID', 'target'], 1).columns.tolist()

    pd_train = pd_data[pd_data.target >= 0]
    pd_test = pd_data[pd_data.target == -1]

    Y = pd_train['target'].values.astype(int)
    test_idx = pd_test['ID'].values.astype(int)

    X = np.array(pd_train.drop(['ID', 'target'],1))
    X_test = np.array(pd_test.drop(['ID','target'], 1))

    return X, Y, X_test, test_idx, pd_data, data_name, cols_features

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

    extra_params_kaggle_cla = {'n_estimators':1200,'max_features':30,'criterion':'entropy',
                           'min_samples_leaf': 2, 'min_samples_split': 2,'max_depth': 30,
                           'min_samples_leaf': 2, 'n_jobs':nthread, 'random_state':seed}

    extra_params_kaggle_reg = {'n_estimators':1200,'max_features':30,'criterion':'mse',
                           'min_samples_leaf': 2, 'min_samples_split': 2,'max_depth': 30,
                           'min_samples_leaf': 2, 'n_jobs':nthread, 'random_state':seed}


    xgb_reg = {'objective':'reg:linear', 'max_depth': 11, 'learning_rate':0.01, 'subsample':.9,
           'n_estimators':10000, 'colsample_bytree':0.45, 'nthread':nthread, 'seed':seed}

    xgb_cla = {'objective':'binary:logistic', 'max_depth': 11, 'learning_rate':0.01, 'subsample':.9,
           'n_estimators':10000, 'colsample_bytree':0.45, 'nthread':nthread, 'seed':seed}


    #NN params
    nb_epoch = 3
    batch_size = 128
    esr = 402

    param1 = {
        'hidden_units': (256, 256),
        'activation': (advanced_activations.PReLU(),advanced_activations.PReLU(),core.activations.sigmoid),
        'dropout': (0., 0.), 'optimizer': RMSprop(), 'nb_epoch': nb_epoch,
    }
    param2 = {
        'hidden_units': (1024, 1024),
        'activation': (advanced_activations.PReLU(),advanced_activations.PReLU(),core.activations.sigmoid),
        'dropout': (0., 0.), 'optimizer': RMSprop(), 'nb_epoch': nb_epoch,
    }
    clfs = [
        (D2, XGBClassifier(**xgb_cla)),
        (D11, XGBClassifier(**xgb_cla)),

        (D2, XGBRegressor(**xgb_reg)),
        (D11, XGBRegressor(**xgb_reg)),

        (D2, ensemble.ExtraTreesClassifier(**extra_params_kaggle_cla)),
        (D11, ensemble.ExtraTreesClassifier(**extra_params_kaggle_cla)),

        (D2, ensemble.ExtraTreesRegressor(**extra_params_kaggle_reg)),
        (D11, ensemble.ExtraTreesRegressor(**extra_params_kaggle_reg)),

    # (D1, NN(input_dim=D1[0].shape[1], output_dim=1, batch_size=batch_size, early_stopping_epoch=esr, verbose=2, loss='binary_crossentropy', class_mode='binary', **param1)),
    # (D3, NN(input_dim=D3[0].shape[1], output_dim=1, batch_size=batch_size, early_stopping_epoch=esr, verbose=2,loss='binary_crossentropy', class_mode='binary', **param1)),
    # (D5, NN(input_dim=D5[0].shape[1], output_dim=1, batch_size=batch_size, early_stopping_epoch=esr, verbose=2,loss='binary_crossentropy', class_mode='binary', **param1)),
    #
    # (D1, NN(input_dim=D1[0].shape[1], output_dim=1, batch_size=batch_size, early_stopping_epoch=esr, verbose=2,loss='binary_crossentropy', class_mode='binary', **param2)),
    # (D3, NN(input_dim=D3[0].shape[1], output_dim=1, batch_size=batch_size, early_stopping_epoch=esr, verbose=2,loss='binary_crossentropy', class_mode='binary', **param2)),
    # (D5, NN(input_dim=D5[0].shape[1], output_dim=1, batch_size=batch_size, early_stopping_epoch=esr, verbose=2,loss='binary_crossentropy', class_mode='binary', **param2))

    ]
    for clf in clfs:
        yield clf

def printResults(dic_logs):
        l_train_logloss = dic_logs['train_error']
        l_val_logloss = dic_logs['val_error']

        string = ("{0:.4f}-{1:.4f}".format(np.mean(l_train_logloss), np.mean(l_val_logloss)))
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

# General params
STORE = True
n_folds = 5
nthread = 8
seed = 123


# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D1_[LE-cat]_[NAmean].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D1 = (X, Y, X_test,test_idx, data_name)

X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D2_[LE-cat]_[NA-999].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D2 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D3_[OH300]_[NAmean].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D3 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D4_[OH300]_[NA-999].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D4 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D5_[OnlyCont]_[NAmean].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D5 = (X, Y, X_test, test_idx, data_name, col_feats)
# #
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D6_[OnlyCatLE].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D6 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D7_[OnlyCatOH].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D7 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D8_[ColsRemoved]_[Namean]_[OH].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D8 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D9_[ColsRemoved]_[NA-999]_[LE-cat].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D9 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D10_[ColsRemoved]_[NA-999]_[OH].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D10 = (X, Y, X_test, test_idx, data_name, col_feats)


X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadParseData('D11_[OH1000]_[NA-999].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D11 = (X, Y, X_test, test_idx, data_name, col_feats)

skf = cross_validation.StratifiedKFold(Y, n_folds=n_folds, shuffle=True, random_state=seed)
clfs = models()

#CV
for clf_indice, data_clf in enumerate(clfs):

    print('-' * 50)
    print("Classifier [%i]" % clf_indice)
    X = data_clf[0][0]
    Y = data_clf[0][1]
    X_test = data_clf[0][2]
    test_idx = data_clf[0][3]
    print(X.shape)

    clf = data_clf[1] ; print(clf)
    clf_name = clf.__class__.__name__
    clf_name_short = clf_name[:3]

    data_name = data_clf[0][4]
    cols_name = data_clf[0][5]

    blend_X = np.zeros((len(X), 1))
    blend_X_test = np.zeros((len(X_test), 1))
    blend_X_test_fold = np.zeros((len(X_test), len(skf)))

    dic_logs = {'name': clf_name, 'feat_importance': None,
                'blend_X': blend_X, 'blend_Y': Y, 'blend_X_test': blend_X_test, 'test_idx': test_idx,
                'params': None, 'prepro': None, 'best_epoch': [], 'best_val_metric': [],
                'train_error': [], 'val_error': []}

    filename = '{}_{}_{}f_CV{}'.format(clf_name_short, data_name,X.shape[1],n_folds)

    for fold_indice, (train_indices, val_indices) in enumerate(skf):
        print("Fold [%i]" % fold_indice)
        xtrain = X[train_indices]
        ytrain = Y[train_indices]
        xval = X[val_indices]
        yval = Y[val_indices]

        if clf_name[:3] == 'XGB':

            added_params = ["_{}".format('-'.join(list(map(lambda x: x[:3], clf.objective.split(':'))))),
                            "_d{}".format(clf.max_depth),
                            "_lr{}".format(clf.learning_rate)
                            ]

            clf.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric=xgb_accuracy,
                    early_stopping_rounds=300, verbose=True)

            dic_logs['params'] = clf.get_params()
            dic_logs['feat_importance'] = str_feat_importance(clf._Booster, cols_name)
            dic_logs['best_epoch'].append(clf.best_iteration)
            dic_logs['best_val_metric'].append(clf.best_score)

            if hasattr(clf, 'predict_proba'):

                train_pred = reshapePrediction(clf.predict_proba(xtrain, ntree_limit=clf.best_ntree_limit))
                val_pred = reshapePrediction(clf.predict_proba(xval, ntree_limit=clf.best_ntree_limit))
                test_pred = reshapePrediction(clf.predict_proba(X_test, ntree_limit=clf.best_ntree_limit))

            elif hasattr(clf, 'predict'):

                train_pred = reshapePrediction(clf.predict(xtrain, ntree_limit=clf.best_ntree_limit))
                val_pred = reshapePrediction(clf.predict(xval, ntree_limit=clf.best_ntree_limit))
                test_pred = reshapePrediction(clf.predict(X_test, ntree_limit=clf.best_ntree_limit))

        elif clf_name == 'NN':
            added_params = [""]
            logs = clf.fit(xtrain, ytrain, eval_set=(xval, yval), show_accuracy=True)

            dic_logs['params'] = clf.model.get_config()
            dic_logs['best_epoch'].append(np.argmin(logs.history['val_loss']))
            dic_logs['best_val_metric'].append(np.min(logs.history['val_loss']))

            if hasattr(clf, 'predict_proba'):

                train_pred = reshapePrediction(clf.predict_proba(xtrain))
                val_pred = reshapePrediction(clf.predict_proba(xval))
                test_pred = reshapePrediction(clf.predict_proba(X_test))

            elif hasattr(clf, 'predict'):

                train_pred = reshapePrediction(clf.predict(xtrain))
                val_pred = reshapePrediction(clf.predict(xval))
                test_pred = reshapePrediction(clf.predict(X_test))

        elif clf_name == "CalibratedClassifierCV":
            sub_clf_name = clf.base_estimator.__class__.__name__
            sub_clf = clf.base_estimator
            if sub_clf_name == "RandomForestClassifier" or sub_clf_name == "ExtraTreesClassifier" :
                dic_logs['params'] = clf.get_params()
                added_params = [
                    "_{}".format(sub_clf_name[:3]),
                    "_{}".format(clf.method[:3]),
                    "_{}".format(sub_clf.criterion[:3]),
                    "_md{}".format(sub_clf.max_depth)
                ]
                clf.fit(xtrain,ytrain)

            if hasattr(clf, 'predict_proba'):

                train_pred = reshapePrediction(clf.predict_proba(xtrain))
                val_pred = reshapePrediction(clf.predict_proba(xval))
                test_pred = reshapePrediction(clf.predict_proba(X_test))

            elif hasattr(clf, 'predict'):

                train_pred = reshapePrediction(clf.predict(xtrain))
                val_pred = reshapePrediction(clf.predict(xval))
                test_pred = reshapePrediction(clf.predict(X_test))

        elif clf_name == "RandomForestRegressor" or clf_name == "ExtraTreesRegressor":
            dic_logs['params'] = clf.get_params()
            added_params = [
                "_{}".format(clf.criterion[:3]),
                "_md{}".format(clf.max_depth)
            ]
            clf.fit(xtrain, ytrain)

            if hasattr(clf, 'predict_proba'):

                train_pred = reshapePrediction(clf.predict_proba(xtrain))
                val_pred = reshapePrediction(clf.predict_proba(xval))
                test_pred = reshapePrediction(clf.predict_proba(X_test))

            elif hasattr(clf, 'predict'):

                train_pred = reshapePrediction(clf.predict(xtrain))
                val_pred = reshapePrediction(clf.predict(xval))
                test_pred = reshapePrediction(clf.predict(X_test))

        else:
            dic_logs['params'] = clf.get_params()
            added_params = [""]
            clf.fit(xtrain, ytrain)

            if hasattr(clf, 'predict_proba'):

                train_pred = reshapePrediction(clf.predict_proba(xtrain))
                val_pred = reshapePrediction(clf.predict_proba(xval))
                test_pred = reshapePrediction(clf.predict_proba(X_test))

            elif hasattr(clf, 'predict'):

                train_pred = reshapePrediction(clf.predict(xtrain))
                val_pred = reshapePrediction(clf.predict(xval))
                test_pred = reshapePrediction(clf.predict(X_test))


        #filling blend datasets
        blend_X_test_fold[:, fold_indice] = test_pred

        #metrics
        train_error = eval_func(ytrain, train_pred)
        val_error = eval_func(yval, val_pred)

        print("train/val error: [{0:.4f}|{1:.4f}]".format(train_error, val_error))
        print(metrics.confusion_matrix(yval, val_pred.round()))

        dic_logs['blend_X'][val_indices, 0] = val_pred
        dic_logs['train_error'].append(train_error)
        dic_logs['val_error'].append(val_error)


    dic_logs['blend_X_test'][:, 0] = np.mean(blend_X_test_fold, axis = 1)


    string_result = printResults(dic_logs)
    filename += "{}_{}".format(''.join(added_params), string_result)

    print(filename)

    if STORE:
        saveDicLogs(dic_logs, SAVE_FOLDER + filename + '.p')

    #submission
    y_test_pred = dic_logs['blend_X_test'][:, 0]

    output_filename = SAVE_FOLDER + filename + '.csv'
    np.savetxt(output_filename, np.vstack((test_idx, y_test_pred)).T,
               delimiter=',', fmt='%i,%.10f', header='ID,PredictedProb', comments="")



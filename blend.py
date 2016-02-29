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
def xgb_accuracy(ypred, dtrain):
        ytrue = dtrain.get_label().astype(int)
        ypred = np.round(ypred).astype(int)
        return 'acc', -metrics.accuracy_score(ytrue, ypred)
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
            logs = self.model.fit(X, y, self.batch_size, self.nb_epoch, self.verbose, validation_data=eval_set, callbacks=[early_stopping], show_accuracy=True, class_weight=class_weight)
        else:
            logs = self.model.fit(X, y, self.batch_size, self.nb_epoch, self.verbose, show_accuracy=True, class_weight=class_weight)

        return logs

    def predict_proba(self, X):

        prediction = self.model.predict_proba(X, verbose=0)

        return prediction
def xgb_accuracy(ypred, dtrain):
        ytrue = dtrain.get_label().astype(int)
        # ypred = np.round(ypred).astype(int)
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
def NNmodels():
    """NN list of models"""
    #NN params
    nb_epoch = 100
    batch_size = 256
    esr = 40

    param1 = {
        'hidden_units': (128, 64, 32),
        'activation': (advanced_activations.PReLU(), advanced_activations.PReLU(), advanced_activations.PReLU(),
                       core.activations.sigmoid),
        'dropout': (0., 0.), 'optimizer': SGD(), 'nb_epoch': nb_epoch,
    }
    clfs = [
        [DATA, NN(input_dim=DATA[0].shape[1], output_dim=1, batch_size=batch_size, early_stopping_epoch=esr, verbose=2, loss='binary_crossentropy', class_mode='binary', **param1)],
    ]
    return clfs
def SKLEARNmodels():
    params = {'n_jobs':nthread,'random_state':seed,'class_weight':'balanced'}

    sgd = linear_model.SGDClassifier(loss='log', n_iter=100, class_weight='balanced', penalty='l2', n_jobs=nthread, random_state=seed)
    extra = ensemble.ExtraTreesClassifier(criterion='gini',max_depth=10,n_estimators=50,**params)
    extra1 = ensemble.ExtraTreesClassifier(criterion='entropy',n_estimators=1000,**params)


    xgb = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=10,nthread=nthread,
                        subsample=0.5, seed=seed)

    clfs = [
        # [DATA, sgd],
        # [DATA, rf],
        # [DATA, rf1],
        # [DATA, extra],
        # [DATA, extra1],
        [DATA, xgb],
    ]
    return clfs
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

# Params
#General params
# filename = 'pd_data_[LEcat].p'
# filename = 'pd_data_[DummyCat-thresh300]_fillNANcont_var.p'
filename = 'pd_data_[LEcat]_fillNANcont_var.p'

TODO = 'SKL'
CV = 'strat'
STORE = True
DOTEST = True
n_folds = 3
test_size = 0.10
nthread = 8
seed = 123


folder = CODE_FOLDER + 'diclogs/'
import glob
l_filenames = glob.glob1(folder, '*.p')


l_X = []
l_Y = []

n_folds = 3
n_models = 2
for filename in l_filenames:
    dic_log = pickle.load(open(folder + filename,'rb'))
    l_ypredproba = dic_log['ypredproba']
    l_X.append(l_ypredproba)



for a in zip(*l_X):
    print(len(a))







skf = cross_validation.StratifiedShuffleSplit(Y, n_iter=n_folds, test_size=test_size, random_state=seed)

for fold_indice, (tr_idx, te_idx) in enumerate(skf):


    for name in l_names:
        dic_log = pickle.load(open(folder + names,'rb'))









X, Y, X_test, test_idx, pd_data = LoadParseData('pd_data_[DummyCat-thresh300]_fillNANcont_var.p')
# X1, Y1, X_test1, test_idx1, pd_data1 = LoadParseData('pd_data_[DummyCat-thresh300]_fillNANcont_var.p')

DATA = (X, Y, X_test,test_idx)
# DATA1 = (X1, Y1)


if CV == 'strat': skf = cross_validation.StratifiedShuffleSplit(Y, n_iter=n_folds, test_size=test_size, random_state=seed)
if CV == 'random': skf = cross_validation.ShuffleSplit(len(Y), n_iter=n_folds, test_size=test_size, random_state=seed)


if TODO == 'NN': clfs = NNmodels()
if TODO == 'SKL': clfs = SKLEARNmodels()


##################################################################################
# CV
##################################################################################
for clf_indice, data_clf in enumerate(clfs):

    print('-' * 50)
    print("Classifier [%i]" % clf_indice)
    X = data_clf[0][0]
    Y = data_clf[0][1]
    X_test = data_clf[0][2]
    test_idx = data_clf[0][3]

    clf = data_clf[1] ; print(clf)
    clf_name = clf.__class__.__name__

    dic_logs = {'name':clf_name, 'fold':[],'ypredproba':[],'yval':[],
                'params':None,'prepro':None,'best_epoch':[],'best_val_metric':[],
                'train_error': [], 'val_error':[]}


    for fold_indice, (tr_idx, te_idx) in enumerate(skf):
        print("Fold [%i]" % fold_indice)
        xtrain = X[tr_idx]
        ytrain = Y[tr_idx]
        xval = X[te_idx]
        yval = Y[te_idx]

        try: #xgb
            dic_logs['params'] = clf.get_params()
            clf.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric=xgb_accuracy, early_stopping_rounds=360, verbose=True)

            dic_logs['best_epoch'].append(clf.best_iteration)
            dic_logs['best_val_metric'].append(clf.best_score)

        except:
            try: #NN
                logs = clf.fit(xtrain, ytrain, eval_set=(xval, yval), show_accuracy=True)

                dic_logs['params'] = clf.model.get_config()
                dic_logs['best_epoch'].append(np.argmin(logs.history['val_loss']))
                dic_logs['best_val_metric'].append(np.min(logs.history['val_loss']))

            except:
                try: #sklearn
                    dic_logs['params'] = clf.get_params()
                    clf.fit(xtrain, ytrain)
                except:
                    print('no fit')

        ypredproba = clf.predict_proba(xval)
        ypredproba = reshapePrediction(ypredproba)

        train_error = eval_func(ytrain, reshapePrediction(clf.predict_proba(xtrain)))
        val_error = eval_func(yval, ypredproba)

        print("train/val error: [{0:.4f}|{1:.4f}]".format(train_error, val_error))
        print(metrics.confusion_matrix(yval, ypredproba.round()))

        dic_logs['fold'].append(fold_indice)
        dic_logs['ypredproba'].append(ypredproba)
        dic_logs['yval'].append(yval)
        dic_logs['train_error'].append(train_error)
        dic_logs['val_error'].append(val_error)


    string_result = printResults(dic_logs); print(string_result)

    filename = '{}_{}feats_CV{}_{}'.format(clf_name,X.shape[1],n_folds,string_result)
    if STORE: saveDicLogs(dic_logs, filename + '.p')

    if DOTEST:
        print('Test prediction...')
        try: #xgb
            clf.n_estimators = np.mean(dic_logs['best_epoch']).astype(int)
            clf.fit(X, Y, )
        except:
            try: #NN
                logs = clf.fit(X, Y)
            except:
                try:
                    clf.fit(X, Y)
                except:
                    print('no fit')

        ypredproba = clf.predict_proba(X_test)
        ypredproba = reshapePrediction(ypredproba)

        pd_submission = pd.DataFrame({'ID':test_idx, 'PredictedProb':ypredproba})
        pd_submission.to_csv(CODE_FOLDER + 'diclogs/' + filename + '.csv', index=False)





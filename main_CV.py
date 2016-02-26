from keras.backend.tensorflow_backend import reshape

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

from keras.preprocessing import text, sequence
from keras.optimizers import *
from keras.models import Sequential
from keras.utils import np_utils

from keras.layers import core, embeddings, recurrent, advanced_activations, normalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

def reshapePrediction(ypredproba):
    result = None
    if len(ypredproba.shape) > 1:
        if ypredproba.shape[1] == 1: result = ypredproba[:, 0]
        if ypredproba.shape[1] == 2: result = ypredproba[:, 1]
    else:
        result = ypredproba.ravel()
    return result
def eval_func(ytrue, ypredproba):
    score = 0

    ypredproba = reshapePrediction(ypredproba)
    # ypred = np.round(ypredproba).astype(int)

    score = metrics.log_loss(ytrue, ypredproba)

    return score
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


    pd_data = pd.read_hdf(CODE_FOLDER + "data/" + filename)

    pd_train = pd_data[pd_data.target >= 0]
    pd_test = pd_data[pd_data.target == -1]

    Y = pd_train['target'].values
    test_idx = pd_test['ID'].values

    X = np.array(pd_train.drop(['ID', 'target'],1))
    X_test = np.array(pd_test.drop(['ID','target'], 1))

    return X, Y, X_test, test_idx
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
            logs = self.model.fit(X, y, self.batch_size, self.nb_epoch, self.verbose, validation_data=eval_set, callbacks=[early_stopping], show_accuracy=True)
        else:
            logs = self.model.fit(X, y, self.batch_size, self.nb_epoch, self.verbose, show_accuracy=True)

        return logs

    def predict_proba(self, X):

        prediction = self.model.predict_proba(X, verbose=0)

        return prediction
def xgb_accuracy(ypred, dtrain):
        ytrue = dtrain.get_label().astype(int)
        # ypred = np.round(ypred).astype(int)
        return 'logloss', metrics.log_loss(ytrue, ypred)
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=operator.itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
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
            logs = self.model.fit(X, y, self.batch_size, self.nb_epoch, self.verbose, validation_data=eval_set, callbacks=[early_stopping], show_accuracy=True)
        else:
            logs = self.model.fit(X, y, self.batch_size, self.nb_epoch, self.verbose, show_accuracy=True)

        return logs

    def predict_proba(self, X):

        prediction = self.model.predict_proba(X, verbose=0)

        return prediction
def NNmodels(X, Y):
    """NN list of models"""

    # from sklearn.preprocessing import StandardScaler
    # scl = StandardScaler()
    # X = scl.fit_transform(X.todense())
    DATA = (X, Y)

    #NN params
    nb_epoch = 5
    batch_size = 32
    esr = 40

    param1 = {
        'hidden_units': (512, 512),
        'activation': (advanced_activations.PReLU(), advanced_activations.PReLU(), core.activations.sigmoid),
        'dropout': (0., 0.), 'optimizer': RMSprop(lr=0.0005), 'nb_epoch': nb_epoch,
    }
    # param1 = {
    # 'hidden_units': (256, 256),
    # 'activation': (advanced_activations.PReLU(), advanced_activations.PReLU() , core.activations.sigmoid),
    # 'dropout': (0., 0.), 'optimizer': RMSprop(lr=0.0005), 'nb_epoch': nb_epoch,
    # }

    clfs = [
        [DATA, NN(input_dim=DATA[0].shape[1], output_dim=1, batch_size=batch_size, early_stopping_epoch=esr, verbose=2, loss='binary_crossentropy', class_mode='binary', **param1)],
    ]
    return clfs
def SKLEARNmodels():

    # sgd = linear_model.SGDClassifier(loss='modified_huber', n_iter=20, class_weight=None, penalty='l2', n_jobs=nthread, random_state=seed)
    # nb = naive_bayes.MultinomialNB(alpha=1., fit_prior=True, class_prior=None)
    # lr = linear_model.LogisticRegression(penalty="l1", dual=False, tol=1e-4, C=5., random_state=seed, max_iter=1000, n_jobs=nthread, class_weight='balanced')
    # rf = ensemble.RandomForestClassifier(criterion="gini", class_weight='balanced', n_estimators=100, max_depth=30, n_jobs=nthread, random_state=seed)
    # ada = ensemble.AdaBoostClassifier(base_estimator=rf,n_estimators=30)
    xgb = XGBClassifier(max_depth=10, learning_rate=0.01, n_estimators=10000, objective='binary:logistic', nthread=nthread,
                        subsample=.7, seed=seed)
    # xgb = XGBClassifier(max_depth=20, learning_rate=0.01, n_estimators=10000, objective='binary:logistic', nthread=nthread,
    #                     subsample=0.9, seed=seed)

    # voting = ensemble.VotingClassifier([('lr', lr), ('xgb', xgb)], voting='soft')


    clfs = [
        # [DATA, sgd],
        # [DATA, nb],
        # [DATA, rf],
        # [DATA, lr],
        # [DATA, ada],
        [DATA, xgb],
        # [DATA, voting]
    ]
    return clfs
def printResults(dic_logs, l_clf_names=[]):
    output_string=[]

    for clf_name in l_clf_names:
        l_pd_result = []
        output_string.append(clf_name)

        l_roc = []
        l_acc = []
        l_precision_0 = []
        l_precision_1 = []
        l_recall_0 = []
        l_recall_1 = []
        l_f1_0 = []
        l_f1_1 = []
        l_matconf = []



        for fold_idx in dic_logs[clf_name]['fold']:


            ypredproba = dic_logs[clf_name]['ypredproba'][fold_idx]
            ypredproba = reshapePrediction(ypredproba)

            ypred = np.round(ypredproba).astype(int)
            yval = dic_logs[clf_name]['yval'][fold_idx]


            #metrics
            roc = metrics.roc_auc_score(yval, ypredproba)
            acc = metrics.accuracy_score(yval, ypred)
            precision_0, recall_0, f1_0, _ = metrics.precision_recall_fscore_support(yval, ypred, average='binary', pos_label=0)
            precision_1, recall_1, f1_1, _ = metrics.precision_recall_fscore_support(yval, ypred, average='binary', pos_label=1)
            matconf = metrics.confusion_matrix(yval, ypred)


            l_roc.append(roc)
            l_acc.append(acc)
            l_precision_0.append(precision_0)
            l_precision_1.append(precision_1)
            l_recall_0.append(recall_0)
            l_recall_1.append(recall_1)
            l_f1_0.append(f1_0)
            l_f1_1.append(f1_1)
            l_matconf.append(matconf)

        string = ("acc|roc = %.3f|%.3f, C0: pres|rec|f1 = %.3f|%.3f|%.3f, C1: pres|rec|f1 = %.3f|%.3f|%.3f" %
              (np.mean(l_acc), np.mean(l_roc),
               np.mean(l_precision_0), np.mean(l_recall_0), np.mean(l_f1_0),
               np.mean(l_precision_1), np.mean(l_recall_1), np.mean(l_f1_1))
              )
        output_string.append(string)


# Params
#General params
n_folds = 1
test_size = 0.20
nthread = 6
seed = 123

filename = 'pd_data_[LEcat].p'
filename = 'pd_data_[DummyCat-thresh300].p'


X, Y, X_test, test_idx = LoadParseData(filename)
DATA = (X, Y)


TODO = 'SKL'
# TODO = 'NN'
STORE = False
skf = cross_validation.StratifiedShuffleSplit(Y, n_iter=n_folds, test_size=test_size, random_state=seed)


# clf = svm.LinearSVC(class_weight=None,random_state=seed)
# clf = naive_bayes.MultinomialNB(alpha=.1)
# clf = linear_model.LogisticRegression(C=5.)
# scores = cross_validation.cross_val_score(clf, X, Y, scoring='accuracy', cv=skf, n_jobs=nthread, verbose=1)
# print(scores)
# print(np.mean(scores))



# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SelectFromModel
# model = SelectFromModel(ExtraTreesClassifier(max_depth=50, n_estimators=500, verbose=1, n_jobs=nthread),
#                         prefit=False, threshold='median')
# model.fit(X, Y)
#
# X = model.transform(X)
# print(X.shape)


if TODO == 'NN':
    clfs = NNmodels(X.todense(), Y)
    print(X.shape)
if TODO == 'SKL':
    DATA = (X, Y)
    print(X.shape)
    clfs = SKLEARNmodels()

##################################################################################
# CV
##################################################################################
dic_logs = {}

for clf_indice, data_clf in enumerate(clfs):

    print('-' * 50)
    print("Classifier [%i]" % clf_indice)
    startime = time.time()
    X = data_clf[0][0]
    Y = data_clf[0][1]

    clf = data_clf[1]
    clf_name = clf.__class__.__name__
    print(clf)

    dic_logs[clf_name] = {
        'fold': [],
        'ypredproba': [],
        'yval': [],
        'best_epoch': [],
        'best_val_metric': [],
        'eval_func': [],
    }


    for fold_indice, (tr_idx, te_idx) in enumerate(skf):
        print("Fold [%i]" % fold_indice)
        xtrain = X[tr_idx]
        ytrain = Y[tr_idx]
        xval = X[te_idx]
        yval = Y[te_idx]


        try: #xgb
            clf.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric=xgb_accuracy, early_stopping_rounds=360, verbose=True)

            dic_logs[clf_name]['best_epoch'].append(clf.best_iteration)
            dic_logs[clf_name]['best_val_metric'].append(clf.best_score)

        except:
            try: #NN
                logs = clf.fit(xtrain, ytrain, eval_set=(xval, yval), show_accuracy=True)

                dic_logs[clf_name]['best_epoch'].append(np.argmin(logs.history['val_loss']))
                dic_logs[clf_name]['best_val_metric'].append(np.min(logs.history['val_loss']))

            except:
                try: #sklearn
                    clf.fit(xtrain, ytrain)
                except:
                    print('no fit')

        ypredproba = clf.predict_proba(xval)
        ypredproba = reshapePrediction(ypredproba)

        error = eval_func(yval, ypredproba)
        print(error)

        dic_logs[clf_name]['fold'].append(fold_indice)
        dic_logs[clf_name]['ypredproba'].append(ypredproba)
        dic_logs[clf_name]['yval'].append(yval)
        dic_logs[clf_name]['eval_func'].append(error)

    print(metrics.confusion_matrix(yval, ypredproba.round()))
    # print(printResults(dic_logs, l_clf_names=[clf_name]))


def saveDicLogs(dic_logs, filename):

    folder2save = CODE_FOLDER + 'data/4_diclogs/'
    if os.path.exists(folder2save + filename):
        print('file exist !')
        raise BrokenPipeError
    else:
        proof_code = open(CODE_FOLDER + 'code/main_CV.py', 'rb').read()
        open(folder2save + filename + '.py', 'wb').write(proof_code)
        pickle.dump(dic_logs, open(folder2save + filename, 'wb'))
    return
if STORE: saveDicLogs(dic_logs, filename)



ypredproba = clf.predict_proba(X_test)
ypredproba = reshapePrediction(ypredproba)

pd_prediction = pd.DataFrame({'ID':test_idx, 'PredictedProb':ypredproba})
pd_prediction.to_csv('test.csv', index=None)







##################################################################################
# debug
##################################################################################
# fold = 0
# ypredproba = dic_logs['RandomForestClassifier']['ypredproba'][fold]
# yval = dic_logs['RandomForestClassifier']['yval'][fold]
# ypred = ypredproba.argmax(1)
# print(np.bincount(ypred))
# print(metrics.confusion_matrix(yval, ypred))
# print(metrics.precision_recall_fscore_support(yval, ypred, average='binary', pos_label=0))




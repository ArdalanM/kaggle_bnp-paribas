__author__ = 'Ardalan'

CODE_FOLDER = "/home/ardalan/Documents/kaggle/bnp/"
# CODE_FOLDER = "/home/arda/Documents/kaggle/bnp/"

import os, sys, time, re, zipfile, pickle, operator
if os.getcwd() != CODE_FOLDER: os.chdir(CODE_FOLDER)
import pandas as pd
import numpy as np

from xgboost import XGBClassifier, XGBRegressor

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
def CreateDataFrameFeatureImportance(model, pd_data):
    dic_fi = model.get_fscore()
    df = pd.DataFrame(dic_fi.items(), columns=['feature', 'fscore'])
    df['col_indice'] = df['feature'].apply(lambda r: r.replace('f','')).astype(int)
    df['feat_name'] = df['col_indice'].apply(lambda r: pd_data.columns[r])
    return df.sort('fscore', ascending=False)
def LoadParseData(l_filenames, folder):

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
    # Y = np.array(l_Y).T

    return X, Y, X_test, test_idx
def xgb_accuracy(ypred, dtrain):
        ytrue = dtrain.get_label().astype(int)
        ypred = np.round(ypred).astype(int)
        return 'acc', -metrics.accuracy_score(ytrue, ypred)
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
def models():
    params = {'n_jobs':nthread,'random_state':seed,'class_weight':None}


    # sgd = linear_model.SGDClassifier(loss='log', n_iter=100, class_weight='balanced', penalty='l2', n_jobs=nthread, random_state=seed)
    # lr = linear_model.LogisticRegression(penalty='l2', tol=1e-3, C=5., **params)
    # extra = ensemble.ExtraTreesClassifier(criterion='gini',max_depth=None,n_estimators=1000,**params)
    # extra1 = ensemble.ExtraTreesClassifier(criterion='entropy',max_depth=None,n_estimators=1000,**params)
    # rf = ensemble.RandomForestClassifier(n_estimators=200, max_features=1., max_depth=10, n_jobs=nthread)
    # rf = ensemble.RandomForestClassifier(n_estimators=50, max_features=.8, max_depth=10, n_jobs=nthread)

    # rf1 = ensemble.RandomForestClassifier(criterion="entropy", n_estimators=1000,max_features='auto', max_depth=None, **params)
    #
    xgb = XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=1000, nthread=nthread, subsample=1., seed=seed)


     #NN params
    nb_epoch = 200
    batch_size = 1024
    esr = 40

    param1 = {
        'hidden_units': (1024, 1024),
        'activation': (advanced_activations.PReLU(),advanced_activations.PReLU(),core.activations.sigmoid),
        'dropout': (0., 0.), 'optimizer': RMSprop(), 'nb_epoch': nb_epoch,
    }


    clfs = [
        # [DATA, lr]
        # [DATA, NN(input_dim=DATA[0].shape[1], output_dim=1, batch_size=batch_size, early_stopping_epoch=esr, verbose=2, loss='binary_crossentropy', class_mode='binary', **param1)],
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
CV = 'strat'
STORE = False
DOTEST = False
n_folds = 2
test_size = 0.20
nthread = 12
seed = 123


folder = CODE_FOLDER + 'diclogs/'
import glob
l_filenames = glob.glob1(folder, '[E|R|X|N]*.p') ; print(l_filenames)
# l_filenames = glob.glob1(folder, 'blend*[0-9].p') ; print(l_filenames)
# l_filenames = glob.glob1(folder, '*.p')
print(len(l_filenames), l_filenames)



# l_filenames = ['XGBClassifier_131feats_CV10_logTRVAL_0.2654|0.4611.p',
#        'ExtraTreesClassifier_D2_508feats_CV10_logTRVAL_0.2862|0.4699.p',
#        'RandomForestClassifier_507feats_CV10_logTRVAL_0.1269|0.4678.p',
#        'XGBClassifier_D5_508feats_CV10_logTRVAL_0.3065|0.4592.p',
#        'ExtraTreesClassifier_507feats_CV10_logTRVAL_0.0000|0.4684.p',
#        'RandomForestClassifier_D4_187feats_CV10_logTRVAL_0.2312|0.4636.p',
#        'ExtraTreesClassifier_131feats_CV10_logTRVAL_0.0000|0.4679.p',
#        'ExtraTreesClassifier_D1_132feats_CV10_logTRVAL_0.2192|0.4615.p',
#        'ExtraTreesClassifier_D5_508feats_CV10_logTRVAL_0.2920|0.4715.p',
#        'XGBClassifier_507feats_CV10_logTRVAL_0.3807|0.4638.p']


# l_filenames = ['ExtraTreesClassifier_D5_508feats_CV10_logTRVAL_0.2990|0.4729.p',
#  'ExtraTreesClassifier_D5_508feats_CV10_logTRVAL_0.2920|0.4715.p',
#  'ExtraTreesClassifier_D3_187feats_CV10_logTRVAL_0.2431|0.4652.p',
#  'ExtraTreesClassifier_507feats_CV10_logTRVAL_0.0000|0.4684.p',
#  'XGBClassifier_D3_187feats_CV10_logTRVAL_0.2687|0.4603.p',
#  'RandomForestClassifier_D4_187feats_CV10_logTRVAL_0.2312|0.4636.p',
#  'ExtraTreesClassifier_D1_132feats_CV10_logTRVAL_0.2192|0.4615.p',
#  'XGBClassifier_D5_508feats_CV10_logTRVAL_0.3065|0.4592.p',
#  'RandomForestClassifier_507feats_CV10_logTRVAL_0.1269|0.4678.p',
#  'XGBClassifier_507feats_CV10_logTRVAL_0.3807|0.4638.p',
#  'ExtraTreesClassifier_D2_508feats_CV10_logTRVAL_0.2862|0.4699.p',
#  'XGBClassifier_131feats_CV10_logTRVAL_0.2654|0.4611.p',
#  'ExtraTreesClassifier_131feats_CV10_logTRVAL_0.0000|0.4679.p',
#  'XGBClassifier_D3_187feats_CV10_logTRVAL_0.2962|0.4601.p',
#  'NN_D3_187feats_CV10_logTRVAL_0.4715|0.4823.p']

# [(857, 'ExtraTreesClassifier_507feats_CV10_logTRVAL_0.0000|0.4684.p'), (514, 'ExtraTreesClassifier_131feats_CV10_logTRVAL_0.0000|0.4679.p'), (393, 'ExtraTreesClassifier_131feats_CV10_logTRVAL_0.0000|0.4692.p'), (378, 'NN_507feats_CV10_logTRVAL_0.4625|0.4838.p'), (377, 'XGBClassifier_507feats_CV10_logTRVAL_0.3052|0.4623.p'), (372, 'NN_507feats_CV10_logTRVAL_0.4619|0.4832.p'), (368, 'ExtraTreesClassifier_D2_508feats_CV10_logTRVAL_0.2862|0.4699.p'), (365, 'ExtraTreesClassifier_D4_187feats_CV10_logTRVAL_0.2526|0.4668.p'), (358, 'ExtraTreesClassifier_D4_187feats_CV10_logTRVAL_0.2607|0.4681.p'), (338, 'NN_D3_187feats_CV10_logTRVAL_0.4715|0.4823.p'), (326, 'XGBClassifier_507feats_CV10_logTRVAL_0.3807|0.4638.p'), (321, 'NN_D3_187feats_CV10_logTRVAL_0.4561|0.4796.p'), (321, 'ExtraTreesClassifier_D5_508feats_CV10_logTRVAL_0.2990|0.4729.p'), (310, 'ExtraTreesClassifier_D3_187feats_CV10_logTRVAL_0.2431|0.4652.p'), (307, 'RandomForestClassifier_131feats_CV10_logTRVAL_0.1269|0.4699.p'), (303, 'RandomForestClassifier_507feats_CV10_logTRVAL_0.1269|0.4678.p'), (303, 'NN_D3_187feats_CV10_logTRVAL_0.4592|0.4803.p'), (301, 'RandomForestClassifier_507feats_CV10_logTRVAL_0.1263|0.4655.p'), (282, 'NN_D3_187feats_CV10_logTRVAL_0.4645|0.4812.p'), (277, 'XGBClassifier_D5_508feats_CV10_logTRVAL_0.3065|0.4592.p'), (276, 'NN_D1_132feats_CV10_logTRVAL_0.4686|0.4826.p'), (275, 'RandomForestClassifier_D2_508feats_CV10_logTRVAL_0.2810|0.4679.p'), (271, 'ExtraTreesClassifier_D1_132feats_CV10_logTRVAL_0.2192|0.4615.p'), (270, 'XGBClassifier_507feats_CV10_logTRVAL_0.2754|0.4595.p'), (266, 'ExtraTreesClassifier_D2_508feats_CV10_logTRVAL_0.2779|0.4683.p'), (265, 'RandomForestClassifier_D1_132feats_CV10_logTRVAL_0.2481|0.4693.p'), (264, 'XGBClassifier_131feats_CV10_logTRVAL_0.3872|0.4661.p'), (263, 'ExtraTreesClassifier_D3_187feats_CV10_logTRVAL_0.2322|0.4635.p'), (263, 'ExtraTreesClassifier_D1_132feats_CV10_logTRVAL_0.2309|0.4630.p'), (248, 'ExtraTreesClassifier_D5_508feats_CV10_logTRVAL_0.2920|0.4715.p'), (222, 'RandomForestClassifier_D3_187feats_CV10_logTRVAL_0.2531|0.4675.p'), (219, 'RandomForestClassifier_D4_187feats_CV10_logTRVAL_0.2530|0.4670.p'), (209, 'RandomForestClassifier_D1_132feats_CV10_logTRVAL_0.2248|0.4653.p'), (206, 'RandomForestClassifier_D5_508feats_CV10_logTRVAL_0.2813|0.4677.p'), (196, 'RandomForestClassifier_D5_508feats_CV10_logTRVAL_0.2639|0.4651.p'), (191, 'RandomForestClassifier_D3_187feats_CV10_logTRVAL_0.2308|0.4639.p'), (182, 'XGBClassifier_D6_132feats_CV10_logTRVAL_0.2811|0.4606.p'), (180, 'RandomForestClassifier_D4_187feats_CV10_logTRVAL_0.2312|0.4636.p'), (167, 'RandomForestClassifier_D2_508feats_CV10_logTRVAL_0.2635|0.4650.p'), (165, 'XGBClassifier_D3_187feats_CV10_logTRVAL_0.2687|0.4603.p'), (156, 'XGBClassifier_131feats_CV10_logTRVAL_0.2654|0.4611.p'), (144, 'XGBClassifier_D1_132feats_CV10_logTRVAL_0.2655|0.4610.p'), (117, 'XGBClassifier_D4_187feats_CV10_logTRVAL_0.2832|0.4599.p'), (107, 'XGBClassifier_D2_508feats_CV10_logTRVAL_0.3073|0.4595.p'), (106, 'XGBClassifier_D1_132feats_CV10_logTRVAL_0.2931|0.4607.p'), (103, 'XGBClassifier_D2_508feats_CV10_logTRVAL_0.3115|0.4596.p'), (101, 'XGBClassifier_D7_508feats_CV10_logTRVAL_0.3058|0.4595.p'), (94, 'XGBClassifier_D3_187feats_CV10_logTRVAL_0.2962|0.4601.p'), (89, 'XGBClassifier_D4_187feats_CV10_logTRVAL_0.2925|0.4599.p')]



X, Y, X_test, test_idx = LoadParseData(l_filenames, folder)


# selectionTOP = 10
# pd_corr = pd.DataFrame(X).corr()
# colsSelected = pd_corr.mean().values.argsort()[:selectionTOP]
# X = X[:, colsSelected] ; print(X.shape)
DATA = (X, Y, X_test,test_idx)


# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# # Sequential Floating Forward Selection
# # clf = linear_model.LogisticRegression(penalty='l2', tol=1e-3, C=5., random_state=seed)
# clf = ensemble.RandomForestClassifier(n_estimators=200, max_features=1., max_depth=5, n_jobs=nthread)
# clf = XGBClassifier()
# sffs = SFS(clf,
#            k_features=10,
#            forward=True,
#            floating=True,
#            scoring='log_loss',
#            print_progress=True,
#            cv=2,
#            n_jobs=1)
# sffs = sffs.fit(X, Y)
#
# print('\nSequential Floating Forward Selection (k=3):')
# print(sffs.k_feature_idx_)
# print('CV Score:')
# print(sffs.k_score_)


# X = sffs.transform(X)




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


        if clf_name == 'XGBClassifier':
            dic_logs['params'] = clf.get_params()
            clf.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric=xgb_accuracy, early_stopping_rounds=10, verbose=1)

            dic_logs['best_epoch'].append(clf.best_iteration)
            dic_logs['best_val_metric'].append(clf.best_score)

        elif clf_name == 'NN':
            logs = clf.fit(xtrain, ytrain, eval_set=(xval, yval), show_accuracy=True)

            dic_logs['params'] = clf.model.get_config()
            dic_logs['best_epoch'].append(np.argmin(logs.history['val_loss']))
            dic_logs['best_val_metric'].append(np.min(logs.history['val_loss']))

        else:
            dic_logs['params'] = clf.get_params()
            clf.fit(xtrain, ytrain)

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


    string_result = printResults(dic_logs)
    filename = 'blend_{}_{}feats_CV{}_{}'.format(clf_name,X.shape[1],n_folds,string_result)
    print(filename)
    if STORE: saveDicLogs(dic_logs, filename + '.p')

    if DOTEST:
        print('Test prediction...')
        if clf_name == '':
            clf.n_estimators = np.mean(dic_logs['best_epoch']).astype(int)
            clf.fit(X, Y)
        else:
            clf.fit(X, Y)

        ypredproba = clf.predict_proba(X_test)
        ypredproba = reshapePrediction(ypredproba)

        pd_submission = pd.DataFrame({'ID':test_idx, 'PredictedProb':ypredproba})
        pd_submission.to_csv(CODE_FOLDER + 'diclogs/' + filename + '.csv', index=False)

#NN
# blendblend_RandomForestClassifier_13feats_CV10_logTRVAL_0.4474|0.4495
#
# blendblend_RandomForestClassifier_11feats_CV10_logTRVAL_0.4476|0.4496


def CreateDataFrameFeatureImportance(model, pd_data):
    dic_fi = model.get_fscore()
    df = pd.DataFrame(dic_fi.items(), columns=['feature', 'fscore'])
    df['col_indice'] = df['feature'].apply(lambda r: r.replace('f','')).astype(int)
    df['feat_name'] = df['col_indice'].apply(lambda r: pd_data.columns[r])
    return df.sort('fscore', ascending=False)


dic_fi = clf._Booster.get_fscore()


FI = [ ( dic_fi[k], l_filenames[int(k[1:])] )  for k in dic_fi]
FI = sorted(FI, reverse=True)

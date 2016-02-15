__author__ = 'Ardalan'

CODE_FOLDER = "/home/arda/Documents/kaggle/BNP/"
# CODE_FOLDER = "/home/arda/Documents/kaggle/BNP/"

import os, sys, time, re
if os.getcwd() != CODE_FOLDER: os.chdir(CODE_FOLDER)
import re, collections, operator
import pandas as pd
import numpy as np
import zipfile
import enchant

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes

from keras.preprocessing import text, sequence
from keras.optimizers import *
from keras.models import Sequential
from keras.utils import np_utils

from keras.layers import core, embeddings, recurrent, advanced_activations, normalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


def eval_func(ytrue, ypredproba):
    score = 0
    if len(ypredproba.shape) > 1:
        if ypredproba.shape[1] == 1:
            ypred = np.round(ypredproba.ravel()).astype(int)
            score =  metrics.accuracy_score(ytrue, ypred)
        else:
            ypred = np.round(ypredproba[:,1]).astype(int)
            score =  metrics.accuracy_score(ytrue, ypred)
    else:
        ypred = np.round(ypredproba.ravel()).astype(int)
        score = metrics.accuracy_score(ytrue, ypred)
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


def vectorize_DATA(maxFeats=10000):
    dico_pattern={'match_lowercase_only':'\\b[a-z]+\\b',
                  'match_word':'\\w{1,}',
                  'match_word1': '(?u)\\b\\w+\\b',
                  'match_word_punct': '\w+|[,.?!;]',
                  'match_NNP': '\\b[A-Z][a-z]+\\b|\\b[A-Z]+\\b',
                  'match_punct': "[,.?!;'-]"
                 }

    vocab = "abcdefghijklmnopqrstuvwxyz?!-:)("
    preprocessing = TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
                          lowercase=False, preprocessor=None, tokenizer=None, analyzer='word',
                          stop_words=None, ngram_range=(1, 1), token_pattern=dico_pattern['match_word1'],
                          max_df=1.0, min_df=1, max_features=maxFeats, vocabulary=None, binary=False,
                          norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=False)

    X = preprocessing.fit_transform(pd_data['Sentence'])
    Y = pd_data['Category'].values

    return X, Y

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
        model.add(core.Activation(activation[0]))
        model.add(normalization.BatchNormalization())
        # model.add(core.Dropout(dropout[0]))

        for i in range(1, len(activation) - 1):
            model.add(core.Dense(hidden_units[i], input_dim=hidden_units[i-1], W_regularizer=None))
            model.add(core.Activation(activation[i]))
            model.add(normalization.BatchNormalization())
            # model.add(core.Dropout(dropout[i]))

        model.add(core.Dense(output_dim, input_dim=hidden_units[-1], W_regularizer=None))
        model.add(core.Activation(activation[1]))
        model.compile(loss=loss, optimizer=optimizer, class_mode=class_mode)

        return model


    def fit(self, X, y, eval_set=None, class_weight=None):


        if self.loss == 'categorical_crossentropy':
            y = to_categorical(y)

        if eval_set != None and self.loss == 'categorical_crossentropy':
            eval_set = (eval_set[0], to_categorical(eval_set[1]))

        self.model = self._build_model(self.input_dim,self.output_dim,self.hidden_units,self.activation,
                                       self.dropout, self.loss, self.optimizer, self.class_mode)

        if eval_set !=None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=self.esr, verbose=1, mode='min')
            logs = self.model.fit(X, y, self.batch_size, self.nb_epoch, self.verbose, validation_data=eval_set, callbacks=[early_stopping])
        else:
            logs = self.model.fit(X, y, self.batch_size, self.nb_epoch, self.verbose)

        return logs

    def predict_proba(self, X):

        prediction = self.model.predict_proba(X, verbose=0)

        return prediction


# def LoadParseData(name, rl, preproc):
data_folder = "data/"
pdtrain = loadFileinZipFile(data_folder + "train.csv.zip", "train.csv" )
pdtest = loadFileinZipFile(data_folder + "test.csv.zip", "test.csv" )


print("class count:")
print(pd_data['Category'].value_counts())

#binarise label columns
from sklearn.preprocessing import LabelBinarizer
le = LabelBinarizer(neg_label=0, pos_label=1)
pd_data['Category'] = le.fit_transform(pd_data['Category'])

#replacing nan => ''
pd_data['Header'].fillna('', inplace=True)
pd_data['Body'].fillna('', inplace=True)
pd_data['Sentence'] = pd_data['Header'] + " " + pd_data['Body']

    # return pd_data, le















##################################################################################
# Params
##################################################################################
name = "meetic"
rl = "with"
preproc = "with"

pd_data, le = LoadParseData(name, rl, preproc)

X, Y = vectorize_DATA(maxFeats=None)

DATA = (X, Y)

#General params
n_folds = 5
test_size = 0.20
nthread = 12
seed = 456



def NNmodels():
    """NN list of models"""


    #NN params
    nb_epoch = 2
    batch_size = 2048
    esr = 5

    # param1 = {
    #     'hidden_units': (256, 256),
    #     'activation': (core.activations.prelu, core.activations.prelu, core.activations.sigmoid),
    #     'dropout': (0.01, 0.01), 'optimizer': adam(lr=0.01), 'nb_epoch': nb_epoch,
    # }
    param1 = {
        'hidden_units': (124, 124),
        'activation': (core.activations.relu, core.activations.relu, core.activations.sigmoid),
        'dropout': (0, 0), 'optimizer': Adam(), 'nb_epoch': nb_epoch,
    }
    clfs = [
        [DATA, NN(input_dim=DATA[0].shape[1], output_dim=1, batch_size=batch_size,
                  early_stopping_epoch=esr, verbose=1, loss='binary_crossentropy',
                  class_mode='binary', **param1)],
    ]
    return clfs

def SKLEARNmodels():

    nb = naive_bayes.MultinomialNB(alpha=.1, fit_prior=False, class_prior=None)
    rf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=100, n_jobs=12)
    lr = linear_model.LogisticRegression(penalty="l2", dual=False, tol=1e-4, C=5., random_state=123, max_iter=100, n_jobs=-1)

    clfs = [
        [DATA, nb],
        [DATA, lr],
        # [DATA, nb3],
        [DATA, ensemble.VotingClassifier([('nb', nb), ('rf', lr)], voting='soft')]

        # [DATA, ensemble.RandomForestClassifier(class_weight='balanced',n_estimators=100, criterion="gini", max_depth=100, n_jobs=12, verbose=0, random_state=seed)],
        # [DATA, ensemble.ExtraTreesClassifier(n_estimators=100, criterion="gini", max_depth=100, n_jobs=12, verbose=0, random_state=seed)],
        # [DATA, linear_model.LogisticRegression(n_jobs=-1, random_state=123)]
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

            ypredproba = dic_logs[clf_name]['ypredproba'][fold_idx][:,1]
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




    return '\n'.join(output_string)

# clfs = XGBmodels()
# clfs = NNmodels()
clfs = SKLEARNmodels()


##################################################################################
# CV
##################################################################################
skf = cross_validation.StratifiedShuffleSplit(Y, n_iter=n_folds, test_size=test_size, random_state=seed)
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
            clf.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric=xgb_accuracy, early_stopping_rounds=34, verbose=True)

            dic_logs[clf_name]['best_epoch'].append(clf.model.best_iteration)
            dic_logs[clf_name]['best_val_metric'].append(clf.model.best_score)

        except:
            try: #NN
                logs = clf.fit(xtrain, ytrain, eval_set=(xval, yval))

                dic_logs[clf_name]['best_epoch'].append(np.argmin(logs.history['val_loss']))
                dic_logs[clf_name]['best_val_metric'].append(np.min(logs.history['val_loss']))

            except:
                try: #sklearn
                    clf.fit(xtrain, ytrain)
                except:
                    print('no fit')

        ypredproba = clf.predict_proba(xval)
        error = eval_func(yval, ypredproba)
        print(error)
        print(metrics.confusion_matrix(yval, ypredproba.argmax(1)))

        dic_logs[clf_name]['fold'].append(fold_indice)
        dic_logs[clf_name]['ypredproba'].append(ypredproba)
        dic_logs[clf_name]['yval'].append(yval)
        dic_logs[clf_name]['eval_func'].append(error)

    print(printResults(dic_logs, l_clf_names=[clf_name]))




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


##################################################################################
# Plot
##################################################################################
def confidence_vs_automation(ypredproba=None, yval=None):
    """
    ypredproba: classifier probabilities
    yval: ground truth
    return: pandas DataFrame containing:
            accuracy, precision and recall for each automation %
    """
    pd_conf = pd.DataFrame({'ypredproba': ypredproba, 'yval': yval})
    pd_conf['confidence'] =  100*np.abs(2*pd_conf['ypredproba'] - 1)
    pd_conf = pd_conf.sort_values(by='confidence', ascending=False).reset_index(drop=True)


    l_acc = []
    l_precision_0 = []
    l_precision_1 = []
    l_recall_0 = []
    l_recall_1 = []

    l_automation_threshold = []

    for automation_threshold in range(100, 10, -1):
        l_automation_threshold.append(automation_threshold)

        #selecting samples that are above the threshold
        n_sample = int(np.round( 1e-2*automation_threshold * len(pd_conf) ))
        pd_sample = pd_conf.head(n_sample)


        ypred = pd_sample['ypredproba'].round()
        yval = pd_sample['yval']

        #get metrics on sampled data
        l_acc.append(metrics.accuracy_score(ypred, yval))
        precision_0, recall_0, f1_0, _ = metrics.precision_recall_fscore_support(yval, ypred, average='binary', pos_label=0)
        precision_1, recall_1, f1_1, _ = metrics.precision_recall_fscore_support(yval, ypred, average='binary', pos_label=1)

        l_precision_0.append(precision_0)
        l_precision_1.append(precision_1)
        l_recall_0.append(recall_0)
        l_recall_1.append(recall_1)

    pd_result = pd.DataFrame(
        {
            'acc': l_acc,
            'precision_0': l_precision_0,
            'precision_1': l_precision_1,
            'recall_0': l_recall_0,
            'recall_1': l_recall_1,
            'ratio_used': l_automation_threshold
        })
    return pd_result

def avgeragingListOfDataFrame(l_df):
    avgDataFrame = l_df[0]

    for i in range(1, len(l_df)):
        avgDataFrame = avgDataFrame.add(l_df[i])

    avgDataFrame = avgDataFrame / len(l_df)

    return avgDataFrame

def plot_report(f, axs, pd_result, clf_name=None, titles=[], ycols=[]):

    for i, (ax, ycol, title) in enumerate(zip(axs, ycols, titles)):

        xvalues = pd_result.index.values
        yvalues = pd_result[ycol]
        ax.set_title(title)

        ax.plot(xvalues, yvalues, label=clf_name)
        ax.legend(loc=2)

    # Fine-tune sfigure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    # f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    # f.axes[0].set_xticks(range(100, 0, -10))
    f.axes[0].set_xticklabels(range(100, 0, -5))

    return f

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.close('all')

ycols = ['acc', 'precision_0', 'precision_1']
titles = ycols

f, axs = plt.subplots(nrows=len(ycols), ncols=1, sharex=True, sharey=False, figsize= (10,10))


for clf_name in dic_logs:
    l_pd_result = []
    print(clf_name)

    for fold_idx in dic_logs[clf_name]['fold']:
        ypredproba = dic_logs[clf_name]['ypredproba'][fold_idx]
        yval = dic_logs[clf_name]['yval'][fold_idx]
        pd_result = confidence_vs_automation(ypredproba[:,1], yval)

        l_pd_result.append(pd_result)

    pd_result_averaged = avgeragingListOfDataFrame(l_pd_result)
    f = plot_report(f, axs, pd_result_averaged, clf_name[:4], titles, ycols)


f.savefig('sdfsdf.png')



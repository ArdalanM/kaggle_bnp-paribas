from Cython.Compiler.Errors import context

__author__ = 'Ardalan'

# CODE_FOLDER = "/home/arda/Documents/kaggle/bnp/"
CODE_FOLDER = "/home/ardalan/Documents/kaggle/bnp/"

import os, sys, time, re
if os.getcwd() != CODE_FOLDER: os.chdir(CODE_FOLDER)
import re, collections, operator
import pandas as pd
import numpy as np
import zipfile

class DummycolumnsBins():

    def __init__(self, cols=None, prefix='LOL_', nb_bins=10):

        self.prefix=prefix
        self.nb_bins = nb_bins
        self.cols = cols
        self.bins = None

    def fit(self, data):

        self.bins = np.linspace(data[self.cols].min(), data[self.cols].max(), self.nb_bins)

        return self

    def transform(self, data):

        pd_dummy = pd.get_dummies(np.digitize(data[self.cols], self.bins), prefix=self.prefix)
        # pd_dummy.index = data[self.cols].index
        # pd_dummy = pd_dummy.groupby(pd_dummy.index).sum()

        return pd_dummy

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

class Dummycolumns():


    def __init__(self, cols=None, prefix='LOL_', nb_features=10):

        self.selected_features = None
        self.rejected_features = None
        self.prefix=prefix
        self.nb_features = nb_features
        self.cols = cols

    def fit(self, pd_train, pd_test):

        #Frequent item ==> Dummify
        selected_features_train = pd_train[self.cols].value_counts().index[:self.nb_features]
        selected_features_test = pd_test[self.cols].value_counts().index[:self.nb_features]

        self.selected_features = list(set(selected_features_train).intersection(set(selected_features_test)))

        #Rare items ==> gather all into a "garbage" column
        rejected_features_train = pd_train[self.cols].value_counts().index[self.nb_features:]
        rejected_features_test = pd_test[self.cols].value_counts().index[self.nb_features:]

        self.rejected_features = list(set(rejected_features_train).intersection(set(rejected_features_test)))

    def transform(self, data):

        df_dummy = data[self.cols].apply(lambda r: r if r in self.selected_features else 'LowFreqFeat')

        #Dummy all items
        df_dummy = pd.get_dummies(df_dummy).groupby(df_dummy.index).sum()


        df_dummy = df_dummy.rename(columns=lambda x: self.prefix + str(x))

        return df_dummy


pdtrain = loadFileinZipFile(CODE_FOLDER + "data/train.csv.zip", "train.csv")
pdtest = loadFileinZipFile(CODE_FOLDER + "data/test.csv.zip", "test.csv")
pdtest['target'] = -1
pd_data = pdtrain.append(pdtest).reset_index(drop=True)

#D1_NN_LE-cat_NAmean
#D2_LE-cat_NA-999
#D3_NN_OH300_NAmean
#D3_OH300_NA-999
#D4_Only-cont_NA-999
#D5_NN_Only-cont_NA-mean
#D5_Only-Cat-LE
#D6_Only-Cat_OH


def model_data(data, LECAT=False, NAMEAN=False, NA999=False, OH=False, ONLYCONT=False, ONLYCAT=False, ONLYCATOH=False,
               COLSREMOVAL=False, cols=[], maxCategories=30):

    data = data.drop(['v107'], 1)
    data['nb_nan'] = data.isnull().sum(1)


    cat_var = list(data.select_dtypes(["object"]).columns)
    cont_var = list(data.select_dtypes(["float", "int"]).columns)

    if COLSREMOVAL:
        data.drop(cols, 1, inplace=True)
        cat_var = list(data.select_dtypes(["object"]).columns)
        cont_var = list(data.select_dtypes(["float", "int"]).columns)


    if NAMEAN:
        for col in cont_var:
            data.loc[data[col].isnull(), col] = data[col].mean()

    if NA999:
        for col in cont_var:
            data.loc[data[col].isnull(), col] = -999

    if LECAT:
        for col in data[cat_var]: data[col] = pd.factorize(data[col])[0]

    if OH:
        maxCategories = 30
        cols2dummy = [col for col in data[cat_var] if len(data[col].unique()) <= maxCategories]
        colsNot2dummy = [col for col in data[cat_var] if len(data[col].unique()) > maxCategories]
        data = pd.get_dummies(data, dummy_na=True, columns=cols2dummy)

        #binning
        for col in colsNot2dummy:
            data[col] = pd.factorize(data[col])[0]
            dcb = DummycolumnsBins(cols=col, prefix=col, nb_bins=20)
            dcb.fit(data)
            pd_binned = dcb.transform(data)
            data = pd.concat([data,pd_binned],1)
    if ONLYCONT:
        data = data[cont_var]

    if ONLYCAT:
        test_idx = data['ID']
        Y = data['target']
        data = data[cat_var]
        data['ID'] = test_idx
        data['target'] = Y

    if ONLYCATOH:
        test_idx = data['ID']
        Y = data['target']
        cols = list(set(data.columns).difference(set(cont_var))) ; print(cols)
        data = data[cols]
        data['ID'] = test_idx
        data['target'] = Y


    return data


D1 = model_data(data=pd_data, LECAT=True, NAMEAN=True)
D1.to_hdf(CODE_FOLDER + 'data/D1_[LE-cat]_[NAmean].p', 'wb')

D2 = model_data(data=pd_data, LECAT=True, NA999=True)
D2.to_hdf(CODE_FOLDER + 'data/D2_[LE-cat]_[NA-999].p', 'wb')

D3 = model_data(data=pd_data, OH=True, NAMEAN=True)
D3.to_hdf(CODE_FOLDER + 'data/D3_[OH30]_[NAmean].p', 'wb')


D4 = model_data(data=pd_data, OH=True, NA999=True)
D4.to_hdf(CODE_FOLDER + 'data/D4_[OH30]_[NA-999].p', 'wb')

D5 = model_data(data=pd_data, ONLYCONT=True, NAMEAN=True)
D5.to_hdf(CODE_FOLDER + 'data/D5_[OnlyCont]_[NAmean].p', 'wb')


D6 = model_data(data=pd_data, ONLYCAT=True, LECAT=True)
D6.to_hdf(CODE_FOLDER + 'data/D6_[OnlyCatLE].p', 'wb')

D7 = model_data(data=pd_data, ONLYCATOH=True, OH=True)
D7.to_hdf(CODE_FOLDER + 'data/D7_[OnlyCatOH].p', 'wb')


#This is a NN dataset
cols2remove = ["v22",'v8','v23','v25','v31','v36','v37','v46',
               'v51','v53','v54','v63','v73','v75','v79','v81','v82',
               'v89','v92','v95','v105','v108','v109','v110',
               'v116','v117','v118','v119','v123','v124','v128']
D8 = model_data(data=pd_data, COLSREMOVAL=True,  cols=cols2remove, NAMEAN=True, OH=True)
D8.to_hdf(CODE_FOLDER + 'data/D8_[ColsRemoved]_[Namean]_[OH].p', 'wb')


cols2remove = ['v8','v23','v25','v31','v36','v37','v46',
               'v51','v53','v54','v63','v73','v75','v79','v81','v82',
               'v89','v92','v95','v105','v108','v109','v110',
               'v116','v117','v118','v119','v123','v124','v128']
D9 = model_data(data=pd_data, COLSREMOVAL=True,  cols=cols2remove, NA999=True, LECAT=True)
D9.to_hdf(CODE_FOLDER + 'data/D9_[ColsRemoved]_[NA-999]_[LE-cat].p', 'wb')


cols2remove = ['v8','v23','v25','v31','v36','v37','v46',
               'v51','v53','v54','v63','v73','v75','v79','v81','v82',
               'v89','v92','v95','v105','v108','v109','v110',
               'v116','v117','v118','v119','v123','v124','v128']
D10 = model_data(data=pd_data, COLSREMOVAL=True,  cols=cols2remove, NA999=True, OH=True)
D10.to_hdf(CODE_FOLDER + 'data/D10_[ColsRemoved]_[NA-999]_[OH].p', 'wb')




























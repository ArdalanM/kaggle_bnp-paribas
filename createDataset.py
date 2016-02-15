__author__ = 'Ardalan'

CODE_FOLDER = "/home/arda/Documents/kaggle/BNP/"
# CODE_FOLDER = "/home/arda/Documents/kaggle/BNP/"

import os, sys, time, re
if os.getcwd() != CODE_FOLDER: os.chdir(CODE_FOLDER)
import re, collections, operator
import pandas as pd
import numpy as np
import zipfile


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
        pd_dummy.index = data[self.cols].index
        pd_dummy = pd_dummy.groupby(pd_dummy.index).sum()

        return pd_dummy

def LabelEncode(pd_train, pd_test):
    le = LabelEncoder()
    for col in pd_train.select_dtypes(['object',]):
        print('le: %s' % col)
        vec = pd_train[col].append(pd_test[col]).values
        y = le.fit_transform(vec)
        pd_train[col] = y[:len(pd_train)]
        pd_test[col] = y[len(pd_train):]

        return pd_train, pd_test

def FillStrategy(pd_train, pd_test):
    for col in set(pd_train).intersection(set(pd_test)):

        nb_null_values = pd_train[col].isnull().sum()
        l_unique = len(pd_train[col].unique())
        if nb_null_values > 0:
            if col in cat_var and replace_na_cat:
                train_most_freq_val = pd_train[col].value_counts().index[0]
                test_most_freq_val = pd_test[col].value_counts().index[0]
                print('%s col (categorial) replace with: %s, %s' % (col, train_most_freq_val, test_most_freq_val))

                pd_train[col].fillna(train_most_freq_val,inplace=True)
                pd_test[col].fillna(test_most_freq_val,inplace=True)

            if col in cont_var and mean_na_cont: #replace with mean
                print('%s col (continous) meaned' % col)
                pd_train[col].fillna(pd_train[col].mean(),inplace=True)
                pd_test[col].fillna(pd_test[col].mean(),inplace=True)

            if col in disc_var and median_na_disc:#replace NAs with median value
                print('%s col (discrete) medianed' % col)
                pd_train[col].fillna(pd_train[col].median(),inplace=True)
                pd_test[col].fillna(pd_test[col].median(),inplace=True)
    return pd_train, pd_test

class MultiColumnLabelEncoder:
    def __init__(self, columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

pdtrain = loadFileinZipFile("data/train.csv.zip", "train.csv" )
pdtest = loadFileinZipFile("data/test.csv.zip", "test.csv" )

cat_var = pdtrain.select_dtypes(["object"]).columns
cont_var = pdtrain.select_dtypes(["float", "int"]).columns


print("Just Filling...")
pdtrain = loadFileinZipFile("data/train.csv.zip", "train.csv" )
pdtest = loadFileinZipFile("data/test.csv.zip", "test.csv" )
# pdtest['target'] = -1
pddata = pd.concat([pdtrain, pdtest])

replace_na_cat = False
mean_na_cont = False
median_na_disc = False
pdtrain , pdtest = FillStrategy(pdtrain, pdtest)
params = '_[Cat%i]_[Cont%i]_[Disc%i]' % (replace_na_cat, mean_na_cont, median_na_disc)
pdtrain.to_hdf('data/train' + params, 'wb')
pdtest.to_hdf('data/test' + params, 'wb')


print("LabelEncode Categorical and fill other variable")
pdtrain, pdtest = load_datasets(script_folder)
replace_na_cat = False
mean_na_cont = True
median_na_disc = True
pd_train , pd_test = FillStrategy(pd_train, pd_test)

le = LabelEncoder() #LabelEncoding Categorical var
for col in pd_train[cat_var]:
    vec = pd_train[col].append(pd_test[col]).values
    y = le.fit_transform(vec)
    pd_train[col] = y[:len(pd_train)]
    pd_test[col] = y[len(pd_train):]

params = '_[Cat%i]_[Cont%i]_[Disc%i]_LE' % (replace_na_cat, mean_na_cont, median_na_disc)
pd_train.to_hdf(data_folder + 'train' + params, 'wb')
pd_test.to_hdf(data_folder + 'test' + params, 'wb')



print('Dataset with fixed amount of features for OneHot')
pd_train, pd_test = load_datasets(script_folder)
replace_na_cat = False
mean_na_cont = True
median_na_disc = True
nb_feats = 100

params = '_[Cat%i]_[Cont%i]_[Disc%i]_[nbFeats%i]' % (replace_na_cat, mean_na_cont, median_na_disc, nb_feats)

pd_train , pd_test = FillStrategy(pd_train, pd_test)
pd_train_final = pd_train[cont_var+disc_var+dummy_var]
pd_test_final = pd_test[cont_var+disc_var+dummy_var]

#Dummy columns
for col in pd_train[cat_var]:
    print(col)
    dcf = Dummycolumns(cols=col, prefix=col+'_', nb_features=nb_feats)
    dcf.fit(pd_train, pd_test)
    pd_train_final = pd.merge(pd_train_final, dcf.transform(pd_train), left_index=True, right_index=True)
    pd_test_final = pd.merge(pd_test_final, dcf.transform(pd_test), left_index=True, right_index=True)

#Harmonizing columns between train and test dataset
cols2keep = list(set(pd_test_final).intersection(set(pd_train_final)))
pd_train_final = pd_train_final[cols2keep]
pd_train_final['Response'] = pd_train['Response']
pd_test_final = pd_test_final[cols2keep]

#save to disk
pd_train_final.to_hdf(data_folder + 'train' + params, 'wb')
pd_test_final.to_hdf(data_folder + 'test' + params, 'wb')


























__author__ = 'Ardalan'

# CODE_FOLDER = "/home/arda/Documents/kaggle/bnp/"
CODE_FOLDER = "/home/ardalan/Documents/kaggle/bnp/"

import os, sys, time, re
if os.getcwd() != CODE_FOLDER: os.chdir(CODE_FOLDER)
import re, collections, operator
import pandas as pd
import numpy as np2
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

pdtrain = loadFileinZipFile(CODE_FOLDER + "data/train.csv.zip", "train.csv")
pdtest = loadFileinZipFile(CODE_FOLDER + "data/test.csv.zip", "test.csv")
pdtest['target'] = -1
pd_data = pdtrain.append(pdtest)
pd_data['nb_nan'] = pd_data.isnull().sum(1)


cat_var = pdtrain.select_dtypes(["object"]).columns
cont_var = pdtrain.select_dtypes(["float", "int"]).columns


#v91 and v107 are duplicates



















print('LE cat imputing mean')
pd_data = pdtrain.append(pdtest)
name = 'D1'
pd_data['nb_nan'] = pd_data.isnull().sum(1)
for col in pd_data[cat_var]:
    print(col)
    pd_data[col] = pd.factorize(pd_data[col])[0]
for col in pd_data[cont_var]:
    pd_data.loc[pd_data[col].isnull(), col] = pd_data[col].mean()

filename = CODE_FOLDER + 'data/{}_[LEcat]_[NA-contvar-mean].p'.format(name)
print(filename)
pd_data.to_hdf(filename, 'wb')







print('OH300 imputing mean')
name = 'D2'
maxCategories = 300
pd_data = pdtrain.append(pdtest)
pd_data['nb_nan'] = pd_data.isnull().sum(1)

for col in pd_data[cont_var]:
    pd_data.loc[pd_data[col].isnull(), col] = pd_data[col].mean()

cols2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) <= maxCategories]
colsNot2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) > maxCategories]

for col in pd_data[colsNot2dummy]:
    pd_data[col] = pd.factorize(pd_data[col])[0]

pd_data = pd.get_dummies(pd_data, dummy_na=True, columns=cols2dummy)
filename = CODE_FOLDER + 'data/{}_[OH-thresh{}]_[NA-contvar-mean].p'.format(name, maxCategories)
pd_data.to_hdf(filename, 'wb')
print(filename)



print('OH10 imputing mean')
# OH cat and LE rest fillNAs
name = 'D3'
maxCategories = 10
pd_data = pdtrain.append(pdtest)
pd_data['nb_nan'] = pd_data.isnull().sum(1)

for col in pd_data[cont_var]:
    pd_data.loc[pd_data[col].isnull(), col] = pd_data[col].mean()

cols2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) <= maxCategories]
colsNot2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) > maxCategories]

for col in pd_data[colsNot2dummy]:
    pd_data[col] = pd.factorize(pd_data[col])[0]

pd_data = pd.get_dummies(pd_data, dummy_na=True, columns=cols2dummy)
filename = CODE_FOLDER + 'data/{}_[OH-thresh{}]_[NA-contvar-mean].p'.format(name, maxCategories)
pd_data.to_hdf(filename, 'wb')
print(filename)


print('OH10 imputing -999')
name = 'D4'
maxCategories = 10
pd_data = pdtrain.append(pdtest)
pd_data['nb_nan'] = pd_data.isnull().sum(1)

for col in pd_data[cont_var]:
    pd_data.loc[pd_data[col].isnull(), col] = -999

cols2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) <= maxCategories]
colsNot2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) > maxCategories]

for col in pd_data[colsNot2dummy]:
    pd_data[col] = pd.factorize(pd_data[col])[0]

pd_data = pd.get_dummies(pd_data, dummy_na=True, columns=cols2dummy)
filename = CODE_FOLDER + 'data/{}_[OH-thresh{}]_[NA-contvar-999].p'.format(name, maxCategories)
pd_data.to_hdf(filename, 'wb')
print(filename)


print('OH300 imputing -999')
name = 'D5'
maxCategories = 300
pd_data = pdtrain.append(pdtest)
pd_data['nb_nan'] = pd_data.isnull().sum(1)

for col in pd_data[cont_var]:
    pd_data.loc[pd_data[col].isnull(), col] = -999

cols2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) <= maxCategories]
colsNot2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) > maxCategories]

for col in pd_data[colsNot2dummy]:
    pd_data[col] = pd.factorize(pd_data[col])[0]

pd_data = pd.get_dummies(pd_data, dummy_na=True, columns=cols2dummy)
filename = CODE_FOLDER + 'data/{}_[OH-thresh{}]_[NA-contvar-999].p'.format(name, maxCategories)
pd_data.to_hdf(filename, 'wb')
print(filename)


print('LE cat nothing else')
pd_data = pdtrain.append(pdtest)
pd_data['nb_nan'] = pd_data.isnull().sum(1)
name = 'D6'

for col in pd_data[cat_var]:
    pd_data[col] = pd.factorize(pd_data[col])[0]

filename = CODE_FOLDER + 'data/{}_[LEcat].p'.format(name)
print(filename)
pd_data.to_hdf(filename, 'wb')



print('OH cat nothing else')
name = 'D7'
maxCategories = 300
pd_data = pdtrain.append(pdtest)
pd_data['nb_nan'] = pd_data.isnull().sum(1)

cols2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) <= maxCategories]
colsNot2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) > maxCategories]

for col in pd_data[colsNot2dummy]:
    pd_data[col] = pd.factorize(pd_data[col])[0]

pd_data = pd.get_dummies(pd_data, dummy_na=True, columns=cols2dummy)
filename = CODE_FOLDER + 'data/{}_[OH-thresh{}].p'.format(name, maxCategories)
pd_data.to_hdf(filename, 'wb')
print(filename)
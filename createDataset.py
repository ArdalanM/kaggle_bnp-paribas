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

pdtrain = loadFileinZipFile(CODE_FOLDER + "data/train.csv.zip", "train.csv")
pdtest = loadFileinZipFile(CODE_FOLDER + "data/test.csv.zip", "test.csv")
pdtest['target'] = -1
pd_data = pdtrain.append(pdtest)
pd_data['nb_nan'] = pd_data.isnull().sum(1)


cat_var = pdtrain.select_dtypes(["object"]).columns
cont_var = pdtrain.select_dtypes(["float", "int"]).columns


# LE cat var
pd_data = pdtrain.append(pdtest)
for col in pd_data[cat_var]:
    print(col)
    pd_data[col] = pd.factorize(pd_data[col])[0]
pd_data.to_hdf(CODE_FOLDER + 'data/pd_data_[LEcat].p', 'wb')



# LE cat var
pd_data = pdtrain.append(pdtest)
for col in pd_data[cat_var]:
    print(col)
    pd_data[col] = pd.factorize(pd_data[col])[0]
for col in pd_data[cont_var]:
    pd_data.loc[pd_data[col].isnull(), col] = pd_data[col].mean()

pd_data.to_hdf(CODE_FOLDER + 'data/pd_data_[LEcat]_fillNANcont_var.p', 'wb')




# OH cat and LE rest
maxCategories = 300
pd_data = pdtrain.append(pdtest)
cols2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) <= maxCategories]
colsNot2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) > maxCategories]
for col in pd_data[colsNot2dummy]: pd_data[col] = pd.factorize(pd_data[col])[0]
pd_data = pd.get_dummies(pd_data, dummy_na=True, columns=cols2dummy)

pd_data.to_hdf(CODE_FOLDER + 'data/pd_data_[DummyCat-thresh300].p', 'wb')




# One Hot Cat var and LE the rest (one column)
maxCategories = 10
pd_data = pdtrain.append(pdtest)
cols2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) <= maxCategories]
colsNot2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) > maxCategories]
for col in pd_data[colsNot2dummy]: pd_data[col] = pd.factorize(pd_data[col])[0]
pd_data = pd.get_dummies(pd_data, dummy_na=True, columns=cols2dummy)

pd_data.to_hdf(CODE_FOLDER + 'data/pd_data_[DummyCat-thresh10].p', 'wb')





# One Hot Cat var and LE the rest (one column)
maxCategories = 5
pd_data = pdtrain.append(pdtest)
cols2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) <= maxCategories]
colsNot2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) > maxCategories]
for col in pd_data[colsNot2dummy]: pd_data[col] = pd.factorize(pd_data[col])[0]
pd_data = pd.get_dummies(pd_data, dummy_na=True, columns=cols2dummy)

pd_data.to_hdf(CODE_FOLDER + 'data/pd_data_[DummyCat-thresh5].p', 'wb')




#fill NAN cont_var
# OH cat and LE rest
maxCategories = 300
pd_data = pdtrain.append(pdtest)
for col in pd_data[cont_var]:
    pd_data.loc[pd_data[col].isnull(), col] = pd_data[col].mean()
cols2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) <= maxCategories]
colsNot2dummy = [col for col in pd_data[cat_var] if len(pd_data[col].unique()) > maxCategories]
for col in pd_data[colsNot2dummy]: pd_data[col] = pd.factorize(pd_data[col])[0]
pd_data = pd.get_dummies(pd_data, dummy_na=True, columns=cols2dummy)
pd_data.to_hdf(CODE_FOLDER + 'data/pd_data_[DummyCat-thresh300]_fillNANcont_var.p', 'wb')

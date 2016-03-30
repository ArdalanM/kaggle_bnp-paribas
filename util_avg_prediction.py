from numpy.linalg._umath_linalg import eig

__author__ = 'Ardalan'

import glob, argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--folder",default="/home/ardalan/Documents/kaggle/",help="folder containing csv")
parser.add_argument("--pattern",default="*|0.4[4-9]*.csv",help="pattern to filter files in folder")
args = parser.parse_args()



folder = args.folder
pattern = args.pattern

# folder = "/home/ardalan/Documents/kaggle/bnp/diclogs/stage2/"
# pattern = "*|0.4[4-9]*.csv"



l_pred = []
l_val_scores = []

l_filename = glob.glob1(folder, pattern)

print("List of file selected :")
for f in l_filename:
    print(f)
    val_score = float(f.split('_')[-1].split(".csv")[0])
    l_val_scores.append(val_score)

for filename in l_filename:

    pd_data = pd.read_csv(folder + filename)
    test_idx = pd_data['ID'].values
    pred = pd_data['PredictedProb'].values
    l_pred.append(pred)

X = np.array(l_pred).T
pred = np.mean(X, 1)


filename = "avg_{:.4f}.csv".format(np.mean(l_val_scores))

np.savetxt(folder + filename, np.vstack((test_idx, pred)).T, delimiter=',',
           fmt='%i,%.10f', header='ID,PredictedProb', comments="")


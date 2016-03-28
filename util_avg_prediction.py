__author__ = 'Ardalan'

import glob
import pandas as pd
import numpy as np




folder = "/home/ardalan/Documents/kaggle/bnp/diclogs/stage2/"
pattern = "*44*.csv"


l_filename = glob.glob1(folder, pattern)
print(l_filename)


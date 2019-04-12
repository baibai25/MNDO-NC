# coding: utf-8
import sys
import os.path
from tqdm import tqdm
import numpy as np
import pandas as pd
from src import multivariate_os, predict_data, preprocessing
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn import over_sampling
from imblearn import combine
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

# set save path
def set_path(basename):
    name = os.path.splitext(basename)
    save_path = 'output/{}.csv'.format(name[0])
    return save_path

# Multivariate over-sampling
def mndo(pos, num_minority, name):
    pos, zero_std = multivariate_os.find_zerostd(pos, num_minority)
    pos, no_corr = multivariate_os.no_corr(pos, num_minority)
    pos = multivariate_os.mnd_os(pos, num_minority)
    mndo_df = multivariate_os.append_data(pos, zero_std, no_corr, name)
    return mndo_df

# train data + mndo data
def append_mndo(X_train, y_train, df):
    X_mndo = df.drop('Label', axis=1)
    y_mndo = df.Label
    X_mndo = np.concatenate((X_mndo, X_train), axis=0)
    y_mndo = np.concatenate((y_mndo, y_train), axis=0)
    #X_mndo = pd.concat([X_mndo, X_train])
    #y_mndo = pd.concat([y_mndo, y_train])
    return X_mndo, y_mndo

if __name__ == '__main__':
    # Load dataset
    data = pd.read_csv(sys.argv[1])
    save_path = set_path(os.path.basename(sys.argv[1]))
    name = os.path.splitext(os.path.basename(sys.argv[1]))
    print(name[0])
    X = data.drop('Label', axis=1)
    y = data.Label
    pos = data[data.Label == 1]
    pos = pos.drop('Label', axis=1)

    # Split the data
    RANDOM_STATE = 6
    X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.4, random_state=RANDOM_STATE)
    cnt = Counter(y_train)
    num_minority = int((cnt[0] - cnt[1]))
    
    #-----------------
    # Preprocessing
    #-----------------
    # Multivariate over-sampling
    mndo_df = mndo(pos, num_minority, name[0])

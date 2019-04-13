import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.datasets import fetch_datasets

# load data from imblearn.datasets
def load_dataset(data_name):

    load_data = fetch_datasets(verbose=True)[data_name]
    print(load_data.data.shape)
    print(Counter(load_data.target))

    X = pd.DataFrame(load_data.data)
    y = pd.DataFrame(load_data.target, columns=['Label'])

    return X, y


# detect nominal data
def detect_nominal(X, y):
    # calc and detect
    likely_cat = {}
    for var in X.columns:
        likely_cat[var] = 1.*X[var].nunique()/X[var].count() < 0.01

    #print(likely_cat)
    nominal = [i for i in likely_cat if likely_cat[i] == True]

    # nominal data
    X_nominal = X[nominal]
    X_nominal = pd.concat([X_nominal, y], axis=1)

    # continuous data
    X_continuous = X.drop(nominal, axis=1)
    X_continuous = pd.concat([X_continuous, y], axis=1)

    return X_nominal, X_continuous



if __name__ == '__main__':

    try:
        X, y = load_dataset(sys.argv[1])
    except KeyError:
        print('select following datasets')
        print('abalone, sick_euthyroid, thyroid_sick, arrhythmia, abalone_19')
        sys.exit()

    X_nominal, X_continuous = detect_nominal(X, y)

    # save 
    os.makedirs('./Predataset/{}'.format(sys.argv[1]), exist_ok=True)
    X_nominal.to_csv('./Predataset/{}/nominal.csv'.format(sys.argv[1]), index=False)
    X_continuous.to_csv('./Predataset/{}/continuous.csv'.format(sys.argv[1]), index=False)


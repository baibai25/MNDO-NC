import sys
import os
import numpy as np
import pandas as pd
from src import multivariate_os, nominal_os
from collections import Counter
from sklearn.model_selection import train_test_split

# Multivariate over-sampling
def mndo(pos, num_minority):
    pos, zero_std = multivariate_os.find_zerostd(pos, num_minority)
    pos, no_corr = multivariate_os.no_corr(pos, num_minority)
    pos = multivariate_os.mnd_os(pos, num_minority, zero_std, no_corr)
    
    return pos

if __name__ == '__main__':

    # Load dataset
    try:
        #data = pd.read_csv('Predataset/abalone_19/continuous.csv')
        #nominal = pd.read_csv('Predataset/abalone_19/nominal.csv')
        data = pd.read_csv('Predataset/{}/continuous.csv'.format(sys.argv[1]))
        nominal = pd.read_csv('Predataset/{}/nominal.csv'.format(sys.argv[1]))
        file_name = sys.argv[1]
    except IndexError:
        sys.exit('error: Must specify dataset file')
    except FileNotFoundError:
        sys.exit('error: No such file or directory')
         
    # split the data
    X = data.drop('Label', axis=1)
    y = data.Label
    X = np.array(X)
    y = np.array(y)

    # split positive class
    pos = data[data.Label == 1]
    pos = pos.drop('Label', axis=1)
    nominal = nominal[nominal.Label == 1]
    nominal = nominal.drop('Label', axis=1)

    # calc number of  samples to synthesize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
            shuffle=True, random_state=42)

    cnt = Counter(y_train)
    num_minority = int((cnt[-1] - cnt[1]))
    
    # over-sampling
    pos_gen = mndo(pos, num_minority)
    
    # nominal over-sampling
    key = nominal_os.distance(pos, pos_gen)
    generated_data = nominal_os.nominal_os(key, nominal, pos_gen)
    nominal_os.save(generated_data, file_name)


import sys
import os
import numpy as np
import pandas as pd
from src import multivariate_os, predict_data, preprocessing
from collections import Counter
from sklearn.model_selection import train_test_split

# set save path
def set_path(basename):
    os.makedirs('./output', exist_ok=True)
    name = os.path.splitext(basename)
    save_path = 'output/{}.csv'.format(name[0]) 
    return save_path, name[0]

# Multivariate over-sampling
def mndo(pos, num_minority, name):
    pos, zero_std = multivariate_os.find_zerostd(pos, num_minority)
    pos, no_corr = multivariate_os.no_corr(pos, num_minority)
    pos = multivariate_os.mnd_os(pos, num_minority)
    mndo_df = multivariate_os.append_data(pos, zero_std, no_corr, name)
    return mndo_df


if __name__ == '__main__':
    # Load dataset
    try:
        data = pd.read_csv(sys.argv[1])
        save_path, file_name = set_path(os.path.basename(sys.argv[1]))
    except IndexError:
        sys.exit('error: Must specify dataset file')
    except FileNotFoundError:
        sys.exit('error: No such file or directory')
    
    # split the data
    X = data.drop('Label', axis=1)
    y = data.Label

    # split positive class
    pos = data[data.Label == 1]
    pos = pos.drop('Label', axis=1)
    
    # split arrays into train and test subsets 
    RANDOM_STATE = 6
    X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.4, random_state=RANDOM_STATE)
    cnt = Counter(y_train)
    num_minority = int((cnt[0] - cnt[1]))

    # over-sampling
    mndo(pos, num_minority, file_name)

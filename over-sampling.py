import sys
import os
import argparse
import copy
import numpy as np
import pandas as pd
from src import multivariate_os, categorical_os
from collections import Counter
from sklearn.model_selection import train_test_split

# set save path
def set_path(basename):
    os.makedirs('./generated', exist_ok=True)
    name = os.path.splitext(basename) 
    save_path = 'generated/{}.csv'.format(name[0]) 
    return save_path, name[0]

def set_multiclass_path(basename, pos_label):
    os.makedirs('./generated/multiclass', exist_ok=True)
    name = os.path.splitext(basename) 
    save_path = 'generated/multiclass/{}_pos{}.csv'.format(name[0], pos_label) 
    return save_path, name[0]

# Multivariate over-sampling
def mndo(pos, num_minority):
    pos, zero_std = multivariate_os.find_zerostd(pos, num_minority)
    pos, no_corr = multivariate_os.no_corr(pos, num_minority)
    pos = multivariate_os.mnd_os(pos, num_minority, zero_std, no_corr)

    return pos

if __name__ == '__main__':
    # Load dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='dataset')
    parser.add_argument('--pos_label', type=int, default='1', help='Positive class: default=1')
    parser.add_argument('--neg_label', type=int, default='0', help='Negative class: default=0')
    parser.add_argument('--multiclass', action='store_true', help='Multiclass: default=False')
    parser.add_argument('--dummies', action='store_true',
            help='If the categorical data has already been preprocessed: default=False')
    args = parser.parse_args()
    
    try:
        data = pd.read_csv(args.data)
        pos_label = args.pos_label
        neg_label = args.neg_label

        if args.multiclass == True:
            save_path, file_name = set_multiclass_path(os.path.basename(args.data), pos_label)
        elif args.multiclass == False:
            save_path, file_name = set_path(os.path.basename(args.data))
    except IndexError:
        sys.exit('error: Must specify dataset file')
    except FileNotFoundError:
        sys.exit('error: No such file or directory')

    # split categorical data
    """
    contraceptive: ['a5', 'a6', 'a9']
    thyroid:
    ['Sintoma2', 'Sintoma3', 'Sintoma4', 'Sintoma5', 'Sintoma6',
    'Sintoma7', 'Sintoma8', 'Sintoma9', 'Sintoma10', 'Sintoma11',
    'Sintoma12', 'Sintoma13', 'Sintoma14', 'Sintoma15', 'Sintoma16']
    """
    if args.dummies == True:
        attribute = [] 
        categorical = data[attribute]
        categorical = pd.concat([categorical, data['Label']], axis=1)
        data = data.drop(attribute, axis=1)

    elif args.dummies == False:
        attribute = [data.columns[i] for i in range(data.shape[1]) if data.dtypes[i] == 'object']
        categorical = data[attribute]
        categorical = pd.get_dummies(categorical)
        categorical = pd.concat([categorical, data['Label']], axis=1)
        data = data.drop(attribute, axis=1)

    # split the data
    X = data.drop('Label', axis=1)
    y = data.Label
    X = np.array(X)
    y = np.array(y)

    # split positive class
    pos = data[data.Label == pos_label]
    pos = pos.drop('Label', axis=1)
    positive = copy.copy(pos) # copy original data
    categorical = categorical[categorical.Label == pos_label]
    categorical = categorical.drop('Label', axis=1)

    # calc number of  samples to synthesize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
            shuffle=True, random_state=42)

    cnt = Counter(y_train)
    num_minority = int((cnt[neg_label] - cnt[pos_label]))
    
    # over-sampling
    pos_gen = mndo(pos, num_minority)
 
    # categorical over-sampling
    key = categorical_os.distance(positive, pos_gen)
    generated_data = categorical_os.categorical_os(key, categorical, pos_gen, pos_label)
    categorical_os.save(generated_data, save_path)

import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from src import multivariate_os, predict_data, preprocessing
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn import over_sampling
from imblearn import combine
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from imblearn.metrics import classification_report_imbalanced

# set save path
def set_path(basename):
    os.makedirs('./output', exist_ok=True)
    name = os.path.splitext(basename)
    save_path = 'output/{}.csv'.format(name[0]) 
    return save_path, name[0]

# train data + mndo data
def append_mndo(X_train, y_train, df):
    X_mndo = df.drop('Label', axis=1)
    y_mndo = df.Label
    X_mndo = np.concatenate((X_mndo, X_train), axis=0)
    y_mndo = np.concatenate((y_mndo, y_train), axis=0)

    return X_mndo, y_mndo


if __name__ == '__main__':
    # Load dataset
    try:
        data = pd.read_csv(sys.argv[1])
        mndo_generated = pd.read_csv(sys.argv[2])
        save_path, file_name = set_path(os.path.basename(sys.argv[1]))
    except IndexError:
        sys.exit('error: Must specify dataset file')
    except FileNotFoundError:
        sys.exit('error: No such file or directory')

    # split the data
    X = data.drop('Label', axis=1)
    y = data.Label
    X = np.array(X)
    y = np.array(y)

    # calc number of  samples to synthesize
    RANDOM_STATE=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
            shuffle=True, random_state=RANDOM_STATE)

    print('y_train: {}'.format(Counter(y_train)))
    print('y_test: {}'.format(Counter(y_test)))

    # SMOTE k-NN error handling
    """
    if cnt[1] < 6:
        print("Can't apply SMOTE. Positive class samples is very small.")
        print("See : https://github.com/scikit-learn-contrib/imbalanced-learn/issues/27")
        sys.exit()
    """

    #-----------------
    # Preprocessing
    #-----------------
    # Multivariate over-sampling
    X_mndo, y_mndo = append_mndo(X_train, y_train, mndo_generated)
    print('y_mndo: {}'.format(Counter(y_mndo)))
    
    sc = Normalizer(norm='l2')
    sc.fit(X_mndo)
    X_mndo = sc.transform(X_mndo) 
    
    sc = Normalizer(norm='l2')
    sc.fit(X_test)
    X_test = sc.transform(X_test) 

    # scaling 
    #os_list, X_test_scaled = preprocessing.normalization([X_mndo, y_mndo], X_test)
    #os_list, X_test_scaled = preprocessing.standardization(os_list, X_test)

    #-------------
    # Learning
    #-------------
    tuned_parameters = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}

    #score = metrics.make_scorer(metrics.f1_score, pos_label = 1)
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='f1', verbose=2)
    clf.fit(X_train, y_train)
    
    #print(clf.cv_results_)
    print(clf.best_params_)
    print(clf.best_score_)
    y_pred_bal = clf.predict(X_test)
    print(classification_report_imbalanced(y_test, y_pred_bal))

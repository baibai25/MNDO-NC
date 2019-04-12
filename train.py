import sys
import os
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
    print('y_train: {}'.format(Counter(y_train)))
    print('y_test: {}'.format(Counter(y_test)))
   
    # SMOTE k-NN error handling
    if cnt[1] < 6:
        print("Can't apply SMOTE. Positive class samples is very small.")
        print("See : https://github.com/scikit-learn-contrib/imbalanced-learn/issues/27")
        sys.exit()

    #-----------------
    # Preprocessing
    #-----------------
    # Multivariate over-sampling
    mndo_df = mndo(pos, num_minority, file_name)

    X_mndo, y_mndo = append_mndo(X_train, y_train, mndo_df)
    #print('y_mndo: {}'.format(Counter(y_mndo)))

    for i in tqdm(range(100), desc="Preprocessing", leave=False):
        # Apply over-sampling
        sm_reg = over_sampling.SMOTE(kind='regular', random_state=RANDOM_STATE)
        sm_b1 = over_sampling.SMOTE(kind='borderline1', random_state=RANDOM_STATE)
        sm_b2 = over_sampling.SMOTE(kind='borderline2', random_state=RANDOM_STATE)
        sm_enn = combine.SMOTEENN(random_state=RANDOM_STATE)
        sm_tomek = combine.SMOTETomek(random_state=RANDOM_STATE)
        ada = over_sampling.ADASYN(random_state=RANDOM_STATE)
        
        X_reg, y_reg = sm_reg.fit_sample(X_train, y_train)
        X_b1, y_b1 = sm_b1.fit_sample(X_train, y_train)
        X_b2, y_b2 = sm_b2.fit_sample(X_train, y_train)
        X_enn, y_enn = sm_enn.fit_sample(X_train, y_train)
        X_tomek, y_tomek = sm_tomek.fit_sample(X_train, y_train)
        X_ada, y_ada = ada.fit_sample(X_train, y_train)
        os_list = [[X_reg, y_reg], [X_b1, y_b1], [X_b2, y_b2], [X_enn, y_enn],
                [X_tomek, y_tomek], [X_ada, y_ada], [X_mndo, y_mndo]]
       
        # scaling 
        os_list, X_test_scaled = preprocessing.normalization(os_list, X_test)
        #os_list, X_test_scaled = preprocessing.standardization(os_list, X_test)

    #-------------
    # Learning
    #-------------
    for i in tqdm(range(100), desc="Learning", leave=False):
        svm_clf = []
        pred_tmp = []

        #svm
        for i in range(len(os_list)):
            svm_clf.append(svm.SVC(gamma='auto', random_state=RANDOM_STATE, probability=True).fit(os_list[i][0], os_list[i][1]))
            
        for i in range(len(svm_clf)):
            # calc auc
            prob = svm_clf[i].predict_proba(X_test_scaled[i])[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, prob, pos_label=1)
            roc_auc_area = auc(fpr, tpr)
            pred_tmp.append(predict_data.calc_metrics(y_test, svm_clf[i].predict(X_test_scaled[i]), roc_auc_area, i))
       
        # tree
        tree_clf = []
        for i in range(len(os_list)):
            tree_clf.append(DecisionTreeClassifier(random_state=RANDOM_STATE).fit(os_list[i][0], os_list[i][1]))
            
        for i in range(len(tree_clf)):
            # calc auc
            prob = tree_clf[i].predict_proba(X_test_scaled[i])[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, prob, pos_label=1)
            roc_auc_area = auc(fpr, tpr)
            pred_tmp.append(predict_data.calc_metrics(y_test, tree_clf[i].predict(X_test_scaled[i]), roc_auc_area, i))

        #k-NN
        k=3
        knn_clf = []
        for i in range(len(os_list)):
            knn_clf.append(KNeighborsClassifier(n_neighbors=k).fit(os_list[i][0], os_list[i][1]))
            
        for i in range(len(knn_clf)):
            # calc auc
            prob = knn_clf[i].predict_proba(X_test_scaled[i])[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, prob, pos_label=1)
            roc_auc_area = auc(fpr, tpr)
            pred_tmp.append(predict_data.calc_metrics(y_test, knn_clf[i].predict(X_test_scaled[i]), roc_auc_area, i))

    pred_df = pd.DataFrame(pred_tmp)
    pred_df.columns = ['os', 'Sensitivity', 'Specificity', 'Geometric mean', 'AUC']
   
    # export resualt
    pred_df.to_csv(save_path, index=False)

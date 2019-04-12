from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

# Normalize
def normalization(os_list, X_test):
    X_test_scaled = []
    for i in range(len(os_list)):
        sc = Normalizer(norm='l2')
        sc.fit(os_list[i][0])
        os_list[i][0] = sc.transform(os_list[i][0]) 
        X_test_scaled.append(sc.transform(X_test))
    return os_list, X_test_scaled

# Standardize
def standardization(os_list, X_test):
    X_test_scaled = []
    for i in range(len(os_list)):
        sc = StandardScaler()
        sc.fit(os_list[i][0])
        os_list[i][0] = sc.transform(os_list[i][0])
        X_test_scaled.append(sc.transform(X_test))

    return os_list, X_test_scaled

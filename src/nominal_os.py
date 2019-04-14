import os
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# calc euclidean distance
# return: index of nearest samples 
def distance(pos, pos_gen):
    # calc distance
    distance = []
    for i in range(len(pos_gen)):
        diff = np.array(pos_gen.iloc[i]) - np.array(pos)
        tmp = [np.linalg.norm(diff[j]) for j in range(len(diff))]
        distance.append(tmp) 
    del tmp

    df_dist = pd.DataFrame(distance).T
    df_dist.index = pos.index
    
    # find nearest samples (k=5)
    key = []
    for i in range(len(df_dist.columns)):
        sort = np.argsort(df_dist[df_dist.columns[i]])
        key.append(sort[0:5])

    key = np.array(key)
        
    return key


# extract nearest nominal samples
# nominal over-sampling
def nominal_os(key, nominal, pos_gen): 
    # extract nearest nominal samples
    nominal_gen = []
    for i in range(key.shape[0]):
        tmp = [nominal.iloc[key[i][j]] for j in range(key.shape[1])]
        nn_df = pd.DataFrame(tmp)

        # voting
        tmp = []
        for k in range(nn_df.shape[1]):
            counter = Counter(nn_df[nn_df.columns[k]])
            value, freq = counter.most_common(1)[0]
            tmp.append(value)
        nominal_gen.append(tmp)

    nominal_gen = pd.DataFrame(nominal_gen)
    df = pd.concat([nominal_gen, pos_gen], axis=1)
    df['Label'] = 1
    
    return df

def save(generated_data, file_name):
    os.makedirs('./Predataset/{}'.format(file_name), exist_ok=True)
    generated_data.to_csv('./Predataset/{}/generated.csv'.format(file_name), index=False)
    print('Generated data is saved in ./Predataset/{}/generated.csv'.format(file_name))    

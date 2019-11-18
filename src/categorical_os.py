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


# extract nearest categorical samples
# categorical over-sampling
def categorical_os(key, categorical, pos_gen, pos_label): 
    # extract nearest categorical samples
    categorical_gen = []
    for i in range(key.shape[0]):
        tmp = [categorical.iloc[key[i][j]] for j in range(key.shape[1])]
        nn_df = pd.DataFrame(tmp)

        # voting
        tmp = []
        for k in range(nn_df.shape[1]):
            counter = Counter(nn_df[nn_df.columns[k]])
            value, freq = counter.most_common(1)[0]
            tmp.append(value)
        categorical_gen.append(tmp)

    categorical_gen = pd.DataFrame(categorical_gen)
    categorical_gen.columns = categorical.columns 
    df = pd.concat([pos_gen, categorical_gen], axis=1)
    df['Label'] = pos_label
    
    return df

def save(generated_data, save_path):
    generated_data.to_csv(save_path, index=False)
    print('Generated data is saved in {}'.format(save_path))    

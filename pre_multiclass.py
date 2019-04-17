import os
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# set save path
def set_path(basename):
    os.makedirs('./Predataset/multiclass', exist_ok=True)
    name = os.path.splitext(basename)
    save_path = './Predataset/multiclass/{}.csv'.format(name[0]) 
    
    return save_path, name[0]

def label_encoder(df):
    le = LabelEncoder()
    le.fit(df['Label'])
    df['Label'] = le.transform(df['Label'])

    return df
    
    
def detect_class(df):
    # get keys and negative class
    cnt = Counter(df['Label'])
    negative = max(cnt, key=cnt.get)
    label = cnt.keys()
    print(cnt)
    print('Negative label:', negative)
    
    # calc Imbalanced ratio
    positive = []
    for i in label:
        ir = cnt[negative] / cnt[i]
        
        if ir > 1.4:
            positive.append(i)
    
    print('Positive label:', positive)
    
    return negative, positive
         

if __name__ == '__main__':
    # Load dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='dataset')
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.data)
        save_path, file_name = set_path(os.path.basename(args.data))
    except IndexError:
        sys.exit('error: Must specify dataset file')
    except FileNotFoundError:
        sys.exit('error: No such file or directory')
        
    # label encoding
    if df['Label'].dtypes == 'object':
        df = label_encoder(df)
        
    # get keys and each classes
    negative, positive = detect_class(df)
    classes = [['Negative', negative], ['Positive', positive]]
    
    
    # export
    df.to_csv(save_path, index=False)
    
    label_path = './Predataset/multiclass/{}_label.txt'.format(file_name)
    with open(label_path, mode='w') as f:
        f.write(str(classes))


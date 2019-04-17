import glob, os, sys
import pandas as pd

def get_name(files):
    name = []
    for i in range(len(files)):
        basename = os.path.basename(files[i])
        split = os.path.splitext(basename)
        name.append(split[0])
    return name

if __name__ == '__main__':
    
    # exception handling
    os.makedirs('./result/data', exist_ok=True)

    if (os.path.isdir('./output') == True):
        files = glob.glob('./output/*.csv')
    else:
        print('Error: Not found ./output')
        sys.exit()
         
    if files == []:
        print('Error: Not found csv file')
        sys.exit()
    
    # aggregate predict results
    name = get_name(files)
    ind = ['sm', 'b1', 'b2', 'enn', 'tom', 'ada', 'mnd']
    col = ['os', 'Sensitivity', 'Specificity', 'G-mean', 'F-1', 'MCC', 'AUC'] 

    for i in range(len(ind)):
        svm = pd.DataFrame(index=[], columns=col)
        tree = pd.DataFrame(index=[], columns=col)
        knn = pd.DataFrame(index=[], columns=col)

        for j in range(len(files)):
            data = pd.read_csv(files[j])
            a = data[data['os'] == ind[i]]

            svm = pd.concat([svm, a.iloc[[0]]], ignore_index=True)
            tree = pd.concat([tree, a.iloc[[1]]], ignore_index=True)
            knn = pd.concat([knn, a.iloc[[2]]], ignore_index=True)
        
        df = pd.concat([svm, tree, knn], axis=1)
        df.index = name
        df.drop('os', axis=1, inplace=True)
        path = 'result/data/{}.csv'.format(ind[i])
        df.to_csv(path)

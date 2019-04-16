import sys
import os

# load file
try:
    directory = sys.argv[1]
    filelist = os.listdir(directory)
except IndexError:
    sys.exit('error: Must specify dataset directory')
except FileNotFoundError:
    sys.exit('error: No such file or directory')

# preprocess for all files
for a in filelist:
    path = os.path.join(directory, a)
   
    # if not directory
    if (os.path.isdir(path) == False):
        # open file
        file = open(path, 'r').readlines()
        basename = os.path.basename(path)
        name = os.path.splitext(basename)
        os.makedirs('./Predataset', exist_ok=True)
        output = 'Predataset/{}.csv'.format(name[0])

        # preprocessing
        file2 = []
        attribute = ''
        for line in file:
            #extract attribute
            if (line.find('@inputs') != -1):
                attribute = line.replace('@inputs', '')
                attribute = attribute.replace(' ', '')
                attribute = attribute.strip()
                attribute += ',Label\n'
                #print(attribute)
            elif (line.find('@input') != -1):
                attribute = line.replace('@input', '')
                attribute = attribute.replace(' ', '')
                attribute = attribute.strip()
                attribute += ',Label\n'

            #delete strings
            if (line.find('@') != -1):
                del line
            
            #replace class label
            #positive=1, negative=0
            else:
                if (line.find('positive') != -1):
                    pos = line.replace('positive', '1')
                    file2.append(pos)
                elif (line.find('negative') != -1):
                    neg = line.replace('negative', '0')
                    file2.append(neg)
        file2.insert(0, attribute)
        
        # save as new file
        out_file = open(output, 'w')
        for i in range(len(file2)):
            out_file.write(file2[i])
        out_file.close()


import sys
import pandas as pd

nominal = pd.read_csv('../Predataset/{}/nominal.csv'.format(sys.argv[1]))
continuous = pd.read_csv('../Predataset/{}/continuous.csv'.format(sys.argv[1]))

print('Nominal', nominal.shape)
print('Continuous', continuous.shape)

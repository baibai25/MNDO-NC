import re
import pandas as pd
from imblearn import metrics

# calc evaluate metrics
def calc_metrics(y_test, pred, auc, i):
        sen = metrics.sensitivity_score(y_test, pred, pos_label=1)   
        spe = metrics.specificity_score(y_test, pred, pos_label=1)   
        geo = metrics.geometric_mean_score(y_test, pred, pos_label=1)   
        index = ['sm', 'b1', 'b2', 'enn', 'tom', 'ada', 'mnd'] 
        metrics_list = [index[i], sen, spe, geo, auc]
        return metrics_list

# convert classification report to dataframe
def report_to_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return(report_df)

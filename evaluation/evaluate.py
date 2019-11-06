from sklearn.metrics import *


def evaluate(y_true, y_pred, verbose=False):
    
    '''
    This function return default 4 metrics
    '''
    
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec= precision_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    
    if verbose:
        print('--------evaluation---------')
        print('accuracy:', acc)
        print('recall:', rec)
        print('precision:', prec)
        print('f1 score:', f1)
        print('---------------------------')
        
    return acc, rec, prec, f1
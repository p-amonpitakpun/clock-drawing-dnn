import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from .evaluate import evaluate
import pickle as pkl  

def run_benchmark(get_split_data,
              get_model,
              data,
              k=1, 
              epochs=100,
              model_name='Model',
              verbose=0,
              figsize=(15,15)
             ):
    
    '''
    This function will run performance benchmark on the given model.
    Input:
        get_split_data : function to return sampled test train data. It must reture x_test, x_train, y_test, y_train in this
                        specified order.
        get_model : function to return a model to run benchmark on
        data : data numpy
        k : iteration to run the tests (higher k ~ better avg. value)
        epochs : number of epochs in training loop
        model_name (default='Model') : Name for the model, used in plotting ROC Curves
        verbose (default=0) : If not zero, then display benchmarking result/logs in each iteration
        
    Output:
        models : list of models which were generated & trained during the benchmarking
        plots : calculated plot data

    '''
    
    acc, rec, prec, f1 = [], [], [], []
    fp, tp, thresh = [], [], []
    mean_tp, _tp = [], []
    domain = np.linspace(0, 1, 100)
    aucs = []
    models = []
    histories = []
    for i in range(k):
        if verbose:
            print('ITER:',i)
        x_test, x_train, y_test, y_train = get_split_data(data) #data should randomly shuffle
    
        #train & predict
        model = get_model(x_test.shape[1])
        history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),verbose=verbose)
        histories.append(history)
        models.append(model) #store model
        y_pred = model.predict(x_test)

        #metric eval for thresh = 0.5
        y2 = np.array(y_pred)
        y2[y2 >= 0.5] = 1
        y2[y2 < 0.5]  = 0
        t = evaluate(y_test, y2,verbose=verbose)

        #append the evaluation results to log
        acc.append(t[0])
        rec.append(t[1])
        prec.append(t[2])
        f1.append(t[3])

        #ROC for other thresh
        f, t, p = roc_curve(y_test, y_pred)
        _tp.append(np.interp(domain, f, t))

        #AUC
        a = auc(f, t)

        #logging
        fp.append(f)
        tp.append(t)
        thresh.append(p)
        aucs.append(a)
        
    if verbose:
        print('-----thresh=0.5--------')
        print('mean acc:',np.mean(acc))
        print('mean recall:',np.mean(rec))
        print('mean precision:',np.mean(prec))
        print('mean f1:',np.mean(f1))
        print('-----------------------')
        plt.title('Evaluation for 10 runs (thresh=0.5)')
        plt.plot(np.arange(k), acc, label='acc')
        plt.plot(np.arange(k), rec, label='recall')
        plt.plot(np.arange(k), prec,label='precision')
        plt.plot(np.arange(k), f1,  label='f1')
        plt.xlabel('iteration')
        plt.ylabel('score')
        plt.legend()
        plt.show()
        
    #Calculate mean and std of roc curve
    mean_tp = np.mean(_tp, axis=0)
    std_tp = np.std(_tp, axis=0)
    upper_tp = mean_tp + std_tp
    lower_tp = mean_tp - std_tp
    mean_auc = auc(domain, mean_tp)
    
    #ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    plt.figure(figsize=figsize)
    plt.title('ROC Curve')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.plot([0,1],[0,1],label='Chance', linestyle='dashed',color='red')
    
    #plot ROC Curve
    for i in range(k):
        plt.plot(fp[i], tp[i], label='%s iteration:%d AUC=%.2f' % (model_name,i,aucs[i]), alpha=0.2)
    plt.plot(domain, mean_tp, color='blue', alpha=1, label='average AUC=%.2f' % mean_auc)
    # plt.plot(domain, upper_tp, color='gray', alpha=0.1)
    # plt.plot(domain, lower_tp, color='gray', alpha=0.1)
    plt.fill_between(domain, upper_tp, lower_tp, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.legend()
    plt.show()
    

    plots = {'mean_tp':mean_tp, 'std_tp':std_tp, 'upper_tp':upper_tp, \
            'lower_tp':lower_tp, 'mean_auc':mean_auc, 'domain':domain,
    }

    results = {'y_pred': y_pred, 'histories':histories}

    return models, plots, results

def save_log(path, log):
    pass
    # pkl.dump(os.path.join(path, log))
import datetime as dt
import time
import tzlocal as tz
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display


def get_time(verbose=False):
    """Helper function to return current time.
    Uses tzlocal to display time in local tz, if available."""
    try: 
        now =  dt.datetime.now(tz.get_localzone())
        tic = time.time()
    except:
        now = dt.datetime.now()
        tic = time.time()
        print("[!] Returning time without tzlocal.")       
    return now,tic
        
    
def get_report(model,X_test,y_test,as_df=False,label="TEST DATA"):
    """Get classification report from sklearn and converts to DataFrame"""
    ## Get Preds and report
    y_hat_test = model.predict(X_test)
    scores = metrics.classification_report(y_test, y_hat_test,
                                          output_dict=as_df) 
    ## convert to df if as_df
    if as_df:
        report = pd.DataFrame(scores).T.round(2)
        report.iloc[2,[0,1,3]] = ''
        return report
    else:
        header="\tCLASSIFICATION REPORT"
        if len(label)>0:
            header += f" - {label}"
        dashes='---'*20
        print(f"{dashes}\n{header}\n{dashes}")
        print(scores)
        
        
        

    
def fit_and_time_model(model, X_train,y_train,X_test,y_test,
                      fit_kws={}, scoring="accuracy",normalize='true',
                       fmt="%m/%d/%y-%T", verbose=True):
    """Fits the provided model and evaluates using provided data."""
    if X_test.ndim==1:
        raise Exception('The arg order has changed to X_train,y_train,X_test,y_test')

    ## Time
    start,tic = get_time()
    if verbose: 
        print(f"[i] Training started at {start.strftime(fmt)}:")
        
    model.fit(X_train, y_train,**fit_kws)
    
    ## Calc stop time and elapse
    stop,toc = get_time()
    elapsed = toc-tic


            
            
    ## Get model scores
    scorer = metrics.get_scorer(scoring)
    scores_dict ={f'Train':scorer(model,X_train,y_train),  
                  f'Test':scorer(model, X_test,y_test)}
    scores_dict['Difference'] = scores_dict['Train'] - scores_dict['Test']
    scores_df = pd.DataFrame(scores_dict,index=[scoring])
    
    ## Time and report back
    if verbose:
#         print(f"[i] Training completed at {stop.strftime(fmt)}")
        if elapsed >120:
            print(f"\tTraining time was {elapsed/60:.4f} minutes.")
        else:
            print(f"\tTraining time was {elapsed:.4f} seconds.")
    print("\n",scores_df.round(2),"\n")
    
    ## Plot Confusion Matrix and display classification report
    get_report(model,X_test,y_test,as_df=False)
    
    fig,ax = plt.subplots(figsize=(10,5),ncols=2)
    metrics.plot_confusion_matrix(model,X_test,y_test,normalize=normalize,
                                  cmap='Blues',ax=ax[0])
    metrics.plot_roc_curve(model,X_test,y_test,ax=ax[1])
    ax[1].plot([0,1],[0,1],ls=':')
    ax[1].grid()
    fig.tight_layout()
    plt.show()
    return model


def evaluate_classification(model, X_test,y_test,normalize='true'):
    ## Plot Confusion Matrix and display classification report
    get_report(model,X_test,y_test,as_df=False)
    
    fig,ax = plt.subplots(figsize=(10,5),ncols=2)
    metrics.plot_confusion_matrix(model,X_test,y_test,normalize=normalize,
                                  cmap='Blues',ax=ax[0])
    metrics.plot_roc_curve(model,X_test,y_test,ax=ax[1])
    ax[1].plot([0,1],[0,1],ls=':')
    ax[1].grid()
    fig.tight_layout()
    plt.show()




def evaluate_grid(grid,X_test,y_test,X_train=None,y_train=None):
    print('The best parameters were:')
    print("\t",grid.best_params_)
    
    model = grid.best_estimator_    

    print('\n[i] Classification Report')
    evaluate_classification(model, X_test,y_test,X_train=X_train,y_train=y_train)
    
    
    
def get_importance(tree, X_train_df, top_n=20,figsize=(10,10)):
    
    df_importance = pd.Series(tree.feature_importances_,
                              index=X_train_df.columns)
    df_importance.sort_values(ascending=True).tail(top_n).plot(
        kind='barh',figsize=figsize,title='Feature Importances',
    ylabel='Feature',)
    return df_importance



def show_tree(clf,X_train_df,figsize=(60,25),class_names=['Died','Survived'],
              savefig=False,fname='titanic_tree.pdf',max_depth=None,):
    from sklearn.tree import plot_tree
    fig,ax = plt.subplots(figsize=figsize)
    plot_tree(clf,filled=True,rounded=True,proportion=True,
              feature_names=X_train_df.columns,
              class_names=class_names,ax=ax);
    fig.tight_layout()
    
    if savefig:
        fig.savefig(fname, dpi=300,orientation='landscape')
    return fig




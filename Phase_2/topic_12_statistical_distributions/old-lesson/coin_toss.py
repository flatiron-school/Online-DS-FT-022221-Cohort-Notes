import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
def coin_toss(n_flips = 2,results=None,verbose=True,p=0.5):
    
    if results is None:
        coin_flips=[]
    else:
        coin_flips = results.copy()
        
    for n in range(n_flips):
        i = np.random.rand()
        res = 'H' if i >= p else 'T'
        coin_flips.append(res)

        if verbose: 
            print(f"Toss {n+1}:   {res}")
    
    return coin_flips


def compare_results(results):
    fig,ax = plt.subplots(ncols=2,figsize=(8,4))
    results_df = pd.DataFrame({'results':results,
                               'trial #':range(1,len(results)+1)})
    
    results_df['results'].value_counts().plot(kind='bar',rot=0,ax=ax[0])
    ax[0].set(ylabel='Count',title='Histogram')
    
    results_df['results'].value_counts(normalize=True).plot(kind='bar',
                                                            rot=0,
                                                            ax=ax[1],
                                                           color='g')
    ax[1].set(ylabel='Probability',title='Probability Mass Function (PMF)')
    plt.suptitle(f'Histogram vs PMF for {len(results_df)} trials')
    fig.tight_layout()
    return fig, ax
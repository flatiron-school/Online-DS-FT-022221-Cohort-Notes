## Our Function From Yesterday(modified) 
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_distribution(df, col=None, verbose=True,boxplot=True):
    """Plots a histogram + KDE and a boxplot of the column.
    Also prints statistics for skew, kurtosis, and normaltest. 

    Args:
        df_ (DataFrame): DataFrame containing column to plot
        col (str): Name of the column to plot.
        verbose (bool, optional): If true show figure and print stats. Defaults to True.
        boxplot (bool, optional): If true, return subplots with boxplot. Defaults to True.

    Returns:
        fig : Matplotlib Figure
        ax : Matplotlib Axis
    """
    
    # df = df_.copy()

    if col is None:
        data = df.copy()
        name = data.name
    else:
        data = df[col].copy()
        name = col

    ## Calc mean and mean skew and curtosis
    median = data.median().round(2)
    mean = data.mean().round(2)
    skew_val = round(stats.skew(data, bias=False),2)
    kurt_val = round(stats.kurtosis(data,bias=False),2)
    
    
    ## Plot distribution 
    fig, ax = plt.subplots(nrows=2,figsize=(10,8))
    sns.histplot(data,alpha=0.5,stat='density',ax=ax[0])
    sns.kdeplot(data,color='green',label='KDE',ax=ax[0])
    ax[0].set(ylabel='Density',title=name.title())
    ax[0].set_title(F"Distribution of {name}")
    ax[0].axvline(median,label=f'median={median:,}',color='black')
    ax[0].axvline(mean,label=f'mean={mean:,}',color='black',ls=':')
    ax[0].legend()
    
    ## Plot Boxplot
    sns.boxplot(data,x=col,ax=ax[1])
    
    ## Tweak Layout & Display
    fig.tight_layout()
    
    ## Delete boxplot if unwanted
    if boxplot == False:
        fig.delaxes(ax[1])
    
    if verbose:
        plt.show()

        print('[i] Distribution Stats:')
        print(f"\tSkew = {skew_val}")
        print(f"\tKurtosis = {kurt_val}")
        print(f"\tN = {len(data):,}")


        ## Test for normality
        result = stats.normaltest(data)
        print('\n',result)
        if result[1]<.05:
            print('\t- p<.05: The distribution is NOT normally distributed.')
        elif result[1] >=.05:
            print('\t- p>=.05: The distribution IS normally distributed')
    
    return fig, ax



import scipy.stats as stats

def Cohen_d(group1, group2, correction = False):
    """Compute Cohen's d
    d = (group1.mean()-group2.mean())/pool_variance.
    pooled_variance= (n1 * var1 + n2 * var2) / (n1 + n2)

    Args:
        group1 (Series or NumPy array): group 1 for calculating d
        group2 (Series or NumPy array): group 2 for calculating d
        correction (bool): Apply equation correction if N<50. Default is False. 
            - Url with small ncorrection equation: 
                - https://www.statisticshowto.datasciencecentral.com/cohens-d/ 
    Returns:
        d (float): calculated d value
         
    INTERPRETATION OF COHEN's D: 
    > Small effect = 0.2
    > Medium Effect = 0.5
    > Large Effect = 0.8
    
    """
    import scipy.stats as stats
    import scipy   
    import numpy as np
    N = len(group1)+len(group2)
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    ## Apply correction if needed
    if (N < 50) & (correction==True):
        d=d * ((N-3)/(N-2.25))*np.sqrt((N-2)/N)
    return d


#Your code here
# def find_outliers_Z(data):
#     """Use scipy to calculate absolute Z-scores 
#     and return boolean series where True indicates it is an outlier.

#     Args:
#         data (Series,or ndarray): data to test for outliers.

#     Returns:
#         [boolean Series]: A True/False for each row use to slice outliers.
        
#     EXAMPLE USE: 
#     >> idx_outs = find_outliers_df(df['AdjustedCompensation'])
#     >> good_data = df[~idx_outs].copy()
#     """
#     import pandas as pd
#     import numpy as np
#     import scipy.stats as stats
#     import pandas as pd
#     import numpy as np
#     ## Calculate z-scores
#     zs = stats.zscore(data)
    
#     ## Find z-scores >3 awayfrom mean
#     idx_outs = np.abs(zs)>3
    
#     ## If input was a series, make idx_outs index match
#     if isinstance(data,pd.Series):
#         return pd.Series(idx_outs,index=data.index)
#     else:
#         return pd.Series(idx_outs)
    
    
    
# def find_outliers_IQR(data):
#     """Use Tukey's Method of outlier removal AKA InterQuartile-Range Rule
#     and return boolean series where True indicates it is an outlier.
#     - Calculates the range between the 75% and 25% quartiles
#     - Outliers fall outside upper and lower limits, using a treshold of  1.5*IQR the 75% and 25% quartiles.

#     IQR Range Calculation:    
#         res = df.describe()
#         IQR = res['75%'] -  res['25%']
#         lower_limit = res['25%'] - 1.5*IQR
#         upper_limit = res['75%'] + 1.5*IQR

#     Args:
#         data (Series,or ndarray): data to test for outliers.

#     Returns:
#         [boolean Series]: A True/False for each row use to slice outliers.
        
#     EXAMPLE USE: 
#     >> idx_outs = find_outliers_df(df['AdjustedCompensation'])
#     >> good_data = df[~idx_outs].copy()
    
#     """
#     df_b=data
#     res= df_b.describe()

#     IQR = res['75%'] -  res['25%']
#     lower_limit = res['25%'] - 1.5*IQR
#     upper_limit = res['75%'] + 1.5*IQR

#     idx_outs = (df_b>upper_limit) | (df_b<lower_limit)

#     return idx_outs


def prep_data_for_tukeys(data, data_col = 'data',group_col='group'):
    """Accepts a dictionary with group names as the keys 
    and pandas series as the values. 
    
    Returns a dataframe ready for tukeys test:
    - with a 'data' column and a 'group' column for sms.stats.multicomp.pairwise_tukeyhsd 
    
    Example Use:
    df_tukey = prep_data_for_tukeys(grp_data)
    tukey = sms.stats.multicomp.pairwise_tukeyhsd(df_tukey['data'], df_tukey['group'])
    tukey.summary()
    """
    import pandas as pd
    
    df_tukey = pd.DataFrame(columns=[data_col,group_col])
    for k,v in  data.items():
        grp_df = v.rename(data_col).to_frame() 
        grp_df[group_col] = k
        df_tukey=pd.concat([df_tukey, grp_df],axis=0)

    ## New lines added to ensure compatibility with tukey's test
    df_tukey[group_col] = df_tukey[group_col].astype('str')
    df_tukey[data_col] = df_tukey[data_col].astype('float')
    return df_tukey



## Check for outliers
from scipy import stats
def find_outliers_z(data):
    """Detects outliers using the Z-score>3 cutoff.
    Returns a boolean Series where True=outlier"""
    zFP = np.abs(stats.zscore(data))
    zFP = pd.Series(zFP, index=data.index)
    idx_outliers = zFP > 3
    return idx_outliers


def find_outliers_IQR(data):
    """Detects outliers using the 1.5*IQR thresholds.
    Returns a boolean Series where True=outlier"""
    res = data.describe()
    q1 = res['25%']
    q3 = res['75%']
    thresh = 1.5*(q3-q1)
    idx_outliers =(data < (q1-thresh)) | (data > (q3+thresh))
    return idx_outliers

def multiplot(df_model,figsize=(10,10),cmap="Reds"):
    
    corr = df_model.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True,cmap="Reds",mask=mask)
    return fig, ax
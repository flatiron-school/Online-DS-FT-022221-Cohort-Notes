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
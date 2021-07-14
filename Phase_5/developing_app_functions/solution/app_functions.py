import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default='ggplot2'
plt.rcParams['figure.figsize'] = (12,4)
import datetime as dt
import pandas_datareader as pdr



today = dt.date.today().strftime("%Y-%m-%d")

## Functions for Getting and PLotting Data
def get_data(start_date='2012-02-01',end_date=today, symbols=['FB','AAPL','GOOGL','AMZN','MSFT']):

    """Gets Stock Data from Pandas Data Reader Using Yahoo Finance.

    Args:
        start_date (str, optional): Start Date to retrieve. Defaults to '2012-02-01'.
        end_date (str, optional): End date to retrieve. Defaults to '2021'.
        symbols (list, optional): List of Stocks to retrieve. Defaults to ['FB','AAPL','GOOGL','AMZN','MSFT'].

    Returns:
        [type]: [description]
    """
    data = {}
    for stock in symbols:
        try:
            data[stock] = pdr.DataReader(stock, 'yahoo', start_date, end_date)['Adj Close']
        except Exception as e:
            print('Error with stock: '+stock)
    df = pd.DataFrame(data)#reset_index()
    return df


def plot_stocks_df(df,x="Date", stocks=None):
    """Plots the stock columns in the dataframe."""
    if df.index.name==x:
            df.reset_index(inplace=True)
    if stocks is None:
        stocks = list(df.drop(columns=x).columns)
    return px.line(df, x='Date', y=stocks)

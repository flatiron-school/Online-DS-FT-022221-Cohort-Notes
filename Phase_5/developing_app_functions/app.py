import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import datetime as dt

# import app_functions as af
# # import pandas_datareader as pdr
# from pandas_datareader import data as pdr
# import yfinance as yfin
# yfin.pdr_override()

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
    df = pd.DataFrame(data)#.reset_index()
    return df


def plot_stocks_df(df,x="Date", stocks=None):
    """Plots the stock columns in the dataframe."""
    if df.index.name==x:
        df.reset_index(inplace=True)
    if stocks is None:
        stocks = list(df.drop(columns=x).columns)
    return px.line(df, x='Date', y=stocks)




## Streamlit App Starts here
st.title('Example Streamlit App')


## Add Date Input widgets
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2012-01-01'))
end_date = st.sidebar.date_input('End Date', datetime.date.today())

## Add Text Input for stocks
stocks = st.sidebar.text_input('Stock Symbols (separate with a ,)','AMZN,MSFT,FB')

## Retrieve and plot selected data
df = af.get_data(start_date=start_date, end_date=end_date,symbols=stocks.split(','))

fig = af.plot_stocks_df(df, stocks=None)
# df.head()

st.plotly_chart(fig)


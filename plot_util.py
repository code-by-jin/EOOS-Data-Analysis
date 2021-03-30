# import necessary packages
import os, io
import numpy as np
import datetime
import pandas as pd
pd.set_option("display.precision", 3)
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings("ignore")

def myboolrelextrema(data, comparator_0, comparator_1, axis=0, order=1, mode='clip'):
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')
    datalen = data.shape[axis]
    locs = np.arange(0, datalen)
    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator_0(main, plus)
        results &= comparator_1(main, minus)
        if(~results.any()):
            return results
    return results

def myargrelextrema(data, comparator_0, comparator_1, axis=0, order=1, mode='clip'):
    results = myboolrelextrema(data, comparator_0, comparator_1,
                              axis, order, mode)
    return np.nonzero(results)

def get_local_max(df_event, column_name, th = 0.005):
    index_max = list(myargrelextrema(df_event.loc[:, column_name].values, np.greater, np.greater_equal, order=15)[0])
    index_min = list(myargrelextrema(df_event.loc[:, column_name].values, np.less, np.less_equal, order=15)[0])
    return index_max, index_min
    
def process_event(df, start_index, end_index):
    df_event = df.loc[start_index:end_index].reset_index()
    df_event.loc[:, 'feces'] = df_event.loc[:, 'feces'] - df_event.loc[0, 'feces']
    df_event.loc[:, 'urine'] = df_event.loc[:, 'urine'] - df_event.loc[0, 'urine']
    index_max_feces, index_min_feces = get_local_max(df_event, 'feces_deriv', 0.005)
    df_event['max_feces'] = df_event.iloc[index_max_feces]['feces_deriv']
    df_event['min_feces'] = df_event.iloc[index_min_feces]['feces_deriv']

    index_max_urine, index_min_urine = get_local_max(df_event, 'urine_deriv', 0.001)
    df_event['max_urine'] = df_event.iloc[index_max_urine]['urine_deriv'] 
    df_event['min_urine'] = df_event.iloc[index_min_urine]['urine_deriv']
    return df_event

def plot_event(df, start_index, end_index):
    df_event = process_event(df, start_index, end_index)
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(20, 10))    
    
    ax.plot(df_event.index, df_event.loc[:, 'feces'], color="red", marker="^", label = "feces tank")
    ax.plot(df_event.index, df_event.loc[:, 'feces']+df_event.loc[:, 'urine'], color="black", marker="*", label = "vft+vut")
    ax.set_ylabel("Feces Tank Volume (L)",color="red", fontsize=30) # set y-axis label
    if df.shape[1] > 5:
        ax.axhline(y=df_event.loc[:, 'flow'].sum()*5, color="black", label = "volume out")
    ax.set_xlabel('Seconds',fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(loc='center left', fontsize = 20)
    
    ax2 = ax.twinx()
    ax2.plot(df_event.index, df_event.loc[:, 'urine'], color="blue", marker="v", label = "urine tank")
    if df.shape[1] > 5:
        ax2.scatter(df_event.index, df_event.loc[:, 'flow'], color="blue", marker="o", label = "flowmeter")
    ax2.set_ylabel("Urine Tank Volume (L)",color="blue",fontsize=30)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.legend(loc='center right', fontsize = 20)
    return fig

def plot_deriv(df, start_index, end_index):
    df_event = process_event(df, start_index, end_index)
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(20, 10))    
    ax.plot(df_event.index, df_event.loc[:, 'feces_deriv'], color="red", marker="^", label = "feces tank")
    ax.scatter(df_event.index, df_event.loc[:, 'max_feces'], c='red', s=200, marker="s")
    ax.scatter(df_event.index, df_event.loc[:, 'min_feces'], c='red', s=200, marker="o")
    ax.set_ylabel("Feces Tank Volume (L)",color="red", fontsize=30) # set y-axis label
    ax.set_xlabel('Seconds',fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(loc='center left', fontsize = 20)
    
    ax2 = ax.twinx()
    ax2.plot(df_event.index, df_event.loc[:, 'urine_deriv'], color="blue", marker="v", label = "urine tank")
    ax2.scatter(df_event.index, df_event.loc[:, 'max_urine'], c='blue', s=200, marker="s")
    ax2.scatter(df_event.index, df_event.loc[:, 'min_urine'], c='blue', s=200, marker="o")
    ax2.set_ylabel("Urine Tank Volume (L)",color="blue",fontsize=30)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.legend(loc='center right', fontsize = 20)
    return fig
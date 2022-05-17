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
from stat_util import *

def plot_event(args, df, start_index, end_index):
    df_event, index_max_feces, index_min_feces, index_max_urine, index_min_urine = process_event(df, start_index, end_index)
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(20, 10))    
    ax.plot(df_event.index, df_event.loc[:, 'feces'], color="red", marker="^", label = "feces tank")
    ax.scatter(index_max_feces, df_event.loc[index_max_feces, 'feces'], c='red', s=200, marker="s")
    ax.scatter(index_min_feces, df_event.loc[index_min_feces, 'feces'], c='red', s=200, marker="o")
    ax.plot(df_event.index, df_event.loc[:, 'feces']+df_event.loc[:, 'urine'], color="black", marker="*", label = "vft+vut")
    ax.set_ylabel("Feces Tank Volume (L)",color="red", fontsize=30) # set y-axis label
    if args and args.flowmeter:
        ax.axhline(y=df_event.loc[:, 'flow'].sum()*args.flowmeter, color="black", label = "volume out")
    ax.set_xlabel('Seconds',fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(loc='center left', fontsize = 20)
    
    ax2 = ax.twinx()
    ax2.plot(df_event.index, df_event.loc[:, 'urine'], color="blue", marker="v", label = "urine tank")
    ax2.scatter(index_max_urine, df_event.loc[index_max_urine, 'urine'], c='blue', s=200, marker="s")
    ax2.scatter(index_min_urine, df_event.loc[index_min_urine, 'urine'], c='blue', s=200, marker="o")
    if args and args.flowmeter:
        ax2.scatter(df_event.index, df_event.loc[:, 'flow'], color="blue", marker="o", label = "flowmeter")
    ax2.set_ylabel("Urine Tank Volume (L)",color="blue",fontsize=30)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.axvline(x=0, ls='--', linewidth=1, color = 'purple', label=str(df.loc[start_index, 'date_time']))
    plt.axvline(x=df_event.index[-1], ls=':',  linewidth=1, color = 'black',  label=str(df.loc[end_index, 'date_time']))
    ax2.legend(loc='center right', fontsize = 20)
    return fig

def plot_deriv(df, start_index, end_index):
    df_event, index_max_feces, index_min_feces, index_max_urine, index_min_urine = process_event(df, start_index, end_index)
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(20, 10))    
    ax.plot(df_event.index, df_event.loc[:, 'feces_deriv'], color="red", marker="^", label = "feces tank")
    ax.scatter(index_max_feces, df_event.loc[index_max_feces, 'max_feces'], c='red', s=200, marker="s")
    ax.scatter(index_min_feces, df_event.loc[index_min_feces, 'min_feces'], c='red', s=200, marker="o")
    
    ax.set_ylabel("Feces Tank Volume (L)",color="red", fontsize=30) # set y-axis label
    ax.set_xlabel('Seconds',fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(loc='center left', fontsize = 20)
    
    ax2 = ax.twinx()
    ax2.plot(df_event.index, df_event.loc[:, 'urine_deriv'], color="blue", marker="v", label = "urine tank")
    ax2.scatter(index_max_urine, df_event.loc[index_max_urine, 'max_urine'], c='blue', s=200, marker="s")
    ax2.scatter(index_min_urine, df_event.loc[index_min_urine, 'min_urine'], c='blue', s=200, marker="o")
    ax2.set_ylabel("Urine Tank Volume (L)",color="blue",fontsize=30)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.axvline(x=0, ls='--', linewidth=1, color = 'purple', label=str(df.loc[start_index, 'date_time']))
    plt.axvline(x=df_event.index[-1], ls=':',  linewidth=1, color = 'black',  label=str(df.loc[end_index, 'date_time']))
    ax2.legend(loc='center right', fontsize = 20)
    return fig
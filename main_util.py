import os, io
import csv
import numpy as np
import datetime
from pathlib import Path
import argparse

import pandas as pd
pd.set_option("display.precision", 3)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter

import warnings
warnings.filterwarnings("ignore")
from detection_util import *

def analyze_one_day(data_dir, date, flowmeter = None, door = False):
    path_date = os.path.join(data_dir, date) # path to data of the date
    # read feces tank data
    df_feces = read_data(os.path.join(path_date, 'feces.xlsx'), date) 
    df_feces = df_feces[['weight']]
    df_feces.columns = ['feces']
    
    # read urine tank data
    df_urine = read_data(os.path.join(path_date, 'urine.xlsx'), date) 
    df_urine = df_urine[['weight']]
    df_urine.columns = ['urine']
    
    # combine urine and feces data into one dataframe
    df = pd.concat([df_feces, df_urine], axis=1)
    df = df.dropna()
    # flowmeter data
    if flowmeter:
        if flowmeter == 5:
            # read flowmeter data with sample rate of /5s
            df_flowmeter = read_data(os.path.join(path_date, 'flowmeter.xlsx'), date, is_interpolate=False) 
            # there was a time diff between fw and weight scales
            df_flowmeter.index = df_flowmeter.index + pd.Timedelta(hours=8, minutes=54, seconds=34)
            df_flowmeter = df_flowmeter[['STALL1']]
            df_flowmeter.columns = ['flow']
        if flowmeter == 2:
            # read flowmeter data with sample rate of /2s
            df_flowmeter = read_new_flowmeter(os.path.join(path_date, 'flowmeter.xlsx'), date, is_interpolate=False) 
            df_flowmeter = df_flowmeter[['flow']]
            
        # For fm, any number lower than 0.5 treated as 0
        df_flowmeter.loc[:, 'flow'] = df_flowmeter.loc[:, 'flow'].clip(lower=0.5)
        df_flowmeter.loc[:, 'flow'] = df_flowmeter.loc[:, 'flow'].replace({0.5:np.nan})
        # Resample the fm data to /1s
        df_flowmeter = df_flowmeter.resample('1S').pad()
        df_flowmeter.loc[:, 'flow'] = df_flowmeter.loc[:, 'flow'].replace({np.nan:0})
        # Change the unit from /1m to /1s
        df_flowmeter.loc[:, 'flow'] = df_flowmeter.loc[:, 'flow']/60
        df = pd.concat([df, df_flowmeter], axis=1)

    df = df.reset_index()
    # Derivatives were calculated based on five-second diff
    df['feces_deriv'] = df['feces'].diff(periods=5)/(df['date_time'].diff(periods=5).dt.total_seconds())
    df['urine_deriv'] = df['urine'].diff(periods=5)/(df['date_time'].diff(periods=5).dt.total_seconds())
    df['feces_deriv_2'] = df['feces_deriv'].diff(periods=5)/(df['date_time'].diff(periods=5).dt.total_seconds())
    df['urine_deriv_2'] = df['urine_deriv'].diff(periods=5)/(df['date_time'].diff(periods=5).dt.total_seconds())
    start_indexes, end_indexes = detect_event(df.loc[:], path_date, th_feces_deriv=0.0, 
                                              th_urine_deriv=0.0, th_end_last=5, th_duration=60)

    return df, start_indexes, end_indexes
    

def process_event(df, start_index, end_index):
    # Get single event as a dataframe w index from 0
    df_event = df.loc[start_index:end_index].reset_index()
    first_valid_index_feces = df_event.loc[:, 'feces'].first_valid_index() # get the first valid index of feces data
    first_valid_index_urine = df_event.loc[:, 'urine'].first_valid_index()
    df_event.loc[:, 'feces'] = df_event.loc[:, 'feces'] - df_event.loc[first_valid_index_feces, 'feces']
    df_event.loc[:, 'urine'] = df_event.loc[:, 'urine'] - df_event.loc[first_valid_index_urine, 'urine']
    return df_event

def plot_event(df, s_event, e_event, flowmeter = None, derivative=None):
    s_vis = max(0, s_event - 40)
    e_vis = min(df.index[-1], e_event + 40)
    df_event = process_event(df, s_vis, e_vis)
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(20, 10))    
    if derivative==1:
        ax.plot(df_event.index, df_event.loc[:, 'feces_deriv'], color="red", marker="^", label = "feces tank")
        y_label_f = "Feces Tank 1st Derivative"
    elif derivative==2:
        ax.plot(df_event.index, df_event.loc[:, 'feces_deriv_2'], color="red", marker="^", label = "feces tank")
        y_label_f = "Feces Tank 2nd Derivative"
    else:
        ax.plot(df_event.index, df_event.loc[:, 'feces'], color="red", marker="^", label = "feces tank")
        y_label_f = "Feces Tank Volume"
    plt.axvline(x = s_event - s_vis, ls='-', linewidth=2, color = 'red', label='event starts')
    plt.axvline(x = e_event - s_vis, ls=':', linewidth=2, color = 'red', label='event ends')
    ax.set_ylabel(y_label_f, color="red", fontsize=30) # set y-axis label
    ax.set_xlabel('Seconds',fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(loc='center left', fontsize = 20)
    
    ax2 = ax.twinx()
    if derivative==1:
        ax2.plot(df_event.index, df_event.loc[:, 'urine_deriv'], color="blue", marker="^", label = "urine tank")
        y_label_u = "Urine Tank 1st Derivative"
    elif derivative==2:
        ax2.plot(df_event.index, df_event.loc[:, 'urine_deriv_2'], color="blue", marker="^", label = "urine tank")
        y_label_u = "Urine Tank 2nd Derivative"
    else:
        ax2.plot(df_event.index, df_event.loc[:, 'urine'], color="blue", marker="^", label = "urine tank")
        y_label_u = "Urine Tank Volume"
    ax2.set_ylabel(y_label_u, color="blue",fontsize=30)
    if flowmeter:
        ax2.scatter(df_event.index, df_event.loc[:, 'flow'], color="blue", marker="o", label = "flowmeter")
    ax2.tick_params(axis='y', labelsize=20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.legend(loc='center right', fontsize = 20)
    return fig

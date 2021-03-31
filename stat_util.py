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

def get_stat(args, df, start_indexes, end_indexes):
    df_start = df.iloc[start_indexes].reset_index()
    df_end = df.iloc[end_indexes].reset_index()
    df_stat = pd.DataFrame() 
    df_stat['duration'] = df_end.loc[:, 'date_time'] - df_start.loc[:, 'date_time']
    df_stat['ft_change'] = df_end.loc[:, 'feces'] - df_start.loc[:, 'feces']
    df_stat['ut_change'] = df_end.loc[:, 'urine'] - df_start.loc[:, 'urine']
    df_stat['tank_change'] = df_stat['ft_change'] + df_stat['ut_change']
    for i, (start_index, end_index) in enumerate(zip(start_indexes, end_indexes)):
        df_event = process_event(df, start_index, end_index)
        df_stat.loc[i, 'event_num'] = i + 1
        df_stat.loc[i, 'duration(s)'] = end_index - start_index
        if args.flowmeter:
            df_stat.loc[i, 'flowmeter'] = df_event.loc[:, 'flow'].sum()*args.flowmeter
        df_stat.loc[i, 'num_max_feces'] = len(df_event.loc[(df_event['max_feces']>0.001)])
        df_stat.loc[i, 'num_max_urine'] = len(df_event.loc[(df_event['max_urine']>0.001)])
    if args.flowmeter:
        df_stat['flowmeter-tank'] = df_stat['flowmeter'] - df_stat['tank_change']
        column_names = ['event_num', 'duration', 'duration(s)', 'ft_change', 'ut_change', 
                        'tank_change', 'flowmeter', 'flowmeter-tank', 'num_max_feces', 'num_max_urine']
    else:
        column_names = ['event_num', 'duration', 'duration(s)', 'ft_change', 'ut_change', 
                        'tank_change', 'num_max_feces', 'num_max_urine']
    df_stat = df_stat[column_names]
    return df_stat
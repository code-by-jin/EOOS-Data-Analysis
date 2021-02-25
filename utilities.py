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

def read_data(path, date, is_interpolate = True):
    if os.path.exists(path):
        df = pd.ExcelFile(path) # read feces file
        sheet_name = df.sheet_names[0] # use the first sheet in the .xlsx file
        df = df.parse(sheet_name, skiprows=0, parse_dates=[['date', 'time']]) 
    else:
        path = path.replace('xlsx', 'csv')
        df = pd.read_csv(path, parse_dates=[['date', 'time']])
    df.columns = df.columns.str.replace(' ', '')
    df = df.groupby('date_time', as_index=True).mean() # Combine reduplicative data
    df = df.interpolate(method='pad', limit_direction='forward', axis=0)
    if is_interpolate:
        df = df.resample('1S').pad()
        df = df.rolling(window=10).mean()
    df.index = df.index.map(lambda t: t.replace(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:])))
    return df

def plot_event(df, start_all = [], end_all=[]):
    # create figure and axis objects with subplots()
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(20, 10))    
    ax2 = ax.twinx()
    # make a plot
    ax.plot(df.index, df.loc[:, 'feces'], color="red", marker="^", label = "feces tank")
    ax.plot(df.index, df.loc[:, 'feces']+df.loc[:, 'urine'], color="black", marker="*", label = "vft+vut")
    ax.set_ylabel("Feces Tank Volume (L)",color="red", fontsize=30) # set y-axis label
    ax.axhline(y=df.loc[:, 'flow'].sum()*5, color="black", label = "volume out")
    # make a plot with different y-axis using second axis object
    ax2.plot(df.index, df.loc[:, 'urine'], color="blue", marker="v", label = "urine tank")
    ax2.scatter(df.index, df.loc[:, 'flow'], color="blue", marker="o", label = "flowmeter")
    ax.set_xlabel('Seconds',fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(loc='center left', fontsize = 20)
    ax2.set_ylabel("Urine Tank Volume (L)",color="blue",fontsize=30)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    for i, (s, e) in enumerate(zip(start_all, end_all)):
        x_s, x_e =s, e
        plt.axvline(x=x_s, ls='--', linewidth=1, color = 'purple', label=str(df.loc[x_s, 'date_time']))
        plt.axvline(x=x_e, ls=':',  linewidth=1, color = 'black',  label=str(df.loc[x_e, 'date_time']))
        if i == 0:
            ax2.legend(loc='center right', fontsize = 20)
    return fig

def plot_derivative(df, start_all = [], end_all=[]):
    # create figure and axis objects with subplots()
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(20, 10))    
    ax2 = ax.twinx()
    # make a plot
    ax.plot(df.index, df.loc[:, 'feces_derivative'], color="red", marker="^", label = "feces tank")
    ax.scatter(df.index, df.loc[:, 'max_feces'], c='red', s=200, marker="s")
    ax.scatter(df.index, df.loc[:, 'min_feces'], c='red', s=200, marker="o")
    ax.set_ylabel("Feces Tank Volume (L)",color="red", fontsize=30) # set y-axis label
    # make a plot with different y-axis using second axis object
    ax2.plot(df.index, df.loc[:, 'urine_derivative'], color="blue", marker="v", label = "urine tank")
    ax2.scatter(df.index, df.loc[:, 'max_urine'], c='blue', s=200, marker="s")
    ax2.scatter(df.index, df.loc[:, 'min_urine'], c='blue', s=200, marker="o")
    ax.set_xlabel('Seconds',fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(loc='center left', fontsize = 20)
    ax2.set_ylabel("Urine Tank Volume (L)",color="blue",fontsize=30)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    for i, (s, e) in enumerate(zip(start_all, end_all)):
        x_s, x_e =s, e
        plt.axvline(x=x_s, ls='--', linewidth=1, color = 'purple', label=str(df.loc[x_s, 'date_time']))
        plt.axvline(x=x_e, ls=':',  linewidth=1, color = 'black',  label=str(df.loc[x_e, 'date_time']))
        if i == 0:
            ax2.legend(loc='center right', fontsize = 20)
    return fig

def start_condition(df, curr_index):
    con_feces = df.loc[curr_index+1, 'feces'] - df.loc[curr_index, 'feces'] != 0 # change in FT != 0
    con_urine = df.loc[curr_index+1, 'urine'] - df.loc[curr_index, 'urine'] != 0 # change in UT != 0
    return con_feces or con_urine

def end_condition(df, curr_index, th_feces = 0.001, th_urine = 0.001, th_end_last=15):
    th_last_index = min(curr_index+th_end_last, df.index[-1])
    con_feces = (df.loc[curr_index+1:th_last_index, 'feces_derivative'] <= th_feces).all()
    con_urine = (df.loc[curr_index+1:th_last_index, 'urine_derivative'] <= th_urine).all()
    con_last_index = (curr_index == df.index[-1])
    return (con_feces and con_urine) or con_last_index

def detect_end(df, start_index, th_feces = 0.001, th_urine = 0.001, th_end_last=15):
    curr_index = start_index + 1
    while (not end_condition(df, curr_index, th_feces, th_urine, th_end_last)):
        curr_index += 1
    return curr_index
          
def is_event(df, start_index, end_index, th_feces_change = 0.5, th_urine_change = 0.05):
    con_feces = df.loc[end_index, 'feces'] - df.loc[start_index, 'feces'] > th_feces_change
    con_urine = df.loc[end_index, 'urine'] - df.loc[start_index, 'urine'] > th_urine_change
    return con_feces or con_urine

def combine_close_events(df, start_indexes, end_indexes, th_duration = 60):
    dists = [s - e for s, e in zip(start_indexes[1:], end_indexes[:-1])]
    for i in reversed(range(len(dists))): # need to be in reverse order so that it won't throw off the subsequent indexes.
        if dists[i] < 15 or ((dists[i] < th_duration) and (df.loc[end_indexes[i]+1:end_indexes[i+1], 'flow'].count()<2)):
            start_indexes.pop(i+1)
            end_indexes.pop(i)
    return start_indexes, end_indexes
        
def adjust_start_index(df, start_indexes, end_indexes, th_flowmeter_before=300):
    last_end_index = 0
    for i in range(len(start_indexes)):
        curr_start_index = start_indexes[i]
        first_flowmeter_index = max(last_end_index, curr_start_index - th_flowmeter_before)
        first_valid_flowmeter_index = df.loc[first_flowmeter_index:curr_start_index, 'flow'].first_valid_index()
        if first_valid_flowmeter_index:
            start_indexes[i] = first_valid_flowmeter_index
        last_end_index = end_indexes[i]
    return start_indexes
   

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
    index_max_feces, index_min_feces = get_local_max(df_event, 'feces_derivative', 0.005)
    df_event['max_feces'] = df_event.iloc[index_max_feces]['feces_derivative']
    df_event['min_feces'] = df_event.iloc[index_min_feces]['feces_derivative']

    index_max_urine, index_min_urine = get_local_max(df_event, 'urine_derivative', 0.001)
    df_event['max_urine'] = df_event.iloc[index_max_urine]['urine_derivative'] 
    df_event['min_urine'] = df_event.iloc[index_min_urine]['urine_derivative']
     
    return df_event, index_max_feces, index_min_feces, index_max_urine, index_min_urine
    
# def count_fw(df_event):
#     count = 0
#     fw = df_event.loc[:, 'flow'].values
#     fw_out = []
#     for i in range(len(fw)):
#         if (not np.isnan(fw[i])) and np.all(np.isnan(fw[i+1:i+5])):
#             count+=1
#             fw_out.append(fw[i])
#     return count, fw_out
            
def get_stat(df, start_indexes, end_indexes):
    df_start = df.iloc[start_indexes].reset_index()
    df_end = df.iloc[end_indexes].reset_index()
    df_stat = pd.DataFrame() 
    df_stat['duration'] = df_end.loc[:, 'date_time'] - df_start.loc[:, 'date_time']
    df_stat['ft_change'] = df_end.loc[:, 'feces'] - df_start.loc[:, 'feces']
    df_stat['ut_change'] = df_end.loc[:, 'urine'] - df_start.loc[:, 'urine']
    df_stat['tank_change'] = df_stat['ft_change'] + df_stat['ut_change']
    fw_all = []
    for i, (start_index, end_index) in enumerate(zip(start_indexes, end_indexes)):
        df_event, index_max_feces, index_min_feces, index_max_urine, index_min_urine = process_event(df, start_index, end_index)
        if len(index_min_feces) == 0 or len(index_min_urine) == 0 or len(index_max_feces) == 0 or len(index_max_urine) == 0:
            fig = plot_event(df_event, [0], [df_event.index[-1]])
            plt.show()
            continue
        
        df_stat.loc[i, 'time_lag'] = index_min_feces[0] - index_min_urine[0]
 
        df_stat.loc[i, 'first_slope_ratio'] = df_event.loc[index_max_feces[0], 
                                                          'feces_derivative']/df_event.loc[index_max_urine[0], 
                                                                                            'urine_derivative']
        first_fm_index = df_event.loc[:, 'flow'].first_valid_index()
        df_stat.loc[i, 'first_fm_index'] = first_fm_index
        if first_fm_index is not None:
            df_stat.loc[i, 'first_fm_value'] = df_event.loc[first_fm_index, 'flow']
        else:
            df_stat.loc[i, 'first_fm_value'] = 0
        df_stat.loc[i, 'event_num'] = i + 1
        df_stat.loc[i, 'duration(s)'] = end_index - start_index
        df_stat.loc[i, 'flowmeter'] = df_event.loc[:, 'flow'].sum()*5
        df_stat.loc[i, 'num_max_feces'] = len(df_event.loc[(df_event['max_feces'] > 0.01)])
        df_stat.loc[i, 'num_max_urine'] = len(df_event.loc[(df_event['max_urine'] > 0.002)])
    df_stat['flowmeter-tank'] = df_stat['flowmeter'] - df_stat['tank_change']

    return df_stat

def detect_event(df, path_date, th_feces_derivative = 0.001, th_urine_derivative = 0.001, 
                 th_feces_change = 0.1, th_urine_change = 0.05, th_end_last = 15, th_duration = 60,
                 th_flowmeter_before=300):
    first_valid_index_feces = df.loc[:, 'feces'].first_valid_index() # get the first valid index of feces data
    first_valid_index_urine = df.loc[:, 'urine'].first_valid_index() # get the first valid index of urine data    
    df = df.loc[max(first_valid_index_feces, first_valid_index_urine):] # the first valid index is the larger one
    start_indexes, end_indexes = [], []
    curr_index = df.index.values[0]
    while curr_index < df.index[-1]:
        if start_condition(df, curr_index):
            start_index = curr_index
            end_index = detect_end(df, start_index, th_feces_derivative, th_urine_derivative, th_end_last)
            if is_event(df, start_index, end_index, th_feces_change, th_urine_change):
                start_indexes.append(start_index)
                end_indexes.append(end_index)
            curr_index = end_index + 1
        else:
            curr_index += 1  
    start_indexes, end_indexes = combine_close_events(df, start_indexes, end_indexes, th_duration)
    start_indexes = adjust_start_index(df, start_indexes, end_indexes, th_flowmeter_before)
    return start_indexes, end_indexes
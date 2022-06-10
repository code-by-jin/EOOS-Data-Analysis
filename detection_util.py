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

def is_event(df, start_index, end_index, th_feces_change = 0.5, th_urine_change = 0.01):
    # check if the event is noise
    con_feces = df.loc[end_index, 'feces'] - df.loc[start_index, 'feces'] > th_feces_change
    con_urine = df.loc[end_index, 'urine'] - df.loc[start_index, 'urine'] > th_urine_change    
    return con_feces or con_urine

def combine_close_events(df, start_indexes, end_indexes, th_duration = 60):
    # two events will be combined into one if they are close enough
    dists = [s - e for s, e in zip(start_indexes[1:], end_indexes[:-1])]
    for i in reversed(range(len(dists))): # need to be in reverse order so that it won't throw off the subsequent indexes.
        if dists[i] < th_duration: 
            start_indexes.pop(i+1)
            end_indexes.pop(i)
    return start_indexes, end_indexes

def start_condition(df, curr_index, th_feces_change = 0.05, th_urine_change = 0.05):
    con_feces = True
    con_urine = True
    tmp_index = curr_index
    for i in range(10):
        if tmp_index >= df.index[-1]:
            break
        con_feces = con_feces and (df.loc[tmp_index+1, 'feces'] - df.loc[tmp_index, 'feces'] > 0) # change in FT != 0
        con_urine = con_urine and (df.loc[tmp_index+1, 'urine'] - df.loc[tmp_index, 'urine'] > 0) # change in UT != 0
        tmp_index += 1
    return con_feces or con_urine

def end_condition(df, curr_index, th_feces_deriv=0, th_urine_deriv=0, th_end_last=15):
    th_last_index = min(curr_index+th_end_last, df.index[-1])
    con_feces = (df.loc[curr_index+1:th_last_index, 'feces_deriv_2'] <= th_feces_deriv).all()
    con_urine = (df.loc[curr_index+1:th_last_index, 'urine_deriv_2'] <= th_urine_deriv).all()
    con_last_index = (curr_index == df.index[-1])
    return (con_feces and con_urine) or con_last_index

def detect_end(df, start_index, th_feces_deriv=0.001, th_urine_deriv=0.001, th_end_last=15):
    curr_index = start_index + 1
    while (not end_condition(df, curr_index, th_feces_deriv, th_urine_deriv, th_end_last)):
        curr_index += 1
    return curr_index

def adjust_start_index(df, start_indexes, end_indexes, th_flowmeter_before=300):
    last_end_index = 0
    for i in range(len(start_indexes)):
        curr_start_index = start_indexes[i]
        first_flowmeter_index = max(last_end_index, curr_start_index - th_flowmeter_before)
        first_valid_flowmeter_index = curr_start_index
        for j in range(curr_start_index, max(0, curr_start_index-th_flowmeter_before), -1):
            if np.sum(df.loc[j-2:j, 'flow'].values) > 0:
                first_valid_flowmeter_index = j
            else:
                break
        if first_valid_flowmeter_index:
            start_indexes[i] = first_valid_flowmeter_index
        last_end_index = end_indexes[i]
    return start_indexes
   
def detect_event(df, path_date, th_feces_deriv=0, th_urine_deriv=0, 
                 th_feces_change=0.05, th_urine_change=0.05, th_end_last=30, th_duration=60,
                 th_flowmeter_before=60):
    first_valid_index_feces = df.loc[:, 'feces'].first_valid_index() # get the first valid index of feces data
    first_valid_index_urine = df.loc[:, 'urine'].first_valid_index() # get the first valid index of urine data    
    df = df.loc[max(first_valid_index_feces, first_valid_index_urine):] # the first valid index is the larger one
    start_indexes, end_indexes = [], []
    curr_index = df.index.values[0]
    while curr_index < df.index[-1]:
        if start_condition(df, curr_index):
            start_index = curr_index
            end_index = detect_end(df, start_index, th_feces_deriv, th_urine_deriv, th_end_last)
            if is_event(df, start_index, end_index, th_feces_change, th_urine_change):
                start_indexes.append(start_index)
                end_indexes.append(end_index)
            curr_index = end_index + 1
        else:
            curr_index += 1  
    start_indexes, end_indexes = combine_close_events(df, start_indexes, end_indexes, th_duration)
    return start_indexes, end_indexes

def read_data(path, date, is_interpolate = True):
    if os.path.exists(path):
        df = pd.ExcelFile(path) # read feces file
        sheet_name = df.sheet_names[0] # use the first sheet in the .xlsx file
        df = df.parse(sheet_name, skiprows=0, parse_dates=[['date', 'time']]) 
    else:
        path = path.replace('xlsx', 'csv')
        df = pd.read_csv(path, parse_dates=[['date', 'time']])
    df.columns = df.columns.str.replace(' ', '') # clean: remove spaces in column names
    df = df.groupby('date_time', as_index=True).mean() # combine reduplicative data
    df = df.interpolate(method='pad', limit_direction='forward', axis=0) # fill in NaNs using existing values.
    if is_interpolate:
        df = df.resample('1S').pad()
        df = df.rolling(window=10).mean()
    df.index = df.index.map(lambda t: t.replace(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:8])))
    return df

def read_new_flowmeter(path, date, is_interpolate = True):
    if os.path.exists(path):
        df = pd.ExcelFile(path) # read feces file
        sheet_name = df.sheet_names[0] # use the first sheet in the .xlsx file
        df = df.parse(sheet_name, skiprows=1, parse_dates=['Date Time, GMT+05:30']) 
    else:
        path = path.replace('xlsx', 'csv')
        df = pd.read_csv(path, skiprows=1, parse_dates=['Date Time, GMT+05:30'])
    df = df.rename(columns={'Date Time, GMT+05:30': 'date_time', 'FM, LPM (LGR S/N: 20965846)': 'flow'})
    df.columns = df.columns.str.replace(' ', '') # clean: remove spaces in column names
    df = df.groupby('date_time', as_index=True).mean() # combine reduplicative data
    df = df.interpolate(method='pad', limit_direction='forward', axis=0) # fill in NaNs using existing values.
    if is_interpolate:
        df = df.resample('1S').pad()
        df = df.rolling(window=10).mean()
    df.index = df.index.map(lambda t: t.replace(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:])))
    return df
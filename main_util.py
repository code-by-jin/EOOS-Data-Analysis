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

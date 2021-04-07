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
from plot_util import *
from stat_util import *

def analyze_one_day(args):
    path_date = os.path.join('data', args.date) # path to data of the date
    # feces tank
    df_feces = read_data(os.path.join(path_date, 'feces.xlsx'), args.date) 
    df_feces = df_feces[['weight']]
    df_feces.columns = ['feces']
    
    # urine tank
    df_urine = read_data(os.path.join(path_date, 'urine.xlsx'), args.date) 
    df_urine = df_urine[['weight']]
    df_urine.columns = ['urine']
    data_srcs = [df_feces, df_urine]
    
    # flowmeter data
    if args.flowmeter == 5:
        df_flowmeter = read_data(os.path.join(path_date, 'flowmeter.xlsx'), args.date, is_interpolate=False) 
        df_flowmeter.index = df_flowmeter.index + pd.Timedelta(hours=8, minutes=54, seconds=34)
        df_flowmeter = df_flowmeter[['STALL1']]
        df_flowmeter.columns = ['flow']
        df_flowmeter.loc[:, 'flow'] = df_flowmeter.loc[:, 'flow']/60
        data_srcs.append(df_flowmeter)
    if args.flowmeter == 2:
        df_flowmeter = read_new_flowmeter(os.path.join(path_date, 'flowmeter.xlsx'), args.date, is_interpolate=False) 
        df_flowmeter = df_flowmeter[['flow']]
        df_flowmeter.loc[:, 'flow'] = df_flowmeter.loc[:, 'flow'].clip(lower=0.5)
        df_flowmeter.loc[:, 'flow'] = df_flowmeter.loc[:, 'flow'].replace({0.5:np.nan})
        df_flowmeter.loc[:, 'flow'] = df_flowmeter.loc[:, 'flow']/60
        print(df_flowmeter.head())
        data_srcs.append(df_flowmeter)
        
    df = pd.concat(data_srcs, axis=1).reset_index()
    df['feces_deriv'] = df['feces'].diff(periods=10)/(df['date_time'].diff(periods=10).dt.total_seconds())
    df['urine_deriv'] = df['urine'].diff(periods=10)/(df['date_time'].diff(periods=10).dt.total_seconds())
    start_indexes, end_indexes = detect_event(df.loc[:], path_date)
    df_stat = get_stat(args, df, start_indexes, end_indexes)
    df_stat.to_csv(os.path.join(path_date, 'stat.csv'), index=False)
    for file in os.listdir(path_date):
        if file.endswith('.png'):
            os.remove(os.path.join(path_date, file)) 
    for i, (start_index, end_index) in enumerate(zip(start_indexes, end_indexes)):  
        fig = plot_event(args, df, start_index, end_index)
        fig.savefig(os.path.join(path_date, 'event_' + str(i+1) + '.png'))
        fig = plot_deriv(df, start_index, end_index)
        fig.savefig(os.path.join(path_date, 'deriv_' + str(i+1) + '.png'))  

def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='EOOS Data Analysis')
    parser.add_argument('--date', default='20201224', type=str, help='date to analyze')
    parser.add_argument('--flowmeter', default=None, type=int, help='sample rate for the flowmeter')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    analyze_one_day(args)
    
if __name__ == '__main__':
    main()

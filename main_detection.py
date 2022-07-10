import os
import numpy as np
import pandas as pd
from detection_util import detect_event
from dataset_util import read_data


def analyze_one_day(data_dir, date):
    path_date = os.path.join(data_dir, date) # path to data of the date
    # read feces tank data
    
    start_indexes, end_indexes = detect_event(df.loc[:])
    return df, start_indexes, end_indexes


def analyze_period(data_dir):
    '''
    Performs eoos analysis based on the period
            when the data was collected.
        Args:
            data_dir: Path to the data.
            fm: if include flowmeter data for analysis
    '''
    # Dataframe used to log the analysis
    cols = ['date', 'plot_num', 'start index', 'end index',
            'start time', 'end time', 'feces volume', 'urine volume']
    df_stat = pd.DataFrame(columns=cols)
    writer = pd.ExcelWriter(data_dir+'.xlsx', engine='xlsxwriter')

    # analyze date by date
    for date in os.listdir(data_dir):
        if date.startswith('.'): continue # skip sys hidden files
        print('Processing: ', date)
        # detect events for one date
        df, s_idxs, e_idxs = analyze_one_day(data_dir, date)
        
        # create figures and stats for each event
        plot_num = 0
        for s, e in zip(s_idxs, e_idxs):

            row = {'date': date,  'plot_num': plot_num, 
                   'start index': s, 'end index': e,
                   'start time': df.loc[s, 'date_time'] , 
                   'end time': df.loc[e, 'date_time'],
                   'feces volume': df.loc[e, 'feces'] - df.loc[s, 'feces'],
                   'urine volume': df.loc[e, 'urine'] - df.loc[s, 'urine'],
                   'duration': e - s}
            if row['feces volume'] + row['urine volume'] < 0.2: continue
            if row['duration'] < 15 or row['duration'] > 450: continue
            df_stat = df_stat.append(row, ignore_index=True, sort=False)
            plot_num += 1
        print(plot_num)
    
    # Save the stats
    df_stat.to_excel(writer, sheet_name=date)
    writer.save()

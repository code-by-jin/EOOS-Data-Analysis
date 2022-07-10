import os
from typing import Dict
import pandas as pd
from detection_util import detect_event
from dataset_util import read_data


def is_outlier(row: Dict):
    if row['total volume'] < 0.2:
        return True
    if row['duration'] < 15 or row['duration'] > 450:
        return True
    return False


def analyze_date(
    data_dir: str,
    date: str,
    th_feces_change: float = 0.05,
    th_urine_change: float = 0.05,
    th_start_last: int = 10,
    th_end_last: int = 5,
    th_duration: int = 60,
):
    cols = ['date', 'event id', 'start index', 'end index',
            'start time', 'end time', 'feces volume', 'urine volume',
            'total volume', 'duration']
    df_stat_date = pd.DataFrame(columns=cols)
    df = read_data(data_dir, date)
    start_idxes, end_idxes = detect_event(
        df,
        th_feces_change,
        th_urine_change,
        th_start_last,
        th_end_last,
        th_duration,
        )
    # create figures and stats for each event
    count_event = 0
    for s, e in zip(start_idxes, end_idxes):
        row = {'date': date,  'event id': count_event,
               'start index': s, 'end index': e,
               'start time': df.loc[s, 'date_time'],
               'end time': df.loc[e, 'date_time'],
               'feces volume': df.loc[e, 'feces'] - df.loc[s, 'feces'],
               'urine volume': df.loc[e, 'urine'] - df.loc[s, 'urine'],
               'duration': e - s}
        row['total volume'] = row['feces volume'] + row['urine volume']
        if is_outlier(row):
            continue
        df_stat_date = df_stat_date.append(row, ignore_index=True, sort=False)
        count_event += 1
    print(date + ": " + str(count_event) + " events")
    return df_stat_date


def analyze_period(
    data_dir: str,
    th_feces_change: float = 0.05,
    th_urine_change: float = 0.05,
    th_start_last: int = 10,
    th_end_last: int = 5,
    th_duration: int = 60,
):
    '''
    Performs eoos analysis based on the period
            when the data was collected.
        Args:
            data_dir: Path to the data.
            fm: if include flowmeter data for analysis
    '''
    stats = []
    # analyze date by date
    for date in os.listdir(data_dir):
        # skip sys hidden files
        if date.startswith('.'):
            continue
        print('Processing: ', date)
        df_stat_date = analyze_date(
            data_dir,
            date,
            th_feces_change,
            th_urine_change,
            th_start_last,
            th_end_last,
            th_duration,
            )
        stats.append(df_stat_date)
    df_stat = pd.concat(stats, ignore_index=True)

    # Save the stats
    writer = pd.ExcelWriter(data_dir+'.xlsx', engine='xlsxwriter')
    df_stat.to_excel(writer, sheet_name=date)
    writer.save()


def main():
    analyze_period('data/period_1')
    analyze_period('data/period_2')
    analyze_period('data/period_3')


if __name__ == '__main__':
    main()

import os
import pandas as pd


def read_weight_scale(path: str, date: str):
    # first read xlsx file, if not exist, try csv file
    if os.path.exists(path):
        df = pd.ExcelFile(path)
        sheet_name = df.sheet_names[0]
        df = df.parse(sheet_name, skiprows=0, parse_dates=[['date', 'time']])
    else:
        path = path.replace('xlsx', 'csv')
        df = pd.read_csv(path, parse_dates=[['date', 'time']])
    # clean: remove spaces in column names
    df.columns = df.columns.str.replace(' ', '')
    # combine reduplicative data
    df = df.groupby('date_time', as_index=True).mean()
    # fill in NaNs using existing values.
    df = df.interpolate(method='pad', limit_direction='forward', axis=0)
    # 10s mean filter
    df = df.resample('1S').pad()
    df = df.rolling(window=10).mean()
    # formatting index as year-month-day
    df.index = df.index.map(lambda t: t.replace(
                    year=int(date[:4]),
                    month=int(date[4:6]),
                    day=int(date[6:8])))
    return df


def read_data(data_dir: str, date: str, periods: int = 5):
    path_date = os.path.join(data_dir, date)

    df_feces = read_weight_scale(os.path.join(path_date, 'feces.xlsx'), date)
    df_feces = df_feces[['weight']]
    df_feces.columns = ['feces']

    # read urine tank data
    df_urine = read_weight_scale(os.path.join(path_date, 'urine.xlsx'), date)
    df_urine = df_urine[['weight']]
    df_urine.columns = ['urine']

    # combine urine and feces data into one dataframe
    df = pd.concat([df_feces, df_urine], axis=1)
    df = df.dropna()

    df = df.reset_index()

    # Derivatives were calculated based on five-second diff
    time_diff = df['date_time'].diff(periods=periods).dt.total_seconds()

    df['feces_deriv'] = df['feces'].diff(periods=periods)/time_diff
    df['urine_deriv'] = df['urine'].diff(periods=periods)/time_diff
    df['feces_deriv_2'] = df['feces_deriv'].diff(periods=periods)/time_diff
    df['urine_deriv_2'] = df['urine_deriv'].diff(periods=periods)/time_diff

    return df

# import necessary packages
import pandas as pd


def is_event(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    th_feces_change: float = 0.05,
    th_urine_change: float = 0.05,
):
    '''
    check if the event is noise
    '''
    feces_change = df.loc[end_idx, 'feces'] - df.loc[start_idx, 'feces']
    urine_change = df.loc[end_idx, 'urine'] - df.loc[start_idx, 'urine']
    return feces_change > th_feces_change or urine_change > th_urine_change


def combine_close_events(start_idxes, end_idxes, th_duration=60):
    '''
    two events will be combined into one if they are close enough
    '''
    dists = [s - e for s, e in zip(start_idxes[1:], end_idxes[:-1])]
    # need to be in reverse order, so that
    # it won't throw off the subsequent indices.
    for i in reversed(range(len(dists))):
        if dists[i] < th_duration:
            start_idxes.pop(i+1)
            end_idxes.pop(i)
    return start_idxes, end_idxes


def start_condition(df: pd.DataFrame, curr_idx: int, th_last: int = 10):
    con_feces = True
    con_urine = True

    for idx in range(curr_idx, curr_idx + th_last):
        if idx >= df.index[-1]:
            break
        # change in FT > 0
        con_feces &= (df.loc[idx+1, 'feces'] - df.loc[idx, 'feces'] > 0)
        # change in UT > 0
        con_urine &= (df.loc[idx+1, 'urine'] - df.loc[idx, 'urine'] > 0)
    return con_feces or con_urine


def end_condition(df: pd.DataFrame, curr_idx: int, th_last: int = 5):
    # if it's the last index of the dataset, return True
    last_idx = df.index[-1]
    if curr_idx == last_idx:
        return True
    # last index to check
    last_idx = min(curr_idx+th_last, last_idx)
    con_feces = (df.loc[curr_idx+1:last_idx, 'feces_deriv_2'] <= 0).all()
    con_urine = (df.loc[curr_idx+1:last_idx, 'urine_deriv_2'] <= 0).all()
    return con_feces and con_urine


def detect_end(df: pd.DataFrame, start_idx: int, th_last: int = 5):
    curr_idx = start_idx + 1
    while (not end_condition(df=df, curr_idx=curr_idx, th_last=th_last)):
        curr_idx += 1
    return curr_idx


def get_valid_dataset(df: pd.DataFrame):
    '''
    detecting won't start until two scales are on
    '''
    # get the first valid index of feces data
    first_valid_index_feces = df.loc[:, 'feces'].first_valid_index()
    # get the first valid index of urine data
    first_valid_index_urine = df.loc[:, 'urine'].first_valid_index()
    # the first valid index is the larger one
    return df.loc[max(first_valid_index_feces, first_valid_index_urine):]


def detect_event(
    df: pd.DataFrame,
    th_feces_change: float = 0.05,
    th_urine_change: float = 0.05,
    th_start_last: int = 10,
    th_end_last: int = 5,
    th_duration: int = 60,
):
    df = get_valid_dataset(df)
    start_idxes, end_idxes = [], []
    curr_idx = df.index.values[0]
    while curr_idx < df.index[-1]:
        # start point detection
        if start_condition(df=df, curr_idx=curr_idx, th_last=th_start_last):
            start_idx = curr_idx
            # end point detection
            end_idx = detect_end(
                df,
                start_idx,
                th_end_last,
                )
            # event filtering
            flag_event = is_event(
                df,
                start_idx,
                end_idx,
                th_feces_change,
                th_urine_change,
                )
            if flag_event:
                start_idxes.append(start_idx)
                end_idxes.append(end_idx)
            curr_idx = end_idx + 1
        else:
            curr_idx += 1
    # event combination
    start_idxes, end_idxes = combine_close_events(
        start_idxes,
        end_idxes,
        th_duration,
        )
    return start_idxes, end_idxes    

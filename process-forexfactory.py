import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sys

SPECIAL_CASE_TIME = ['All Day', 'Dec Data', 'Day 1', 'Day 2', 'Jan Data', 'Feb Data', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Tentative',
'Mar Data', 'Apr Data', 'May Data', 'Jun Data', 'Jul Data', 'Aug Data', 'Sep Data', 'Oct Data', 'Nov Data', '24th-30th', '29th-3rd']

def get_view(price, time, interval=30):
    return price[(price.Timestamp - time > pd.Timedelta(0)) & (price.Timestamp - time < pd.Timedelta(interval, unit='m'))]

def small_process(filename):
    """
    This function will pre-process a small part of dataset before merging into bigger on
    ver 2021-0925

    Task:
    - remove null value
    - remove other currency
    - add new columns
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    # origin_df = df.copy()

    # print('Originary shape: [{row}][{col}]'.format(row=df.shape[0], col=df.shape[1]))

    # copy date
    date = None
    time1 = None
    for idx in range(df.shape[0]):
        if not df.loc[idx, 'Date'] is pd.NaT:
            date = df.loc[idx, 'Date']
        else:
            df.loc[idx, 'Date'] = date

        if not df.loc[idx, 'Time'] == '':
            time1 = df.loc[idx,'Time']
        else:
            if not time1.isnull():
                df.loc[idx,'Time'] = time1

    for idx in range(df.shape[0]):

        if (type(df.loc[idx, 'Time']) == float) or (df.loc[idx, 'Time'] in SPECIAL_CASE_TIME):
            if (idx > 0):
                df.loc[idx, 'Time'] = df.loc[idx-1, 'Time']
            else:
                df.loc[idx, 'Time'] = df.loc[idx, 'Date']
        else:
            isAM = df.loc[idx, 'Time'][-2:] == 'am'
            v =  df.loc[idx, 'Time'][:-2].split(':')
            if len(v) > 1:
                hour, minute = v
            else:
                hour = v[0]
                minute = 0

            if isinstance(df.loc[idx, 'Date'], str):
                print(df.loc[idx, 'Date'])
                df.loc[idx, 'Date'] = pd.to_datetime(df.loc[idx, 'Date'])
            df.loc[idx, 'Time'] = df.loc[idx, 'Date'] + pd.to_timedelta(int(hour), unit='h') + pd.to_timedelta(int(minute), unit='m')
            df.loc[idx, 'Time'] += pd.Timedelta(12, unit='h') * int(not isAM)

    # only use USD and EUR
    df = df[(df.Currency == 'USD') | (df.Currency == 'EUR')]

    # remove Holiday
    df = df[df.Impact != "Holiday"]

    # drop null value
    df = df.dropna()
    df = df.drop(columns=['Date'])

    df = df.reset_index(drop=True)
    df.Time = df.Time + pd.Timedelta(11, unit='h')

    print('Preprocess -')
    return df

def addEffect(df, price):
    """
    price is Exness[['Timestamp', 'Bid']]
    df is the result of small_process()
    """
    for idx in range(df.shape[0]):
        try:
            # get view
            date = pd.to_datetime(df.loc[idx, 'Time'].date())
            masked = (price.Timestamp - date > pd.Timedelta(0)) & (price.Timestamp - date < pd.Timedelta(days=1))
            viewDay = price[masked]
            # get first index
            idx1 = viewDay.index.min()
            diff = round((viewDay.Bid.max() - viewDay.Bid.min()) * 10000)
            signed = 1 if (viewDay.Bid.max() + viewDay.Bid.min()) / 2 > viewDay.loc[idx1, 'Bid'] else -1

            df.loc[idx, 'EffectDay'] = diff * signed

        except Exception as e:
            print('Exception {e}, date {date}, view shape {shape}'.format(e=e, date=date, shape=viewDay.shape[0]))
            df.loc[idx, ['EffectDay', 'Effect60', 'Effect30']] = 0
            continue

        try:
            view60 = get_view(viewDay, df.loc[idx, 'Time'], 60)
            idx1 = view60.index.min()
            diff = round((view60.Bid.max() - view60.Bid.min()) * 10000)
            signed = 1 if (view60.Bid.max() + view60.Bid.min()) / 2 > view60.loc[idx1, 'Bid'] else -1

            df.loc[idx, 'Effect60'] = diff * signed
        except Exception as e:
            print('view60 shape: {}'.format(view60.shape[0]))
            df.loc[idx, ['Effect60', 'Effect30']] = 0
            continue

        try:
            view30 = get_view(view60, df.loc[idx, 'Time'], 30)
            idx1 = view30.index.min()
            diff = round((view30.Bid.max() - view30.Bid.min()) * 10000)
            signed = 1 if (view30.Bid.max() + view30.Bid.min()) / 2 > view30.loc[idx1, 'Bid'] else -1

            df.loc[idx, 'Effect30'] = diff * signed
        except Exception as e:
            print('view30 shape: {}'.format(view30.shape[0]))
            df.loc[idx, 'Effect30'] = 0
            continue

        del view30, view60, viewDay

    # print(df.columns)
    print('Add effect --')
    return df

def get_sunday(year):
    year = int(year)
    d = date(year, 1, 1)
    d += timedelta(days = 6 - d.weekday())
    while d.year == year and d - date.today() < timedelta(0):
        month = d.strftime("%b").lower()
        target_week = month + d.strftime("%d") + '.' + str(year)
        yield target_week
        d += timedelta(days=7)

def main(input):
    year = input[-8:-4]
    allSundays = get_sunday(year)

    exness = pd.read_csv(input, parse_dates=['Timestamp'])
    exness = exness[['Timestamp', 'Bid']]
    print(exness.shape)
    print('Read file --- Complete')

    exness.Timestamp = exness.Timestamp.dt.tz_convert('Asia/Ho_Chi_Minh')
    print('Convert to Asia/Ho_Chi_Minh --- Complete')

    # exness.Timestamp = exness.Timestamp.apply(lambda x: x.replace(tzinfo=None))
    exness.Timestamp = exness.Timestamp.dt.tz_localize(None)
    print('Remove timezone --- Complete')

    print(exness.Timestamp.dtype)
    # print('Sample of timestamp converted')
    # date = pd.to_datetime('2020-01-06 00:00:00')
    # print(exness[exness.Timestamp.dt.date == pd.to_datetime('2020-01-06 00:00:00')][:3])
    # print(exness[(exness.Timestamp - date > pd.Timedelta(0)) & (exness.Timestamp - date < pd.Timedelta(1, unit='d'))])
    print('Small sample: ')
    print(exness[:3])

    # exness.Timestamp += pd.to_timedelta(7, unit='h')

    result_df = pd.DataFrame(columns=['Time', 'Currency', 'Impact', 'Event', 'Actual', 'Forecast', 'Previous', 'EffectDay', 'Effect60', 'Effect30', 'Effect15'])
    print('Create result dataframe --- Complete')

    for sunday in allSundays:
        filename = 'crawl_output{}/{}_ff.csv'
        df = small_process(filename.format(year, sunday))
        # print(df.loc[:3, 'Time'])
        print(df.Time.dtype)
        df = addEffect(df, exness)
        print(df[:3])
        result_df = result_df.append(df)
        print('{} --- Complete'.format(sunday))

    print('Final shape: [{row}][{col}]'.format(row=result_df.shape[0], col=result_df.shape[1]))
    print(result_df.head(5))
    result_df.to_csv('{}_ff.csv'.format(year), index=False)
    print('Save file --- Complete')

    return

if __name__ == '__main__':
    start_time = datetime.now()

    # input is Exness file
    main(sys.argv[-1])
    print('Processing time: {}'.format(datetime.now() - start_time))

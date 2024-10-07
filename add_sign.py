import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def get_view(price, time, interval=30):
    return price[(price.Timestamp - time > pd.Timedelta(0)) & (price.Timestamp - time < pd.Timedelta(interval, unit='m'))]

def add_sign(filename):
    df = pd.read_csv(filename, parse_dates = ['Time'])

    year = filename[:4]
    if int(year) < 2021:
        M1 = pd.read_csv('input/{}_M1.csv'.format(year), parse_dates=['Timestamp'])
    else:
        M1 = pd.read_csv('input/2021_M1_1_9.csv', parse_dates=['Timestamp'])

    M1.Timestamp = M1.Timestamp + pd.Timedelta(7, unit='h')

    print('Load file --- Complete')
    print('ff shape {}, M1 shape {}'.format(df.shape[0], M1.shape[0]))

    for idx in range(df.shape[0]):
        try:
            date = pd.to_datetime(df.loc[idx, 'Time'].date())
            masked = (M1.Timestamp - date > pd.Timedelta(0)) & (M1.Timestamp - date < pd.Timedelta(days=1))
            viewDay = M1[masked].reset_index(drop=True)
            df.loc[idx, 'SignEffectDay'] = 1 if (viewDay.loc[viewDay.index.min(), 'Bid'] > viewDay.loc[viewDay.index.max(), 'Bid']) else 0
        except Exception as e:
            print('Exception {}, viewDay shape {}'.format(e, viewDay.shape[0]))
            df.loc[idx, ['SignEffectDay', 'SignEffect60', 'SignEffect30', 'SignEffect15']] = None
            continue

        try:
            view60 = get_view(viewDay, df.loc[idx, 'Time'], 60)
            df.loc[idx, 'SignEffect60'] = 1 if (view60.loc[view60.index.min(), 'Bid'] > view60.loc[view60.index.max(), 'Bid']) else 0
        except Exception as e:
            print('Exception {e}, view60 shape {view}'.format(e=e, view=view60.shape[0]))
            df.loc[idx, ['SignEffect60', 'SignEffect30', 'SignEffect15']] = None
            continue

        try:
            view30 = get_view(view60, df.loc[idx, 'Time'], 30)
            df.loc[idx, 'SignEffect30'] = 1 if (view30.loc[view30.index.min(), 'Bid'] > view30.loc[view30.index.max(), 'Bid']) else 0
        except Exception as e:
            print('Exception {e}, view30 shape {view}'.format(e=e, view=view30.shape[0]))
            df.loc[idx, ['SignEffect30', 'SignEffect15']] = None
            continue

        try:
            view15 = get_view(view30, df.loc[idx, 'Time'], 15)
            df.loc[idx, 'SignEffect15'] = 1 if (view15.loc[view15.index.min(), 'Bid'] > view15.loc[view15.index.max(), 'Bid']) else 0
        except Exception as e:
            print('Exception {e}, view15 shape {view}'.format(e=e, view=view15.shape[0]))
            df.loc[idx, 'SignEffect15'] = None
            continue

        del view15, view30, view60, viewDay

    print('Add sign --- Complete')
    df.to_csv('signed_{}'.format(filename), index=False)
    print('Save file --- Complete')
    return

if __name__ == '__main__':
    start_time = datetime.now()
    filename = sys.argv[1]
    add_sign(filename)
    print('Processing time: {}'.format(datetime.now() - start_time))

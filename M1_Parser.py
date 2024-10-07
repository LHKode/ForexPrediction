import pandas as pd
import numpy as np
import argparse

def read_csv (path):
  source_df = pd.read_csv(path)
  #drop unnecessary columns
  source_df = source_df[['Timestamp', 'Bid']]
  #convert date format to yyyy-mm-dd hh:mm
  source_df['Timestamp'] = [':'.join(s.split('.')[0].split(':')[:-1]) for s in source_df['Timestamp']]
  return source_df


def parse_value (df):
  timestamps = []
  bids = []

  for i in range(len(df['Timestamp']) - 1):
    current_minute = df['Timestamp'][i].split(':')[1]
    next_minute = df['Timestamp'][i+1].split(':')[1]
    print(f'\rProcessing {i} rows', end='')
    if (current_minute != next_minute):
      bids.append('{:.4f}'.format(round(df['Bid'][i],4)))
      timestamps.append(df['Timestamp'][i])
  print(f'\nResult got {len(bids)} rows !')  
  return timestamps, bids

def convert_to_df (time, bid):
  df = pd.DataFrame(
    {'Timestamp':time,
     'Bid': bid,
    })
  return df

def do_parser(path, save_path):
  source = read_csv(path)
  time, bid = parse_value(source)
  res_df = convert_to_df(time, bid)
  res_df.to_csv(save_path, index = False) 
  print('Saved file in the output path !')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()
        
    do_parser(args.input_path, args.output_path)





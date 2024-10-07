import pandas as pd
import numpy as np
import argparse
import datetime
import calendar
import plotly.graph_objects as go

def processing (x):
  return int(round((float(x)*10000),0))

def read_csv (path):
  source_df = pd.read_csv(path)
  bid = [processing(df_bid) for df_bid in source_df['Bid']]
  return source_df, bid

def find_trend (idx, bid, TREND):
  max_global = bid[idx]
  min_global = bid[idx]
  max_idx = idx
  min_idx = idx
  pips = max_global - min_global
  while (pips <= TREND and idx<len(bid)):
    idx+=1
    current = bid[idx]
    if (current < min_global):
      min_global = current
      min_idx = idx
    if (current > max_global):
      max_global = current
      max_idx = idx
    pips = max_global - min_global
  
  return idx, max_global, max_idx, min_global, min_idx

def trend_de (idx, min_global, min_idx, max_global, max_idx, bid):
  current = bid[idx]
  #find until get current bid > min global
  while (current <= min_global):
    idx += 1 
    current = bid[idx]
    #update min_global
    if current < min_global:
      min_global = current
      min_idx = idx
  #got reverse pip
  reverse = current - min_global
  #update max_global
  if current > max_global:
    max_global = current
    max_idx = idx
  
  return idx, max_global, max_idx, min_global, min_idx, reverse

def trend_in (idx, min_global, min_idx, max_global, max_idx, bid):
  current = bid[idx]
  while (current >= max_global):
    idx += 1 
    current = bid[idx]
    if current > max_global:
      max_global = current
      max_idx = idx
      # range
  reverse = max_global - current
  if current < min_global:
    min_global = current
    min_idx = idx
  
  return idx, max_global, max_idx, min_global, min_idx, reverse


def parse_value (df, bid, RANGE, REVERSE, TREND):
  start = []
  end = []
  reverse_start = []
  reverse_end = []
  max_range_idx = []
  min_range_idx = []

  index = 0
  while index < len(bid)-2:
    reverse = -1
    end_point = -1
    reverse_start_point = -1
    reverse_end_point = -1
    start_point = index
    index, max_global, max_idx, min_global, min_idx = find_trend(index, bid, TREND)
    try:
      while (reverse < REVERSE):
        if (max_idx < min_idx): # XH giam
          if index > len(bid)-1:
            break
          index, max_global, max_idx, min_global, min_idx, reverse = trend_de(index+1, min_global, min_idx, max_global, max_idx, bid)
          reverse_start_point = min_idx
          reverse_end_point = index
          if(reverse>= REVERSE):
            end_point = index
            break
        #--------------------------------------------------------------
        else: # XH tang
          if index > len(bid)-1:
            break
          index, max_global, max_idx, min_global, min_idx, reverse = trend_in(index+1, min_global, min_idx, max_global, max_idx, bid)
          reverse_start_point = max_idx
          reverse_end_point = index
          if(reverse>= REVERSE):
            end_point = index
            break
        if index >=len(bid):
          break
        else:
          index+=1
      range_term = max_global - min_global
      #--------------------------------------------------------------
      if range_term > RANGE:
        
        while (reverse < (range_term-RANGE)+REVERSE):
          if (max_idx < min_idx): # XH p
            index, max_global, max_idx, min_global, min_idx, reverse = trend_de(index+1, min_global, min_idx, max_global, max_idx, bid)
            reverse_start_point = min_idx
            reverse_end_point = index
            if(reverse>=(range_term-RANGE)+8):
              end_point = index
              break
          #--------------------------------------------------------------
          else: # XH tang
            index, max_global, max_idx, min_global, min_idx, reverse = trend_in(index+1, min_global, min_idx, max_global, max_idx, bid)
            reverse_start_point = max_idx
            reverse_end_point = index
            if(reverse>=(range_term-RANGE)+REVERSE):
              end_point = index
              break
          index+=1
        # print(reverse)
    except:
      pass
  # print(i, reverse)
    max_range_idx.append(max_idx)
    min_range_idx.append(min_idx)
    start.append(start_point)
    end.append(end_point)
    reverse_start.append(reverse_start_point)
    reverse_end.append(reverse_end_point)
  return max_range_idx, min_range_idx, start, end, reverse_start, reverse_end

def convert_to_df (df, bid, max_range_idx, min_range_idx, start, end, reverse_start, reverse_end):
  timestamp_start = []
  timestamp_end = []
  reverse = []
  reverse_pip = []
  RANGE = []
  reverse_date = []

  for i in range(len(start)-1):
    timestamp_start.append(df['Timestamp'][start[i]])
    timestamp_end.append(df['Timestamp'][end[i]])
    reverse.append(df['Timestamp'][reverse_start[i]])
    reverse_pip.append(abs(bid[reverse_start[i]]- bid[reverse_end[i]]))
    RANGE.append(abs(bid[max_range_idx[i]]- bid[min_range_idx[i]]))
    a= df['Timestamp'][reverse_start[i]].split(' ')[0].split('-')
    y = int(a[0])
    m = int(a[1])
    d = int(a[2])
    day = datetime.datetime(y,m,d)
    reverse_date.append(day.strftime("%a"))
    df1 = pd.DataFrame(
      {'Timestamp_start': timestamp_start,
      'Reverse_start': reverse,
      'Reverse_date': reverse_date,
      'Timestamp_end': timestamp_end,
      'Reverse_Range': reverse_pip,
      'Range': RANGE
      })
  return df1

def summary (df, Range, Reverse):
  # Summary 1
  small_40 = []
  great_40 = []
  reverse_start = []
  reverse_date = []
  reverse_range = []

  reverse = []
  b1= "True"
  b2 = "False"

  for i in range(len(df['Range'])):
    if (df['Range'][i] > Range):
      great_40.append(df['Range'][i])
      reverse_start.append(df['Reverse_start'][i])
      reverse_date.append(df['Reverse_date'][i])
      reverse_range.append(df['Reverse_Range'][i])
      reverse_new = (df['Range'][i] - Range) + Reverse
      if (df['Reverse_Range'][i] >= reverse_new):
        reverse.append(b1)
      else:
        reverse.append(b2)
    else:
      small_40.append(df['Range'][i])
  df1 = pd.DataFrame(
      {'Timestamp': reverse_start,
       'Date': reverse_date,
       'Reverse_Range': reverse_range,
       'Range': great_40,
       'Reverse': reverse
      })

  # Summary 2
  max_in_range = max(small_40)
  max_out_range = max(great_40)
  cases = len(great_40)
  true_val = df1['Reverse'].value_counts()[0]
  false_val = df1['Reverse'].value_counts()[1]
  df2 = pd.DataFrame(
      {'Max_in_Range': max_in_range,
       'Max_out_Range': max_out_range,
       'Cases': cases,
       'Successful_cases': true_val,
       'Failed_cases': false_val
      }, index = ["Total"])
  return df1, df2

def plot(df, df1, path):
  time_reverse = []
  bid_reverse = []
  for i in range(len(df1['Reverse_start'])):
    start = df.index[df['Timestamp'] == df1['Reverse_start'][i]].values[0]
    end = df.index[df['Timestamp'] == df1['Timestamp_end'][i]].values[0]
    for j in range(start,end+1):
      time_reverse.append(df['Timestamp'][j])
      bid_reverse.append(df['Bid'][j])
    time_reverse.append(None)
    bid_reverse.append(None)
  
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Bid'], name="trend",
                      line_shape='linear'))
  fig.add_trace(go.Scatter(x=time_reverse, y=bid_reverse, name="reverse",
                      line_shape='linear'))

  fig.update_traces(mode="lines", hovertemplate=None)
  fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))
  fig.update_layout(hovermode="x")
  fig.write_html(path)



def do_parser(path, Range, Reverse, Trend, save_path, plot_path, s1_path, s2_path):
  source, bid = read_csv(path)
  max_range_idx, min_range_idx, start, end, reverse_start, reverse_end = parse_value(source, bid, Range, Reverse, Trend)
  res_df = convert_to_df(source, bid, max_range_idx, min_range_idx, start, end, reverse_start, reverse_end)
  
  sum_1, sum_2 = summary(res_df, Range, Reverse)
  plot(source, res_df, plot_path)
  res_df.to_csv(save_path, index = False)
  sum_1.to_csv(s1_path, index = False)
  sum_2.to_csv(s2_path, index = False)
  print('Saved all files in output path !')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--Range", type=int)
    parser.add_argument("--Reverse", type=int)
    parser.add_argument("--Trend", type=int)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--Plot_path", type=str)
    parser.add_argument("--Summary1_path", type=str)
    parser.add_argument("--Summary2_path", type=str)
    args = parser.parse_args()
        
    do_parser(args.input_path, args.Range, args.Reverse, args.Trend, args.output_path, args.Plot_path, args.Summary1_path, args.Summary2_path)





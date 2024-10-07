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
  #convert GMT+0 to VN (GMT+7) 
  new_time = []
  for time in source_df['Timestamp']:
    russia_datetime = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M")
    hours = 7
    hours_added = datetime.timedelta(hours = hours)
    date_vn = russia_datetime + hours_added
    new_time.append(str(date_vn)[:-3]) 
  source_df['Timestamp'] = new_time
  
  #get bid array
  bid = [processing(df_bid) for df_bid in source_df['Bid']]
  return source_df, bid

def find_trend (idx, bid, trend_lower):
  max_global = bid[idx]
  min_global = bid[idx]
  max_idx = idx
  min_idx = idx
  pips = trend_lower - 1
  while ( pips < trend_lower and idx<len(bid)):
    idx+=1
    current = bid[idx]
    if (current < min_global):
      min_global = current
      min_idx = idx
    elif (current > max_global):
      max_global = current
      max_idx = idx
    pips = max_global - min_global

  return idx, max_global, max_idx, min_global, min_idx

def trend_de (idx, TP, min_global, bid):
  current = bid[idx]
  found = False
  reverse = current - min_global

  while (current >= min_global and reverse <TP):
    idx+=1
    current = bid[idx]
    reverse = current - min_global

  if (reverse >= TP):
    print('get')
    found = True

  return found, idx, reverse

def trend_in (idx, TP, max_global, bid):
  current = bid[idx]
  found = False
  reverse = max_global - current

  while (current <= max_global and reverse <TP):
    idx+=1
    current = bid[idx]
    reverse = max_global - current

  if (reverse >= TP):
    print('get')
    found = True

  return found, idx, reverse

def parse_value (bid, TP, TREND):
  start = []
  end = []
  reverse_start = []
  reverse_end = []
  max_range_idx = []
  min_range_idx = []
  index = 0
  while (index < len(bid)-2):
    #set all index value at current
    
    start_point = index
    max_idx = index
    min_idx = index
    reverse_start_point = index
    reverse_end_point = index
    #find until get value fit Trend-Tp
    found = False
    fail_time = 0
    trend_lower = TREND[0]
    try:
      while(found == False and fail_time<75):
        print('a',index)
        #find max, min which max-min > trend_lower
        index, max_global, max_idx, min_global, min_idx = find_trend(index, bid, trend_lower)
        print('b',index, max_global-min_global)
        tp = 0
        if (max_global-min_global<TREND[1]):
          tp = TP[0]
        elif (max_global-min_global<TREND[2]):
          tp = TP[1]
        elif (max_global-min_global<TREND[3]):
          tp = TP[2]
        elif (max_global-min_global<TREND[4]):
          tp = TP[3]
        elif (max_global-min_global<TREND[5]):
          tp = TP[4]
        elif (max_global-min_global<TREND[6]):
          tp = TP[5]
        elif (max_global-min_global<TREND[7]):
          tp = TP[6]
        elif (max_global-min_global<TREND[8]):
          tp = TP[7]
        elif (max_global-min_global<TREND[9]):
          tp = TP[8]
        elif (max_global-min_global<TREND[10]):
          tp = TP[9]
        elif (max_global-min_global<TREND[11]):
          tp = TP[10]
        elif (max_global-min_global<TREND[12]):
          tp = TP[11]
        elif (max_global-min_global<TREND[13]):
          tp = TP[12]
        elif (max_global-min_global == TREND[13]):
          tp = TP[13]
        else :
          tp = ((max_global-min_global)-TREND[13])+TP[13]

        if (max_idx > min_idx): # Trend Tăng
          found, index, reverse = trend_in(index+1, tp, max_global, bid)
          # reverse start = index of max
          reverse_start_point = max_idx
        else: # Trend Giảm
          found, index, reverse = trend_de(index+1, tp, min_global, bid)
          # reverse start = index of min
          reverse_start_point = min_idx
        # reverse end = last
        reverse_end_point = index
        # increase trend_lower and reset current to start point for new iteration in case of not found 
        if found == False:
          print('False')
          reverse_start_point = index
          trend_lower+=1
          fail_time+=1
          index = start_point
        else:
          break
    except:
      pass

    max_range_idx.append(max_idx)
    min_range_idx.append(min_idx)
    start.append(start_point)
    end.append(index)
    reverse_start.append(reverse_start_point)
    reverse_end.append(reverse_end_point)
    index+=1
  print('done')
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

def cluster_zero_tp (df):
  try:
    for i in range(len(df)):
      while (df['Reverse_Range'][i] == 0 and df['Reverse_Range'][i + 1] == 0):
        df['Reverse_start'][i] = "None"
        df['Timestamp_end'][i] = df['Timestamp_end'][i+1]
        df.drop([df.index[i+1]], axis=0, inplace = True) 
        df.reset_index(drop = True, inplace =True)
  except:
    pass
  df = df[['Timestamp_start', 'Reverse_start', 'Reverse_date',
       'Timestamp_end', 'Reverse_Range', 'Range']]
  return df

def summary (df, Range = 73, Reverse = 6):
  # Summary 1
  small_40 = []
  great_40 = []
  reverse_start = []
  reverse_date = []
  reverse_range = []

  reverse = []
  b1= "Success"
  b2 = "Fail"

  for i in range(len(df['Range'])):
    if (df['Range'][i] > Range):
      great_40.append(df['Range'][i])
      reverse_start.append(df['Timestamp_end'][i])
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
       'Status': reverse
      })

  # Summary 2
  max_in_range = max(small_40)
  max_out_range = max(great_40)
  cases = len(great_40)
  true_val = df1['Status'].value_counts()[0]
  false_val = df1['Status'].value_counts()[1]
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
    try:
      start = df.index[df['Timestamp'] == df1['Reverse_start'][i]].values[0]
    except:
      continue
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



def do_parser(path, TP, TREND, save_path, plot_path, s1_path, s2_path):
  source, bid = read_csv(path)
  max_range_idx, min_range_idx, start, end, reverse_start, reverse_end = parse_value(bid, TP, TREND)
  res_df = convert_to_df(source, bid, max_range_idx, min_range_idx, start, end, reverse_start, reverse_end)
  df = cluster_zero_tp(res_df)
  sum_1, sum_2 = summary(df)

  plot(source, res_df, plot_path)
  df.to_csv(save_path, index = False)
  sum_1.to_csv(s1_path, index = False)
  sum_2.to_csv(s2_path, index = False)
  print('Saved all files in output path !')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument('--TP', nargs='+', type=int)
    parser.add_argument('--TREND', nargs='+', type=int)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--Plot_path", type=str)
    parser.add_argument("--Summary1_path", type=str)
    parser.add_argument("--Summary2_path", type=str)
    args = parser.parse_args()
        
    do_parser(args.input_path, args.TP, args.TREND, args.output_path, args.Plot_path, args.Summary1_path, args.Summary2_path)





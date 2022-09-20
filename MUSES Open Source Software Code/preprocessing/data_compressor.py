""" Boyuan Chen:
This file quickly trims a data file to your desired data range. You can use it either on raw data file, or processed
data file.
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Semantic Scan args parser')

parser.add_argument('--input_file', type=str, default="../data/ED_2016/ED_2016_full.txt")
parser.add_argument('--output_file', type=str, default="../data/ED_2016/ED_2016_comma.txt")
parser.add_argument('--start_time', type=str, default="12/01/2015 00:00")
parser.add_argument('--end_time', type=str, default="12/27/2016 23:59")

FLAGS = parser.parse_args()

def compress_by_date(input_file, output_file, start_date, end_date):
    chunk_idx = 0
    chunksize = 10000
    reader = pd.read_csv(input_file, sep='\t', iterator=True, chunksize=chunksize, encoding='latin1', low_memory=False)
    for cc_data in reader:
        chunk_idx += 1
        print(f"Chunk {chunk_idx} starts")
        cc_data['date'] = cc_data['date'].astype(str)
        cc_data['time'] = cc_data['time'].astype(str)
        cc_data['time'] = cc_data['time'].str.replace('24:00', '23:59')
        cc_data = cc_data.join(
            cc_data['date'].str.split('/', expand=True).rename(index=int, columns={0: 'month', 1: 'day', 2: 'year'}))
        cc_data = cc_data.join(
            cc_data['time'].str.split(':', expand=True).rename(index=int, columns={0: 'hour', 1: 'minute'}))
        # Drop the rows with invalid date and time
        cc_data.dropna(subset=['day', 'month', 'year', 'hour', 'minute'], how='any', inplace=True)

        # Reconstruct valid full date
        cc_data['year'] = '20' + cc_data['year'].astype(str)
        cc_data['month'] = cc_data['month'].astype(str).str.zfill(2)
        cc_data['day'] = cc_data['day'].astype(str).str.zfill(2)
        cc_data['hour'] = cc_data['hour'].astype(str).str.zfill(2)
        cc_data['full_date'] = pd.to_datetime(
            cc_data['year'] + '/' + cc_data['month'] + '/' + cc_data['day'] + ' ' + cc_data['hour'] + ':' + cc_data[
                'minute'], format='%Y/%m/%d %H:%M')

        cc_data = cc_data[(cc_data['full_date'] >= start_date) & (cc_data['full_date'] <= end_date)]
        cc_data = cc_data.drop(columns=['month','day','year','hour','minute','full_date'])
        cc_data.to_csv(output_file, sep=",", index=False, header=(chunk_idx==1), mode='a', encoding='utf-8')


def main(args):
    pd.set_option('expand_frame_repr', False)
    compress_by_date(args.input_file, args.output_file, args.start_time, args.end_time)

if __name__=="__main__":
    main(FLAGS)
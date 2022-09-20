import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Semantic Scan args parser')

parser.add_argument('--input_file', type=str, default="../data/Covid/ED_2020.txt")
parser.add_argument('--output_file', type=str, default="../data/Covid/ED_2020_formatted.txt")

FLAGS = parser.parse_args()

def date_format(input_file, output_file):
    chunk_idx = 0
    chunksize = 10000
    reader = pd.read_csv(input_file, sep='\t', iterator=True, chunksize=chunksize, encoding='latin1', low_memory=False)
    for cc_data in reader:
        chunk_idx += 1
        print(f"Chunk {chunk_idx} starts")
        cc_data['date'] = cc_data['date'].astype(str)
        cc_data['time'] = cc_data['time'].astype(str)
        cc_data = cc_data.join(
            cc_data['date'].str.split('-', expand=True).rename(index=int, columns={0: 'year', 1: 'month', 2: 'day'}))
        cc_data = cc_data.join(
            cc_data['time'].str.split(':', expand=True).rename(index=int, columns={0: 'hour', 1: 'minute', 2: 'second'}))
        # Drop the rows with invalid date and time
        cc_data.dropna(subset=['day', 'month', 'year', 'hour', 'minute', 'second'], how='any', inplace=True)

        # Reconstruct valid full date
        cc_data['year'] = cc_data['year'].astype(str).str[-2:]
        cc_data['month'] = cc_data['month'].astype(str).str.zfill(2)
        cc_data['day'] = cc_data['day'].astype(str).str.zfill(2)
        cc_data['hour'] = cc_data['hour'].astype(str).str.zfill(2)
        cc_data['date'] = cc_data['month'] + '/' + cc_data['day'] + '/' + cc_data['year']
        cc_data['time'] = cc_data['hour'] + ':' + cc_data['minute']

        cc_data = cc_data.drop(columns=['month','day','year','hour','minute','second'])
        cc_data.to_csv(output_file, sep="\t", index=False, header=(chunk_idx==1), mode='a', encoding='utf-8')


def main(args):
    pd.set_option('expand_frame_repr', False)
    date_format(args.input_file, args.output_file)

if __name__=="__main__":
    main(FLAGS)
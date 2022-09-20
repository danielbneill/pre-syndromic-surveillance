""" Boyuan Chen 02/18/2022:
This is a script to skim through the data file one chunksize each time. It mainly helps users to
get a general picture of the data, as well as understanding what possible noises are there.
To run this, put in the right filename is the argument part. In random mode, you will see the
rows at random; in sequential mode, you will see the rows sequentially.
"""

import pandas as pd
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser(description='Data file skimmer')

parser.add_argument('--mode', type=str, default='sequential', help="random, sequential or target")
parser.add_argument('--data_file', type=str, default='../data/Covid/ED_2020.txt')
parser.add_argument('--sep', type=str, default='\t')
parser.add_argument('--chunksize', type=int, default=20, help="How many rows you want to see each time")
parser.add_argument('--target', type=int, default=657938, help="the row index you want to see")

FLAGS = parser.parse_args()

def main(args):
    full_reader = pd.read_csv(args.data_file, sep=args.sep, encoding='latin1', low_memory=False)
    print(f"The data file has the following HOSP code: \n{np.sort(full_reader['hospcode'].unique())}")
    data_size = len(full_reader)
    print(f"The data has {data_size} caselines")
    if args.mode=="sequential":
        reader = pd.read_csv(args.data_file, sep=args.sep, iterator=True, chunksize=args.chunksize,
                                encoding='latin1', low_memory=False)
        pd.set_option('expand_frame_repr', False)
        for cc_data in reader:
            print(cc_data)
            input()

    elif args.mode=="random":
        pd.set_option('expand_frame_repr', False)
        while True:
            num_chunks = int(data_size/args.chunksize)  # fixed DBN 9-17-2022
            index = random.randint(0, num_chunks-1)
            print(full_reader.iloc[index*args.chunksize:index*args.chunksize+args.chunksize])
            input()

    elif args.mode=="target":
        data = full_reader[['cc', 'date', 'time']]
        pd.set_option('expand_frame_repr', False)
        target = args.target
        lower = target-2 if target>=2 else 0
        upper = target+3 if target<=len(data.index)-3 else len(data.index)-1
        print(data.iloc[lower:upper])

    else:
        print("Please put 'random', 'sequential' or 'target' in mode")

if __name__ == '__main__':
    main(FLAGS)
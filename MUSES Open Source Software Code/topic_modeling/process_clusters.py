"""
A simple script to process the scan output.
"""

import argparse
from utils import process_clusters

parser = argparse.ArgumentParser(description='Process clusters args parser')

parser.add_argument('--cluster_dir', type=str, default="../data/Covid/cluster_MarApr")
parser.add_argument('--concatenate_agegroup', type=str, default="False")

FLAGS = parser.parse_args()


def main(args):
    concatenate_agegroup = False
    if args.concatenate_agegroup=="True" or args.concatenate_agegroup=="true" or args.concatenate_agegroup=="TRUE":
        concatenate_agegroup = True
    process_clusters(args.cluster_dir, concatenate_agegroup)

if __name__=="__main__":
    main(FLAGS)

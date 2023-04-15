"""
A simple script to process the scan output.
"""

import argparse
from utils import process_clusters

parser = argparse.ArgumentParser(description='Process clusters args parser')

parser.add_argument('--cluster_dir', type=str, default="../data/Covid/cluster_MarApr")
parser.add_argument('--concatenate_agegroup', type=str, default="False")
parser.add_argument('--merged_cluster', type=str, default="False")
parser.add_argument('--cluster_type', type=str, default="novel")

FLAGS = parser.parse_args()


def main(args):
    concatenate_agegroup = False
    if args.concatenate_agegroup=="True" or args.concatenate_agegroup=="true" or args.concatenate_agegroup=="TRUE":
        concatenate_agegroup = True
    merged_cluster = False
    if args.merged_cluster=="True" or args.merged_cluster=="true" or args.merged_cluster=="TRUE":
        merged_cluster = True
    if args.cluster_type=="monitored" or args.cluster_type=="Monitored" or args.cluster_type=="MONITORED":
        cluster_type = "monitored"
    if args.cluster_type=="novel" or args.cluster_type=="Novel" or args.cluster_type=="NOVEL":
        cluster_type = "novel"
    process_clusters(args.cluster_dir, concatenate_agegroup, merged_cluster, cluster_type)
    
if __name__=="__main__":
    main(FLAGS)

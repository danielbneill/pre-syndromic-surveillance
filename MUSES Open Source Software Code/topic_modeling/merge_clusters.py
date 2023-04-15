"""
A simple script to process the scan output.
"""
import os.path
import argparse
from utils import merge_clusters, process_clusters

parser = argparse.ArgumentParser(description='Merge clusters args parser')

parser.add_argument('--cluster_dir', type=str, default="../data/synthetic_data/clusters")
parser.add_argument('--merge_on_duplicate_case_only', type=str, default="True")
parser.add_argument('--merge_window', type=float, default=1)
parser.add_argument('--similarity_threshold', type=float, default=0.90)

FLAGS = parser.parse_args()


def main(args):

	cluster_dir = args.cluster_dir
	merge_on_duplicate_case_only = True
	if args.merge_on_duplicate_case_only=="False" or args.merge_on_duplicate_case_only=="false" or args.merge_on_duplicate_case_only=="FALSE":
		merge_on_duplicate_case_only = False
	merge_window = args.merge_window
	similarity_threshold = args.similarity_threshold
  
	#run on files if exists
	if os.path.isfile(cluster_dir+"/novel_raw.csv"):
		merge_clusters(cluster_dir+"/novel_raw.csv", cluster_dir, "novel", merge_on_duplicate_case_only, merge_window, similarity_threshold)
		process_clusters(cluster_dir, True, True, "novel")

	if os.path.isfile(cluster_dir+"/monitored_raw.csv"):
		merge_clusters(cluster_dir+"/monitored_raw.csv", cluster_dir, "monitored", merge_on_duplicate_case_only, merge_window, similarity_threshold)
		process_clusters(cluster_dir, True, True, "monitored")
		
if __name__=="__main__":
    main(FLAGS)
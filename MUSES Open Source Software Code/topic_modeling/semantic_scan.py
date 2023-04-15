""" Semantic Scan:
This is the step after static training. It scans through the data window-by-window. In each time window, we
first train a static LDA model, and then we train a contrastive LDA model along with the static topics on the
background. Finally, we scan each foreground and monitored topic, each search group, and find the cluster with
the highest score F(S).
"""

import pandas as pd
import numpy as np
import argparse
import pickle
import os
from SpatialScan import SpatialScan
from utils import topic_strings_to_Phi_b, process_clusters

parser = argparse.ArgumentParser(description='Semantic Scan args parser')

# Data files
parser.add_argument('--full_scan_file', type=str, default="../data/tainted_coffee/processed_tc.csv")
parser.add_argument('--dict_file', type=str, default="../data/tainted_coffee/word_dict_tc.txt")
parser.add_argument('--sg_dict', type=str, default="../data/tainted_coffee/sg_dict_tc.pickle")
parser.add_argument('--sg_list', type=str, default="../data/tainted_coffee/sg_tc.csv")

# Checkpoints
parser.add_argument('--static_checkpoint', type=str, default="../data/tainted_coffee/checkpoint/tc_static25.npz")
parser.add_argument('--import_monitored', type=str, default="")

# Foreground topic model training
parser.add_argument('--dynamic_model_type', type=str, default='ContrastiveLDA')
parser.add_argument('--num_foreground_topics', type=int, default=5)
parser.add_argument('--static_iters', type=int, default=1000)
parser.add_argument('--contrastive_iters', type=int, default=1000)
parser.add_argument('--verbose', type=str, default="True")

# Spatial scan
parser.add_argument('--step_size', type=int, default=1, help="How many hours to jump forward in each iter")
parser.add_argument('--window_size', type=int, default=3, help="How many hours in each foreground window")
parser.add_argument('--baseline_size', type=int, default=21, help="How many days in baseline before the windowed date")
parser.add_argument('--score_threshold', type=float, default=3.0)
parser.add_argument('--start_date', type=str, default="2014/11/24")
parser.add_argument('--end_date', type=str, default="2014/11/24")
parser.add_argument('--cluster_word_num', type=int, default=12)

# Cluster processing
parser.add_argument('--topic_weight', type=str, default="True")
parser.add_argument('--cluster_dir', type=str, default="../data/tainted_coffee/cluster")
parser.add_argument('--concatenate_agegroup', type=str, default="False")
parser.add_argument('--merged_cluster', type=str, default="False")
parser.add_argument('--cluster_type', type=str, default="novel")

FLAGS = parser.parse_args()



def main(args):
    if args.topic_weight=="False":
        topic_weight = False
    elif args.topic_weight=="True":
        topic_weight = True
    else:
        raise Exception("Please give topic_weight a boolean value")

    if args.cluster_type=="monitored" or args.cluster_type=="Monitored" or args.cluster_type=="MONITORED":
        cluster_type = "monitored"
    else:
        cluster_type = "novel"
        
    cluster_dir = args.cluster_dir
    pd.options.mode.chained_assignment = None
    if not os.path.exists(cluster_dir):
        os.mkdir(cluster_dir)

    # Prepare the word-to-index dictionary
    word_dict_df = pd.read_csv(args.dict_file)
    word_dict = {}
    for i in range(len(word_dict_df)):
        word_dict[word_dict_df.loc[i, 'word']] = word_dict_df.loc[i, 'index']
    print("Word-index dictionary and remove list loaded.\n")

    # --- Load background checkpoint ---
    print("Loading background checkpoint...")
    monitored_static_topics = []
    if args.import_monitored != "" and args.static_checkpoint == "":
        topic_strings = pd.read_csv(args.import_monitored, header=None).iloc[:, 0].values
        Phi_b, monitored_static_topics = topic_strings_to_Phi_b(topic_strings, word_dict)
    elif args.import_monitored == "" and args.static_checkpoint != "":
        static_checkpoint = np.load(args.static_checkpoint, allow_pickle=True)
        Phi_b = static_checkpoint['Phi']
    elif args.import_monitored != "" and args.static_checkpoint != "":
        topic_strings = pd.read_csv(args.import_monitored, header=None).iloc[:, 0].values
        Phi_b_monitored, monitored_static_topics = topic_strings_to_Phi_b(topic_strings, word_dict)
        static_checkpoint = np.load(args.static_checkpoint, allow_pickle=True)
        Phi_b_static = static_checkpoint['Phi']
        Phi_b = np.concatenate((Phi_b_monitored, Phi_b_static), axis=0)
    else:
        raise Exception("Please provide at least one valid static file")

    K_prime = args.num_foreground_topics
    n_t = Phi_b.shape[0] + K_prime
    alpha = 1 / n_t
    beta = 1 / Phi_b.shape[1]

    # BOW-format the input dataframe and assign topic based on trained Phi
    print("Loading the processed df for scan...")
    scan_df = pd.read_csv(args.full_scan_file, header=[0])

    # Read the list of all search groups
    with open(args.sg_dict, 'rb') as handle:
        sg_lookup_dict = pickle.load(handle)
    sg_list_df = pd.read_csv(args.sg_list)
    sg_list = sg_list_df['location_ids'].values.tolist()

    # Initialize the names for output files
    novel_raw_file = args.cluster_dir + "/novel_raw.csv"
    monitored_raw_file = args.cluster_dir + "/monitored_raw.csv"

    # Process verbose from string to boolean
    if args.verbose=="True" or args.verbose=="true":
        verbose = True
    else:
        verbose = False

    # --- Do the spatial scan and generate clustered raw caselines ---
    print("\nSpatial Scan starts.")
    spatial_scan = SpatialScan(args.step_size, args.window_size, args.baseline_size, args.score_threshold,
                               topic_weight, verbose, args.cluster_word_num, novel_raw_file,
                               monitored_raw_file, epsilon=0.5)
    spatial_scan.scan(scan_df, sg_lookup_dict, sg_list, Phi_b, K_prime, alpha, beta, args.static_iters,
                      args.contrastive_iters, word_dict, args.start_date, args.end_date, monitored_static_topics)


    # --- Generate the files that we need for caseline visualization ---
    concatenate_agegroup = False
    if args.concatenate_agegroup=="True" or args.concatenate_agegroup=="true" or args.concatenate_agegroup=="TRUE":
        concatenate_agegroup = True
    merged_cluster = False 
    if args.merged_cluster=="True" or args.merged_cluster=="true" or args.merged_cluster=="TRUE":
        merged_cluster = True
    process_clusters(args.cluster_dir, concatenate_agegroup, merged_cluster, cluster_type)


if __name__ == '__main__':
    main(FLAGS)

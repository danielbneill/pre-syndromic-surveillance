""" Train Staitic Topic Modle on Background Data

This script trains the static/background topic model, by creating an object of StaticModel. The output of
this script is the checkpoint, named in format of [{datasize}_K{args.K}.npz]. It saves the following values:

Theta: The count for document-topic distribution, with shape (n_d,n_t);
Phi: The count for topic-word distribution, with shape (n_t, n_w);
topic_count: The count for the total number of words assigned to each topic in the whole corpus;
static_alpha: The Dirichlet prior for Theta;
static_beta: The Dirichlet prior for Phi;
model_type: "StaticLDA";
X: The BOW format corpus;
Z: Topic assignment for each word in the whole corpus.

These values will be used in dynamic topic modeling on foreground data, and the semantic scan.
"""

from StaticModel import StaticLDA
from utils import gen_bow_corpus, Phi_to_dist

import argparse
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser(description='Semantic Scan args parser')

# Data files
parser.add_argument('--data_file', type=str, default="../data/ED_2016/processed_halfyear.csv")
parser.add_argument('--dict_file', type=str, default="../data/hurricane_sandy/word_dict_ED_hs.txt")

# Static Topic Model Parameters
parser.add_argument('--num_static_topics', type=int, default=50, help='Number of static topics')
parser.add_argument('--static_model_type', type=str, default='StaticLDA')
parser.add_argument('--static_iters', type=int, default=3000)
parser.add_argument('--verbose', type=int, default=100)

# Output
parser.add_argument('--checkpoint', type=str, default="../data/ED_2016/checkpoint/halfyear_static50.npz")

FLAGS = parser.parse_args()



def main(args):
    # Make sure the checkpoint directory exists
    checkpoint_dir = args.checkpoint.rsplit('/',1)[0]
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Note that this word dict is on the full dataset, not just training or testing. There is chance
    # that some words won't appear in the training set.
    word_dict_df = pd.read_csv(args.dict_file)
    word_dict = {}
    for i in range(len(word_dict_df)):
        word_dict[word_dict_df.loc[i, 'word']] = word_dict_df.loc[i, 'index']

    # Generating the training dataset in numpy format
    print("\nLoading training dataframe to corpus...")
    X = gen_bow_corpus(args.data_file, word_dict, sep=',', chunksize=100000)
    print(f"Data loaded. Training set has {X.shape[0]} caselines. \n")

    # --- Train background topic model ---
    n_d = len(X)
    n_w = len(word_dict)
    K = args.num_static_topics
    static_alpha = 1/K
    static_beta  = 1/n_w
    print("Initializing static LDA model based on corpus. This might take a while...\n")
    static_lda = StaticLDA(X, n_d, n_w, K, static_alpha, static_beta)
    print("Static training starts.")
    static_lda.train(X, iterations=args.static_iters, verbose=args.verbose)

    # Save checkpoint
    Phi_count = np.array(static_lda.get_Phi())
    alpha = np.array(static_lda.get_alpha())
    beta  = np.array(static_lda.get_beta())
    Phi_dist = Phi_to_dist(Phi_count, beta)
    checkpoint_name = args.checkpoint
    np.savez(checkpoint_name, Phi=Phi_dist, model_type="StaticLDA")
    print(f"Staitic checkpoint saved to {checkpoint_name}")



if __name__ == '__main__':
    main(FLAGS)

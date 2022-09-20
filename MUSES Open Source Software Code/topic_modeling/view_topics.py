import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Preprocessing args parser')
parser.add_argument('--checkpoint', type=str, default='../data/tainted_coffee/checkpoint/tc_static25.npz')
parser.add_argument('--word_dict', type=str, default='../data/tainted_coffee/word_dict_tc.txt')
parser.add_argument('--display_words', type=int, default=10)
parser.add_argument('--foreground_topic_num', type=int, default=0)
FLAGS = parser.parse_args()


def main(args):
    checkpoint = np.load(args.checkpoint, allow_pickle=True)
    Phi = checkpoint['Phi']
    print(f"Phi's shape is: {Phi.shape}")
    word_dict_df = pd.read_csv(args.word_dict)
    for t in range(Phi.shape[0]):
        print(f"\nTop words in topic {t}:")
        topic_arr = Phi[t, :]
        sorting_index = np.argsort(topic_arr)
        topic_df = pd.DataFrame(columns=["word", "weight"])
        topic_df["word"] = word_dict_df.loc[sorting_index[::-1][:args.display_words].tolist(), "word"]
        topic_df["weight"] = topic_arr[sorting_index][::-1][:args.display_words]
        print(topic_df)

if __name__=="__main__":
    main(FLAGS)
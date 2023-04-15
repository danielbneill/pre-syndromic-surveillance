""" Spatial Scan:
SpatialScan's scan function does everything for you. It slides through the data window by window, train
static LDA on windowed data, train contrastive LDA on windowed data, and finally calculates F(S) score
and writes the cluster with the highest score.

Note that F_S = B - C + C*math.log(C/B), which is uncalculable if B=0. When C>B=0, we set B=epsilon,
a small scalar to make the equation valid. We usually set it to 0.5.
"""
import sys
import pandas as pd
import numpy as np
import math
from datetime import timedelta
from StaticModel import StaticLDA
from ContrastiveModel import ContrastiveLDA
from utils import get_word_dist, gen_bow_corpus_df, str_to_datetime, get_sg_column, get_caseline_in_sg, Phi_to_dist, \
    assign_topic_from_Phi


class SpatialScan:
    def __init__(self, step_size, window_size, baseline_size, score_threshold, topic_weight, verbose,
                 cluster_word_num, novel_caseline_file, monitored_caseline_file, epsilon=0.5):
        self.step_size = step_size
        self.window_size = window_size
        self.baseline_size = baseline_size
        self.score_threshold = score_threshold
        self.topic_weight = topic_weight
        self.verbose = verbose
        self.cluster_word_num = cluster_word_num
        self.novel_caseline_file = novel_caseline_file
        self.monitored_caseline_file = monitored_caseline_file
        # This is for when B=0 in F(S) calculation
        self.epsilon = epsilon


    # Here, df is the foreground data. It should contain the range of scan, as well as a month earlier than that,
    # since we need those days for baseline calculation.
    def scan(self, df, sg_lookup_dict, sg_list, Phi_b, K_prime, alpha, beta, static_iters, contrast_iters,
             word_dict, start_date, end_date, monitored_indices):
        # Initialize headers for caseline files
        novel_caselines = pd.DataFrame(columns=["index", "score", "topic", "cc", "cc_processed", "icd", "VISITID", "time", "date",
                                          "sex", "agegroup", "hospcode", "word_dist"])
        novel_caselines.to_csv(self.novel_caseline_file, index=False, header=True)
        monitored_caselines = pd.DataFrame(columns=["index", "score", "topic", "cc", "cc_processed", "icd", "VISITID", "time", "date",
                                          "sex", "agegroup", "hospcode", "word_dist"])
        monitored_caselines.to_csv(self.monitored_caseline_file, index=False, header=True)

        # Create date_time column and sort for both background and foreground data.
        df['date'] = df['date'].astype(str)
        df['time'] = df['time'].astype(str)
        df['time'] = df['time'].str.replace('24:00', '23:59')
        df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M')
        df = df.sort_values(by="date_time")
        earliest_date = df['date'].iloc[0]
        latest_date = df['date'].iloc[-1]
        df.fillna('', inplace=True)
        if start_date < earliest_date:
            print(f"start date is too early. Changed to {earliest_date}")
            start_date = earliest_date
        if end_date > latest_date:
            print(f"end date is too late. Changed to {latest_date}")
            end_date = latest_date

        # Initialize parameters. Note that we start from the previous day's 10pm to current day's 1am.
        n_w = len(word_dict.keys())
        # Note: novel and monitored clusters will have different cluster index. So each file will have indices
        # consecutively from 0.
        novel_cluster_idx = 0
        monitored_cluster_idx = 0
        start_datetime = str_to_datetime(start_date+" 00:00") - timedelta(hours=(self.window_size - 1))
        end_datetime = start_datetime + timedelta(hours=self.window_size) - timedelta(seconds=1)
        last_datetime = str_to_datetime(end_date+" 23:59") + timedelta(seconds=59)

        # Compute all distinct elements of search groups
        sg_all = []
        for sg_txt in sg_list:
            sg = str(sg_txt).split("-")
            sg = [int(i) for i in sg]
            sg_all = sg_all + sg
        sg_all = list(set(sg_all))

        # Outmost loop over time
        while end_datetime <= last_datetime:
            print(f"Scanning from {start_datetime} to {end_datetime}")
            if self.verbose:   # added DBN 9-16-2022
              print()

            # Prepare windowed data and generate bag-of-words corpus for static and contrastive training
            window_df = df[(df['date_time'] >= start_datetime) & (df['date_time'] <= end_datetime)]
            if len(window_df)==0:
                start_datetime = start_datetime + timedelta(hours=self.step_size)
                end_datetime = end_datetime + timedelta(hours=self.step_size)
                if self.verbose:
                    print("Window length 0. Moving on to next time window. ")
                    print()
                continue
            start_baseline_datetime = start_datetime - timedelta(days=self.baseline_size)
            window_df = get_sg_column(window_df, sg_lookup_dict)
            window_X  = gen_bow_corpus_df(window_df, word_dict)
            n_d = len(window_X)

            # --- Train static topic model ---
            static_lda = StaticLDA(window_X, n_d, n_w, K_prime, alpha, beta)
            static_lda.train(window_X, iterations=static_iters, verbose=-1)
            Phi_f_count = static_lda.get_Phi()
            Phi_f = Phi_to_dist(Phi_f_count, beta)
            np.set_printoptions(threshold=sys.maxsize)

            # --- Train contrastive topic model ---
            contrastive_lda = ContrastiveLDA(window_X, np.array(Phi_b), np.array(Phi_f), alpha, beta)
            contrastive_lda.train(window_X, iterations=contrast_iters, verbose=-1)
            Phi = np.array(contrastive_lda.get_Phi())
            Phi = Phi_to_dist(Phi, beta)
            Phi = np.concatenate((Phi_b, Phi[-K_prime:]), axis=0)

            # Prepare the baseline data frame for F(S) calculation; assign search group index
            baseline_df = df[(df['date_time'] >= start_baseline_datetime)
                           & (df['date_time'] <= start_datetime)]
            search_group_values = np.zeros(len(baseline_df), dtype=int)
            for i in range(len(baseline_df)):
                hosp = baseline_df['hospcode'].iloc[i]
                age = baseline_df['agegroup'].iloc[i]
                sg_number = sg_lookup_dict[(hosp, age)]
                search_group_values[i] = sg_number
            baseline_df['search_group'] = search_group_values

            # Assign topics to baseline and window df
            baseline_X = gen_bow_corpus_df(baseline_df, word_dict)
            baseline_df['topic'] = assign_topic_from_Phi(baseline_X, Phi, alpha, iters=50).astype(int)
            window_df['topic'] = assign_topic_from_Phi(window_X, Phi, alpha, iters=50).astype(int)

            # --- Spatial Scan ---
            # First get the scanning topic indices. We only scan over foreground topics and monitored static
            # topics. Other static topics should be ignored.
            scan_topic_indices = list(range(Phi.shape[0]-K_prime, Phi.shape[0]))
            scan_topic_indices = monitored_indices + scan_topic_indices
            for t in scan_topic_indices:
                # Scan through each age-location search group S and calculate their score F(S)
                scan_baseline_df = baseline_df[baseline_df['topic'] == t]
                scan_baseline_df['hour'] = scan_baseline_df['date_time'].apply(lambda x: x.hour)
                # Now baseline_df has columns ['cc', 'date', 'time', 'sex', 'hospcode', 'agegroup',
                # 'location_forscan', 'topic', 'date_time', 'search_group', 'hour']
                scan_window_df = window_df[window_df['topic'] == t]
                if len(scan_window_df)==0:
                    # if self.verbose:  # removed DBN 9-16-2022
                    #    print()
                    continue
                    
                if self.verbose:
                    print("Scanning topic", t, "with", len(scan_window_df), "cases in window and",
                          len(scan_baseline_df), "cases in baseline")
                    print(scan_window_df)

                # Calculate maximum score. If even the hypothetical maximum of this window won't exceed
                # the score threshold, we will just skip this topic.
                hypothetical_max_count = scan_window_df['hospcode'].value_counts().max()
                hypothetical_min_baseline = self.epsilon
                hypothetical_max_score = hypothetical_max_count * math.log(hypothetical_max_count/hypothetical_min_baseline) \
                                         + hypothetical_min_baseline - hypothetical_max_count
                if hypothetical_max_score < self.score_threshold:
                    if self.verbose:
                        print("Shortcut! Hypothetical maximum score of", hypothetical_max_score,
                              "is less than score threshold of", self.score_threshold)
                        print()
                    continue
                
                # compute individual c_s and b_s for all elements of all search groups
                count_dict = {}
                baseline_dict = {}
                c_s = np.zeros(self.window_size)
                b_s = np.zeros(self.window_size)

                for sg in sg_all:
                    for h in range(self.window_size):
                        sub_start_datetime = start_datetime + timedelta(hours=h)
                        sub_end_datetime = sub_start_datetime + timedelta(minutes=59)
                        sub_window_df = scan_window_df[(scan_window_df['date_time'] >= sub_start_datetime) &
                                                  (scan_window_df['date_time'] <= sub_end_datetime)]

                        # Get the aggregate count c in this hour
                        sg_window_nparray = sub_window_df['search_group'].to_numpy()
                        c_s[h] = (sg_window_nparray == sg).sum() 

                        # Get the baseline b in this hour, (AC_h+AC_oh)/2.
                        this_hour = sub_start_datetime.hour
                        baseline_h  = scan_baseline_df[scan_baseline_df['hour'] == this_hour]
                        baseline_oh = scan_baseline_df[scan_baseline_df['hour'] != this_hour]
                        sg_h_nparray  = baseline_h['search_group'].to_numpy()
                        sg_oh_nparray = baseline_oh['search_group'].to_numpy()
                        AC_h = (sg_h_nparray == sg).sum()         
                        AC_oh = ((sg_oh_nparray == sg).sum())/23  # Average over the other 23 hours of the day.
                        b_s[h] = (AC_h+AC_oh)/(2*self.baseline_size)    # Get average over days.

                    count_dict[sg] = c_s.copy()
                    baseline_dict[sg] = b_s.copy()

                # Loop through each search group (sg) to find the cluster with the highest F(S) in
                # this time window. Note that for a 3-hour window, say 00:00-02:59, we need to calculate
                # three F values: [02:00-02:59, 01:00-02:59, 00:00-02:59].
                # To calculate B and C for each of them, we need arrays to store values of b and c over
                # each single hour.
                highest_cluster = pd.DataFrame(columns=["index", "score", "topic", "cc", "icd",
                                "VISITID", "time", "date", "sex", "agegroup", "hospcode", "word_dist"])
                highest_F_S = 0
                highest_C = 0
                highest_B = 0
                for sg_txt in sg_list:
                    c_s = np.zeros(self.window_size)
                    b_s = np.zeros(self.window_size)
                    sg_current = str(sg_txt).split("-")
                    sg_current = [int(i) for i in sg_current]
                    # Add up each singular search group to get total. For example, sg_current could be
                    # 84-88-92, so we add up count and baseline in each of them.
                    for i in sg_current:                               
                        c_s = c_s + count_dict[i]
                        b_s = b_s + baseline_dict[i]

                    # Get all caselines for each hour in this search group. It will be used to find the highest
                    # cluster by concatenating consecutive hours.
                    clusters_each_hour = []
                    for h in range(self.window_size):
                        sub_start_datetime = start_datetime + timedelta(hours=h)
                        sub_end_datetime = sub_start_datetime + timedelta(minutes=59)
                        sub_window_df = scan_window_df[(scan_window_df['date_time'] >= sub_start_datetime) &
                                                  (scan_window_df['date_time'] <= sub_end_datetime)]
                        sg_window_nparray = sub_window_df['search_group'].to_numpy()
                        caseline_indices = np.array([])
                        for i in range(len(sg_window_nparray)):
                            if sg_window_nparray[i] in sg_current:
                                caseline_indices = np.append(caseline_indices, i)
                        clusters_each_hour.append(sub_window_df.iloc[caseline_indices])

                    # Calculate F(S) of all hour-sizes ending on the time-window's end. For example, if the
                    # window is 3 hours, then we calculate F(S) for 123, 23, 3.
                    for i in range(1, c_s.shape[0]+1):
                        C = np.sum(c_s[-i:])
                        B = np.sum(b_s[-i:])
                        if B < self.epsilon:
                            B = self.epsilon
                        F_S = 0
                        if C>B:
                            F_S = B - C + C*math.log(C/B)
                        if F_S > highest_F_S and F_S >= self.score_threshold:
                            highest_F_S = F_S
                            highest_C = C
                            highest_B = B
                            highest_cluster = pd.concat(clusters_each_hour[-i:])
                            highest_cluster["score"] = F_S
                            highest_cluster["word_dist"] = get_word_dist(highest_cluster["cc_processed"], word_dict,
                                                                    Phi, t, self.cluster_word_num, self.topic_weight)

                # Append valid cluster
                if highest_F_S > 0:
                    if highest_cluster["topic"].iloc[0] in monitored_indices:
                        highest_cluster_idx = monitored_cluster_idx
                        highest_cluster["index"] = highest_cluster_idx
                        print(f"Monitored cluster {highest_cluster_idx} appended. Length={len(highest_cluster)}; "
                              f"F={highest_F_S}; C={highest_C}; B={highest_B}.")
                        monitored_cluster_idx += 1
                        highest_cluster = highest_cluster.reindex(columns=["index", "score", "topic", "cc", "cc_processed", "icd",
                                            "VISITID", "time", "date", "sex", "agegroup", "hospcode", "word_dist"])
                        highest_cluster.to_csv(self.monitored_caseline_file, index=False, header=False, mode='a', encoding='utf-8')

                    else:  # Cluster in novel topic
                        highest_cluster_idx = novel_cluster_idx
                        highest_cluster["index"] = highest_cluster_idx
                        print(f"Novel cluster {highest_cluster_idx} appended. Length={len(highest_cluster)}; "
                              f"F={highest_F_S}; C={highest_C}; B={highest_B}.")
                        novel_cluster_idx += 1
                        highest_cluster = highest_cluster.reindex(columns=["index", "score", "topic", "cc", "cc_processed", "icd",
                                            "VISITID", "time", "date", "sex", "agegroup", "hospcode", "word_dist"])
                        highest_cluster.to_csv(self.novel_caseline_file, index=False, header=False, mode='a', encoding='utf-8')

                if self.verbose:
                    print()

            # Move forward start and end datetime by one step size of hours
            start_datetime = start_datetime + timedelta(hours=self.step_size)
            end_datetime = end_datetime + timedelta(hours=self.step_size)
            if self.verbose:
                print("--------------------------------------------------------\n")

"""
Helper functions for topic model training and spatial scan.
"""
import pandas as pd
import numpy as np
import pdb
import os
from datetime import datetime
from numba import njit
from flashtext import KeywordProcessor
from nltk.stem.wordnet import WordNetLemmatizer

def minimum(a, b):
    if a <= b:
        return a
    else:
        return b

def gen_bow_corpus(df_file, word_dict, sep=',', chunksize=100000):
    corpus = []
    max_len = 0
    reader = pd.read_csv(df_file, sep=sep, iterator=True, chunksize=chunksize, encoding='latin1', low_memory=False)
    for df in reader:
        cc = df['cc_processed']
        for string in cc:
            str_list = string.split('_')
            doc_dict = {}
            doc_list = []
            for i in range(len(str_list)):
                if str_list[i] in word_dict:   # We see a valid word. If you see sth like "of", it won't count
                    word_index = word_dict[str_list[i]]
                    if word_index not in doc_dict:
                        doc_dict[word_index] = 1
                    else:
                        doc_dict[word_index] += 1
            for word_i in doc_dict:
                doc_list.append((word_i, doc_dict[word_i]))
            corpus.append(doc_list)
            if len(doc_list) > max_len:
                max_len = len(doc_list)
    return X_to_numpy(corpus, max_len)


def gen_bow_corpus_df(df, word_dict):
    corpus = []
    max_len = 0
    cc = df['cc_processed']
    for string in cc:
        str_list = string.split('_')
        doc_dict = {}
        doc_list = []
        for i in range(len(str_list)):
            if str_list[i] in word_dict:   # We see a valid word. If you see sth like "of", it won't count
                word_index = word_dict[str_list[i]]
                if word_index not in doc_dict:
                    doc_dict[word_index] = 1
                else:
                    doc_dict[word_index] += 1
        for word_i in doc_dict:
            doc_list.append((word_i, doc_dict[word_i]))
        corpus.append(doc_list)
        if len(doc_list) > max_len:
            max_len = len(doc_list)
    return X_to_numpy(corpus, max_len)


def gen_bow_corpus_bf(background_file, foreground_file, word_dict, sep=',', chunksize=100000):
    corpus = []
    max_len = 0
    background_reader = pd.read_csv(background_file, sep=sep, iterator=True, chunksize=chunksize, encoding='latin1')
    for df in background_reader:
        cc = df['cc_processed']
        for string in cc:
            str_list = string.split('_')
            doc_dict = {}
            doc_list = []
            for i in range(len(str_list)):
                if str_list[i] in word_dict:   # We see a valid word. If you see sth like "of", it won't count
                    word_index = word_dict[str_list[i]]
                    if word_index not in doc_dict:
                        doc_dict[word_index] = 1
                    else:
                        doc_dict[word_index] += 1
            for word_i in doc_dict:
                doc_list.append((word_i, doc_dict[word_i]))
            corpus.append(doc_list)
            if len(doc_list) > max_len:
                max_len = len(doc_list)
    foreground_reader = pd.read_csv(foreground_file, sep=sep, iterator=True, chunksize=chunksize, encoding='latin1')
    for df in foreground_reader:
        cc = df['cc_processed']
        for string in cc:
            str_list = string.split('_')
            doc_dict = {}
            doc_list = []
            for i in range(len(str_list)):
                if str_list[i] in word_dict:   # We see a valid word. If you see sth like "of", it won't count
                    word_index = word_dict[str_list[i]]
                    if word_index not in doc_dict:
                        doc_dict[word_index] = 1
                    else:
                        doc_dict[word_index] += 1
            for word_i in doc_dict:
                doc_list.append((word_i, doc_dict[word_i]))
            corpus.append(doc_list)
            if len(doc_list) > max_len:
                max_len = len(doc_list)
    return X_to_numpy(corpus, max_len)


def X_to_numpy(X, max_len):
    X_np = np.full((len(X), max_len, 2), -1)
    for i in range(len(X)):
        for j in range(len(X[i])):
            X_np[i][j][0] = X[i][j][0]
            X_np[i][j][1] = X[i][j][1]
    return X_np


def Z_to_numpy(Z, max_len, n_t):
    Z_np = np.full((len(Z), max_len, n_t), -1)
    for i in range(len(Z)):
        for j in range(len(Z[i])):
            for t in range(n_t):
                Z_np[i][j][t] = Z[i][j][t]
    return Z_np


# This function assumes that W = [0,1,2,...,n_w-1]. wt short for word-topic
@njit
def wt_from_Phi(Phi, w):
    wt_dist = Phi[:,w]
    wt_dist = wt_dist / np.sum(wt_dist)
    return wt_dist


def Theta_Phi_to_dist(Theta_count, Phi_count, alpha, beta):
    Theta = Theta_count[:]
    Phi = Phi_count[:]
    for i in range(len(Theta_count)):
        # There is a small chance that the document has no valid word in dict so Theta will be all 0.
        # In that case, we just ignore that one.
        if not np.any(Theta[i]):
            continue
        Theta[i] = (Theta_count[i]+alpha) / (np.sum(Theta_count[i]) + alpha*Theta.shape[1])
    for j in range(len(Phi_count)):
        Phi[j] = (Phi_count[j]+beta) / (np.sum(Phi_count[j]) + beta*Phi.shape[1])
    return Theta, Phi


def Phi_to_dist(Phi_count, beta):
    Phi = Phi_count[:]
    if np.isnan(Phi_count).any():
        print("Aha!!!")
    for j in range(len(Phi_count)):
        Phi[j] = (Phi_count[j]+beta) / (np.sum(Phi_count[j]) + beta*Phi.shape[1])
    return Phi


def str_to_datetime(datetime_str):
    # '%Y/%m/%d %H:%M'
    datetime_object = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M')
    return datetime_object


def get_sg_column(df, sg_lookup_dict, hosp_attribute="hospcode", age_attribute="agegroup"):
    # Create a search group column based on the HOSP code col and age group.
    # Mapping is provided by the lookup dictionary.
    df['search_group'] = -1
    def f(x):
        x_np = x.to_numpy()
        return sg_lookup_dict[x_np[0], x_np[1]]
    df['search_group'] = df[[hosp_attribute, age_attribute]].apply(lambda x: f(x), axis=1)
    return df


@njit
def get_caseline_in_sg(sg_h_nparray, sg, caseline_indices):
    count = 0
    for i in range(len(sg_h_nparray)):
        if sg_h_nparray[i] in sg:
            caseline_indices = np.append(caseline_indices, i)
            count += 1
    return caseline_indices, count

def remove_tense(string):
    nulist = string.split(' ')
    ulist = []
    for word in nulist:
        ulist.append(WordNetLemmatizer().lemmatize(word))
    ustring = ' '.join(ulist)
    return ustring

def get_word_dist(caseline_df, word_dict, Phi, t, cluster_word_num, topic_weight=True):
    word_dist_df = pd.DataFrame(columns=['word', 'weight'])
    for cc in caseline_df:
        word_list = cc
        #added to permit processing of word lists that are separated by ' ' or '_'
        if "_" in word_list:
            word_list = cc.split("_")
        else:
            word_list = cc.split()
        for word in word_list:
            word = remove_tense(word)
            if topic_weight:
                if word not in word_dict.keys():
                    continue
                word_idx = word_dict[word]
                weight = Phi[t][word_idx]
            else:
                weight = 1.0
            # If word already exists, add to the value; else add a row with that word
            if word in word_dist_df['word'].values:
                row_idx = word_dist_df['word'].loc[lambda x: x == word].index
                current_weight = word_dist_df['weight'].iloc[row_idx]
                word_dist_df.loc[row_idx, 'weight'] = weight + current_weight
            else:
                idx = len(word_dist_df)
                word_dist_df.loc[idx, 'weight'] = weight
                word_dist_df.loc[idx, 'word'] = word

    # Process weights
    word_dist_df = word_dist_df[word_dist_df['weight'] >= 0.01]  # Remove numerical noises
    word_dist_df = word_dist_df.sort_values(by='weight', ascending=False)  # Sort
    word_dist_df = word_dist_df.head(cluster_word_num)   # Only keep the top [cluster_word_num] words.
    word_dist_df['weight'] = word_dist_df['weight'] / word_dist_df['weight'].sum()  # Normalize

    # Convert df to string: word0_weight0_word1_weight1_...
    result = ""
    for i in range(len(word_dist_df)):
        if result!="":
            result += "_"
        result += word_dist_df['word'].iloc[i]+"_"+str(word_dist_df['weight'].iloc[i])
    return result


def get_words_from_word_dist(word_dist):
    if type(word_dist) == float:
        word_dist = str(word_dist)
    result = ""
    weights = []
    array = word_dist.split("_")
    for i in range(len(array)):
        if i%2==0:
            result += array[i]+" "
        else:
            weights.append(float(array[i]))
    return result[:-1], weights


# The function that concatenates consecutive age ranges together
def unify_agegroup(age_groups, concatenate_agegroup):
    result = ""
    age_groups = np.unique(age_groups)
    age_groups = np.sort(age_groups)
    if concatenate_agegroup:
        for i in range(age_groups.shape[0]):
            ag = age_groups[i]
            if ag[1]=="-":
                ag = "0"+ag
            if ag[-2]=="-":
                ag = ag[:-1]+"0"+ag[-1]
            age_groups[i] = ag
        age_groups = np.sort(age_groups)
        current_start = -1
        current_end = -1
        for i in range(len(age_groups)):
            ag = age_groups[i]
            if ag=="95+" or ag=="95":  # "95+"
                result += ", "
                result += ag
                continue

            start = int(ag[0:2])
            end = int(ag[3:])

            if current_start < 0:
                current_start = start
                current_end = end
            elif start==current_end+1 or start==current_end:
                current_end = end
            else:
                appended_string = f"{current_start:02d}-{current_end:02d}"
                if result!="":
                    result+=", "
                result += appended_string
                current_start = start
                current_end = end

            if i==len(age_groups)-1 or (i==len(age_groups)-2 and (age_groups[-1]=="95+" or age_groups[-1]=="95")):
                appended_string = f"{current_start:02d}-{current_end:02d}"
                if result!="":
                    result+=", "
                result += appended_string

        result_lst = result.split(", ")
        result_lst.sort()
        result = ""
        for i in range(len(result_lst)):
            if i==len(result_lst)-1:
                result += result_lst[i]
            else:
                result += result_lst[i]+", "
    else:
        result = ""
        for i in range(len(age_groups)):
            if i==len(age_groups)-1:
                result += age_groups[i]
            else:
                result += age_groups[i]+", "
    return result


@njit
def assign_topic_from_Phi(X, Phi, alpha=0, iters=1):
    n_t = Phi.shape[0]
    if alpha==0:
        alpha = 1/Phi.shape[0]
    # This is the EM implementation from 2016 paper's Algorithm II.
    topic_assignments = np.zeros(X.shape[0])
    for d in range(X.shape[0]):
        Theta_doc = np.ones(n_t) / n_t
        for _ in range(iters):
            theta = np.zeros(n_t)
            for word in X[d]:
                if word[1]==-1:
                    break
                Phi_word  = Phi[:,word[0]]   # each word is (index, count)
                topic_dist = np.multiply(Phi_word, Theta_doc) * n_t
                topic_dist = topic_dist / np.sum(topic_dist)
                theta = theta+topic_dist
            theta = theta+alpha
            Theta_doc = theta / np.sum(theta)
        topic_assignments[d] = np.argmax(Theta_doc)
    return topic_assignments


def topic_strings_to_Phi_b(topic_strings, word_dict):
    monitored = []
    Phi_b = np.array([])
    for i in range(len(topic_strings)):
        Phi_caseline = np.zeros(len(word_dict))
        topic_str = topic_strings[i]
        if topic_str[0]=='1':
            monitored.append(i)

        # Get caseline distribution
        word_list = topic_str[2:].split("_")
        for j in range(int(len(word_list)/2)):
            word = word_list[2*j]
            weight = word_list[2*j+1]
            if word in word_dict.keys():
                word_idx = word_dict[word]
                Phi_caseline[word_idx] = weight

        # Append caseline distribution to Phi_b
        Phi_caseline = np.array([Phi_caseline])
        if Phi_b.size==0:
            Phi_b = Phi_caseline
        else:
            Phi_b = np.concatenate((Phi_b, Phi_caseline))

    return Phi_b, monitored


def view_topics(Phi, word_dict_file, display_words):
    for t in range(Phi.shape[0]):
        print(f"\nTop words in topic {t}:")
        topic_0_df = pd.DataFrame(columns=["word", "weight"])
        word_dict_df = pd.read_csv(word_dict_file)
        for w in range(Phi.shape[1]):
            index = word_dict_df.index[word_dict_df['index'] == w].tolist()[0]
            word = word_dict_df.loc[index, 'word']
            count = Phi[t][index]
            if count>0:
                topic_0_df.loc[len(topic_0_df.index)] = [word, count]
        topic_0_df = topic_0_df.sort_values(by=['weight'], ascending=False)
        print(topic_0_df[['word', 'weight']].iloc[:display_words])


# concatenate_agegroup is only set to true if the agegroup contents look like 11-15, 66-70, etc.
def process_clusters(cluster_dir, concatenate_agegroup, merged_cluster, cluster_type):
    #novel
    novel_raw_file = cluster_dir + "/novel_raw.csv"
    novel_caselines_file = cluster_dir + "/novel_caselines.txt"
    novel_cluster_summary_file = cluster_dir + "/novel_cluster_summary.txt"
    novel_topicwords_file = cluster_dir + "/novel_topicwords.txt"
    #monitored:
    monitored_raw_file = cluster_dir + "/monitored_raw.csv"
    monitored_caselines_file = cluster_dir + "/monitored_caselines.txt"
    monitored_cluster_summary_file = cluster_dir + "/monitored_cluster_summary.txt"
    monitored_topicwords_file = cluster_dir + "/monitored_topicwords.txt"
    #merged novel:
    merged_novel_raw_file = cluster_dir + "/merged_novel_raw.csv"
    merged_novel_caselines_file = cluster_dir + "/merged_novel_caselines.txt"
    merged_novel_cluster_summary_file = cluster_dir + "/merged_novel_cluster_summary.txt"
    merged_novel_topicwords_file = cluster_dir + "/merged_novel_topicwords.txt"
    #merged monitored:
    merged_monitored_raw_file = cluster_dir + "/merged_monitored_raw.csv"
    merged_monitored_caselines_file = cluster_dir + "/merged_monitored_caselines.txt"
    merged_monitored_cluster_summary_file = cluster_dir + "/merged_monitored_cluster_summary.txt"
    merged_monitored_topicwords_file = cluster_dir + "/merged_monitored_topicwords.txt"

    if(merged_cluster==False):
        _process_cluster(novel_raw_file, novel_caselines_file, novel_cluster_summary_file, novel_topicwords_file,
                     concatenate_agegroup)
        _process_cluster(monitored_raw_file, monitored_caselines_file, monitored_cluster_summary_file,
                         monitored_topicwords_file, concatenate_agegroup)
    elif(merged_cluster==True and cluster_type == "novel"):
        _process_cluster(merged_novel_raw_file, merged_novel_caselines_file, merged_novel_cluster_summary_file, 
                         merged_novel_topicwords_file, concatenate_agegroup)
    elif(merged_cluster==True and cluster_type == "monitored"):
        _process_cluster(merged_monitored_raw_file, merged_monitored_caselines_file, merged_monitored_cluster_summary_file, 
                         merged_monitored_topicwords_file, concatenate_agegroup)

    import_static_topics_file = cluster_dir+"/import_static_topics.csv"
    import_static_topics = pd.DataFrame(columns=["topic_dist"])
    import_static_topics.to_csv(import_static_topics_file, sep='\t', header=False, index=False)


def _process_cluster(raw_file, caselines_file, cluster_summary_file, topicwords_file, concatenate_agegroup):
    raw_caseline = pd.read_csv(raw_file, header=[0])
    if "VISITID" not in raw_caseline.columns:
        raw_caseline["VISITID"] = "/"
    # raw_caseline["VISITID"] = raw_caseline["VISITID"].fillna("/")  # removed DBN 9-16-2022

    # Initiate the columns of the output files
    caselines = raw_caseline[["index", "cc", "icd", "VISITID", "time", "date", "sex", "agegroup", "hospcode"]]
    cluster_summary = pd.DataFrame(columns=["index", "start_time", "end_time", "hospcode", "score", "top_words",
                                            "agegroup"])
    caseline_words = pd.DataFrame(columns=["topic", "weight", "word"])

    # Iterate over raw caseline that are detected in clusters and write the files for GUI
    current_index = -1
    start_datetime_str = end_datetime_str = ""
    age_groups = np.array([])
    prev_row = pd.DataFrame(columns=raw_caseline.columns)
    word_dist_prev = ""
    for i, row in raw_caseline.iterrows():
        # Write in if at the end of the file
        if i == len(caselines) - 1:
            age_groups = np.append(age_groups, row['agegroup'])
            words, weights = get_words_from_word_dist(row['word_dist'])
            age_group = unify_agegroup(age_groups, concatenate_agegroup)
            cluster_summary.loc[len(cluster_summary.index)] = [current_index, start_datetime_str,
                                    end_datetime_str, prev_row["hospcode"], prev_row["score"], words, age_group]
            words_lst = words.split(" ")
            for j in range(len(words_lst)):
                caseline_words.loc[len(caseline_words)] = [current_index, weights[j], words_lst[j]]
            break

        # Otherwise, write in the previously indexed cluster if the index changes
        if row['index'] != current_index:
            if current_index >= 0:
                # words, weights = get_words_from_word_dist(row['word_dist'])
                words, weights = get_words_from_word_dist(word_dist_prev)
                age_group = unify_agegroup(age_groups, concatenate_agegroup)
                start_datetime_str = start_datetime.strftime('%Y-%m-%d %H:%M')
                end_datetime_str   = end_datetime.strftime('%Y-%m-%d %H:%M')
                cluster_summary.loc[len(cluster_summary.index)] = [current_index, start_datetime_str,
                                    end_datetime_str, prev_row["hospcode"], prev_row["score"], words, age_group]
                words_lst = words.split(" ")
                for j in range(len(words_lst)):
                    caseline_words.loc[len(caseline_words)] = [current_index, weights[j], words_lst[j]]
                age_groups = np.array([])
            prev_row = row
            current_index = row['index']
            word_dist_prev = row['word_dist']
            start_datetime_str = end_datetime_str = row['date'].replace('/', '-') + " " + row['time']
        age_groups = np.append(age_groups, row['agegroup'])
        date_time = pd.to_datetime(row['date'] + ' ' + row['time'], format='%Y/%m/%d %H:%M')
        start_datetime = pd.to_datetime(start_datetime_str, format='%Y-%m-%d %H:%M')
        end_datetime = pd.to_datetime(end_datetime_str, format='%Y-%m-%d %H:%M')
        if start_datetime > date_time:
            start_datetime = date_time
            start_datetime_str = start_datetime.strftime('%Y-%m-%d %H:%M')
        if end_datetime < date_time:
            end_datetime = date_time
            end_datetime_str = end_datetime.strftime('%Y-%m-%d %H:%M')

    caseline_words.to_csv(topicwords_file, sep='\t', header=False, index=False)
    caselines.to_csv(caselines_file, sep='\t', header=False, index=False)
    cluster_summary.to_csv(cluster_summary_file, sep='\t', header=False, index=False)

def case_overlap(working_i,working_j):
    visithash = set()
    resthash = set()
    for _, row in working_i.iterrows():
        if (row["VISITID"] != '' and not np.isnan(row["VISITID"])):
            visithash.add(row["VISITID"])
        resthash.add(row["cc"]+"__"+pd.to_datetime(row["date"] + ' ' + row["time"], format='%Y/%m/%d %H:%M').strftime('%Y-%m-%d %H:%M')+"__"+row["agegroup"])
    for _, row in working_j.iterrows():
        if (row["VISITID"] != '' and not np.isnan(row["VISITID"]) and row["VISITID"] in visithash):
            return True
        if (row["cc"]+"__"+pd.to_datetime(row["date"] + ' ' + row["time"], format='%Y/%m/%d %H:%M').strftime('%Y-%m-%d %H:%M')+"__"+row["agegroup"] in resthash):
            return True
    return False

def time_gap(working_i,working_j):
    date_time_i = pd.to_datetime(working_i['date'] + ' ' + working_i['time'], format='%Y/%m/%d %H:%M')
    date_time_j = pd.to_datetime(working_j['date'] + ' ' + working_j['time'], format='%Y/%m/%d %H:%M')
    diff1 = (date_time_i.min()-date_time_j.max()).total_seconds()/3600
    diff2 = (date_time_j.min()-date_time_i.max()).total_seconds()/3600
    return np.max([diff1,diff2,0])

def topic_overlap(working_i,working_j):
    if working_i.empty or working_j.empty:
        return 0
    weighthash = {}
    totalweight = 0
    words_i, weights_i = get_words_from_word_dist(working_i.iloc[0,:].loc['word_dist'])
    words_i = words_i.split(" ")
    for loc in range(len(words_i)):
       weighthash[words_i[loc]] = weights_i[loc]
    words_j, weights_j = get_words_from_word_dist(working_j.iloc[0,:].loc['word_dist'])
    words_j = words_j.split(" ")
    for loc in range(len(words_j)):
        if words_j[loc] in weighthash:
            totalweight += np.min([weighthash[words_j[loc]], weights_j[loc]])
    return totalweight

def merge_clusters(raw_file, cluster_dir, cluster_type, merge_on_duplicate_case_only, merge_window, similarity_threshold):

    from scipy.sparse.csgraph import connected_components

    if cluster_type=="monitored" or cluster_type=="Monitored" or cluster_type=="MONITORED":
        merged_caseline_file = cluster_dir + "/merged_monitored_raw.csv"
    elif cluster_type=="novel" or cluster_type=="Novel" or cluster_type=="NOVEL":
        merged_caseline_file = cluster_dir + "/merged_novel_raw.csv"

    working_caseline = pd.read_csv(raw_file, header=[0])

    if (working_caseline.empty):
        working_caseline.to_csv(merged_caseline_file)
        return

    maxIndex = working_caseline["index"].iloc[-1]
    merge_edges = np.zeros(shape=(maxIndex+1,maxIndex+1))
    for i in range(maxIndex):
        for j in range(i+1,maxIndex+1):
            working_i = working_caseline[working_caseline["index"]==i]
            working_j = working_caseline[working_caseline["index"]==j]
            if (time_gap(working_i,working_j) <= merge_window) and (not merge_on_duplicate_case_only or case_overlap(working_i,working_j)) and (topic_overlap(working_i,working_j) >= similarity_threshold):
                merge_edges[i,j] = 1
    cc = connected_components(merge_edges)
    num_cc = cc[0]
    cc = cc[1]
    new_caseline = pd.DataFrame(columns=working_caseline.columns)
    for thecomponent in range(num_cc):
        indices = np.asarray(cc == thecomponent).nonzero()[0].tolist()
        rows = working_caseline[working_caseline["index"].apply(lambda x: x in indices)].copy()
        rows["index"] = thecomponent
        rows["topic"] = rows.loc[rows["score"].idxmax(),"topic"]
        rows["score"] = rows["score"].max()
        del rows["word_dist"]
        rows.drop_duplicates(inplace=True)
        rows["word_dist"] = get_word_dist(rows["cc_processed"], None, None, None, None, topic_weight=False)
        new_caseline = pd.concat([new_caseline,rows],ignore_index=True)
    new_caseline.to_csv(merged_caseline_file,sep=',', header=True, index=False)
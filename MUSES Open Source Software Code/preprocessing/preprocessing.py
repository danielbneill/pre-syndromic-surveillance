""" Boyuan Chen
Data processing file adapted from Mallory's code in C. When run with the full functionality (clean_data),
this file generates the
1) processed data file that has invalid rows deleted and a column of processed chief complaint
2) a search group (sg) file
3) a lookup dictionary from sg combination to sg index
4) a word-to-word-index dictionary.
4 will be used in topic model training, while 2 and 3 will be used in scanning.

Below are the explanations for some parameters that might be hard to understand:
functionality: "clean_data" does the full work, giving you 1,2,3,4.
               "get_search_group" only gives you 2 and 3.
input_file: the data file's directory. ie ../data/ED_2016_only.txt
all_search_groups: ie preprocessed_search_groups_file.csv
dict_dir: The dictionary dir where .npy files are stored.

This is the processing of chief complaint text:
1) Remove all the rows with invalid time and date.
2) We generate the search group files from the remaining rows.
3) Remove punctuations in chief complaint
4) i. Convert icd code to text and concatenate to the original text
   ii. Correct misspelled words
   iii. Remove trash words
   iv. Remove word tense (Done with nltk wordnet). The dictionary is downloaded at the beginning of the run
   v. Remove rows with empty chief complaints; link the remaining words with '_'

Finally, the coding format for all output files is utf-8.
"""

import pickle
import pandas as pd
import numpy as np
from flashtext import KeywordProcessor
from datetime import datetime
import argparse
from collections import defaultdict
from itertools import product
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.wordnet import WordNetLemmatizer

parser = argparse.ArgumentParser(description='Preprocessing args parser')
parser.add_argument('--functionality', type=str, default="clean_data")

# Input files
parser.add_argument('--input_file', type=str, default="../data/12-18_raw.csv")
parser.add_argument('--sep', type=str, default=',')
parser.add_argument('--output_all_search_groups', type=str, default="../data/search_group_12-18.csv")
parser.add_argument('--search_group_file_names', type=str,
                    default="./Settings/SearchGroupFiles/hospcode_searchgroups.csv "
                            "./Settings/SearchGroupFiles/age_search_groups_rolling.csv")
parser.add_argument('--icd_map_file', type=str, default="./dict/icd10_to_text_list_concat.txt")
parser.add_argument('--word_index_load_file', type=str, default="")
parser.add_argument('--correcting_misspell', type=str, default="./dicts/correct_common_mistakes_list.txt")
parser.add_argument('--remove_word_list', type=str, default="./dicts/remove_word_list.txt")

# Output files
parser.add_argument('--output_file_name', type=str, default="../data/processed_12-18.csv")
parser.add_argument('--output_sg_dict', type=str, default="../data/lookup_dict_12-18.pickle")
parser.add_argument('--output_word_index_dict_file', type=str, default="")

# Time
parser.add_argument('--start_date', type=str, default="01/01/2016 00:00", help="Out of range values are acceptable")
parser.add_argument('--end_date', type=str, default="12/31/2016 23:59", help="Out of range values are acceptable")

# Processing paratemers
parser.add_argument('--chunksize', type=str, default=10000)

# Attribute Names
parser.add_argument('--attributes_in_sequence', type=str, default="")
parser.add_argument('--search_group_attributes', type=str, default="hospcode agegroup")

FLAGS = parser.parse_args()


def clean_data(input_file, search_group_attributes, search_group_file_names, start_date, end_date,
               output_file_name, output_all_search_groups, chunksize, sep, output_sg_dict, icd_map_file,
               word_index_dict_file, word_index_load_file, correcting_misspell, remove_word_list,
               attributes_in_sequence):
    print("Cleaning the data that will be scanned...")

    # Sort out attributes. We will use "attributes_array" to represent all attributes of the raw data, in sequence.
    attributes_default = "cc icd VISITID time date sex agegroup hospcode"
    if len(attributes_in_sequence) == 0:
        attributes_in_sequence = attributes_default
    attributes_array = attributes_in_sequence.split(" ")
    default_array = attributes_default.split(" ")
    attributes_mandatory = ["cc", "time", "date", "agegroup", "hospcode"]
    for att_mandatory in attributes_mandatory:
        if att_mandatory not in attributes_array:
            print(f"Attributes error. Must include {att_mandatory}.")
            raise Exception("Mandatory attribute does not exist.")
    for att in attributes_array:
        if (att not in default_array) & (att != "x"):  # added the "x" option to ignore an attribute DBN 9-16-2022
            print(f"{att} is not a valid attribute. Please choose attributes from: {attributes_default}")
            raise Exception("Attribute invalid.")

    # Load Files for Correcting Text
    correct_common_mistakes_list = pd.read_csv(correcting_misspell, sep='\t')
    correct_common_mistakes_dictionary = {}
    for i in range(len(correct_common_mistakes_list)):
        wrong_word = correct_common_mistakes_list.loc[i, 'with_error']
        correct_word = correct_common_mistakes_list.loc[i, 'corrected']
        correct_common_mistakes_dictionary[wrong_word] = correct_word
    _remove_list = pd.read_csv(remove_word_list, sep='\t', header=None, names=['words']).words.tolist()
    remove_list = []
    for x in _remove_list:
        if x not in remove_list:
            remove_list.append(x)
    icd_map_list = pd.read_csv(icd_map_file, sep='\t')
    icd_att = icd_map_list.columns[0]
    icd_map_dict = icd_map_list.set_index(icd_att).T.to_dict(orient='index')['description']

    icd_processor = KeywordProcessor()
    correct_processor = KeywordProcessor()
    remove_processor = KeywordProcessor()
    for k in remove_list:
        remove_processor.add_keyword(k, ' ')
    for k, v in correct_common_mistakes_dictionary.items():
        correct_processor.add_keyword(k, v)
    for k, v in icd_map_dict.items():
        icd_processor.add_keyword(str(k), v)

    # Initialize word-index dictionary dataframe. If we want to append on an existing dictionary, initialize
    # from that file's value.
    seen_words = remove_list
    dict_df = pd.DataFrame([], columns=['word'])
    if word_index_load_file!="":
        prev_dict_df = pd.read_csv(word_index_load_file, sep=',', header=[0])
        prev_words = prev_dict_df['word'].values.tolist()
        seen_words = seen_words + prev_words
        dict_df = prev_dict_df
        dict_df.drop(columns=['index'])

    # Creaing dictionaries needed for the location_id column
    # search_group_file_names: ['hospcode_searchgroups.csv', 'age_search_groups_rolling.csv']
    # search_group_attributes:  ['hospcode', 'agegroup']
    # lookup_combinations_dict: {('HOSP46', '40-44'): 0, ('HOSP46', '30-34'): 1, ('HOSP46', '70-74'): 2, ...}
    lookup_combinations_dict, unique_attribute_keys_dict = \
        make_location_search_groups_dict(search_group_file_names, search_group_attributes)
    all_unique_lookup_combinations_in_data = []

    # Initialize output files and columns. The output attributes are constant. If some attributes do not exist
    # in the raw data, then the content of that column will be empty.
    cleaned_chunks = 0
    output_header_names = pd.DataFrame([['cc', 'cc_processed', 'icd', 'VISITID', 'date', 'time', 'sex', 'hospcode',
                                         'agegroup']])
    output_header_names.to_csv(output_file_name, index=False, header=False)

    # Convert start and end times to this datetime format: 2015-12-03 00:01:00
    start_date = datetime.strptime(start_date, '%m/%d/%Y %H:%M')
    end_date = datetime.strptime(end_date, '%m/%d/%Y %H:%M')

    # Load data (500k per chunk)
    chunksize = int(chunksize)
    reader = pd.read_csv(input_file, sep=sep, iterator=True, chunksize=chunksize, encoding='latin1')  # low_memory=False
    for cc_data in reader:
        # Rename all the attributes to default
        if len(cc_data.columns) != len(attributes_array):
            print("Attribute sequence length incorrect. Please check again!")
            raise Exception("Attribute sequence length incorrect.")
        print(f"Starting chunk {cleaned_chunks+1}")
        # Make sure the attributes are named consistently
        for col in range(len(attributes_array)):
            original_attribute = cc_data.columns[col]
            new_attribute = attributes_array[col]
            cc_data.rename(columns={original_attribute: new_attribute}, inplace=True)

        # Fill in empty columns
        if 'VISITID' not in attributes_array:
            cc_data['VISITID'] = ""
        if 'icd' not in attributes_array:
            cc_data['icd'] = ""
        if 'sex' not in attributes_array:
            cc_data['sex'] = ""

        # Drop the rows whose date and time format is not correct
        print("Cleaning Data Step 1 of 5 - filter by time and date")
        cc_data['time'] = cc_data['time'].str.replace('24:00', '23:59')
        cc_data['time'] = cc_data['time'].str.replace('24:01', '23:59')
        cc_data['time'] = cc_data['time'].str.replace('24:02', '23:59')
        cc_data['time'] = cc_data['time'].str.replace('24:03', '23:59')
        cc_data['time'] = cc_data['time'].str.replace('24:04', '23:59')
        cc_data['time'] = cc_data['time'].str.replace('24:05', '23:59')
        cc_data['time'] = cc_data['time'].str.replace('24:06', '23:59')
        cc_data['time'] = cc_data['time'].str.replace('24:07', '23:59')
        cc_data = cc_data.join(
            cc_data['date'].str.split('/', expand=True).rename(index=int, columns={0: 'month', 1: 'day', 2: 'year'}))
        cc_data = cc_data.join(
            cc_data['time'].str.split(':', expand=True).rename(index=int, columns={0: 'hour', 1: 'minute'}))
        # Drop the rows with invalid date and time
        cc_data.dropna(subset=['day', 'month', 'year', 'hour', 'minute'], how='any', inplace=True)

        # Reconstruct valid full date
        cc_data['year'] = cc_data['year'].astype(str).apply(lambda x: '20'+x if len(x) == 2 else x) # added DBN 9-16-2022
        cc_data['month'] = cc_data['month'].astype(str).str.zfill(2)
        cc_data['day']   = cc_data['day'].astype(str).str.zfill(2)
        cc_data['hour']  = cc_data['hour'].astype(str).str.zfill(2)
        cc_data['date']  = cc_data['year'] + '/' + cc_data['month'] + '/' + cc_data['day']

        cc_data['full_date'] = pd.to_datetime(
            cc_data['year']+'/'+cc_data['month']+'/'+cc_data['day']+' '+ cc_data['hour']+':'+cc_data[
                'minute'], format='%Y/%m/%d %H:%M')

        # Filter out the dates that aren't in the range
        cc_data = cc_data[(cc_data['full_date'] >= start_date) & (cc_data['full_date'] <= end_date)]
        if cc_data.empty:
            cleaned_chunks += 1
            continue


        # Filter By Other Attributes
        print("Cleaning Data Step 2 of 5 - spatial scan")
        hosp_attribute = search_group_attributes[0]
        age_attribute = search_group_attributes[1]
        cc_data[hosp_attribute] = cc_data[hosp_attribute].str.upper()
        for i in range(0, len(search_group_attributes)):
            cc_data = cc_data.loc[
                cc_data[search_group_attributes[i]].isin(unique_attribute_keys_dict[search_group_attributes[i]])]
            if cc_data.empty:
                continue

        # Create Location for Scan
        # The 'zipped_location_attributes' is 'hospcode' + 'agegroup'. That is, the "location" is not the geological
        # location/coordinates but a combination of the hopsital and the age group, which is in turned the "location"
        # used by the algorithm for the scan.
        cc_data['zipped_location_attributes'] = [tuple(cc_data.iloc[i][n] for n in search_group_attributes) for i in
                                                 range(0, len(cc_data))]

        # Then, using the dictionaries created earlier, the 'hospcode' + 'agegroup' location is mapped to a single number.
        # It is assumed that this number will be used by the spatial scanning model to compute the scan.
        cc_data['location_forscan'] = cc_data['zipped_location_attributes'].map(lookup_combinations_dict)
        all_unique_lookup_combinations_in_data.extend(cc_data['location_forscan'].unique().tolist())


        # Clean the Chief Complaint Text
        print("Cleaning Data Step 3 of 5 - remove punctuations")
        cc_data.loc[:, 'cc_clean'] = cc_data.loc[:, 'cc'].str.replace(',', ' ', regex=True)
        cc_data.loc[:, 'icd'].fillna('', inplace=True)   # Replace NaN with empty string
        cc_data['icd_copy'] = cc_data['icd']
        cc_data.loc[:, 'icd'] = cc_data.loc[:, 'icd'].str.replace('.', '', regex=False)
        cc_data.loc[:, 'icd'] = cc_data.loc[:, 'icd'].str.replace('|', ' ', regex=False)
        cc_data.loc[:, 'cc_clean'] = cc_data.loc[:, 'cc_clean'] + ' ' + cc_data.loc[:, 'icd']
        cc_data.loc[:, 'cc_clean'] = cc_data.loc[:, 'cc_clean'].str.replace('-|\\|@|&|//', ' ', regex=True)
        cc_data.loc[:, 'cc_clean'] = cc_data.loc[:, 'cc_clean'].str.replace(r'([A-Z]{2}| |^)/([A-Z]{2}| |$)',
                                                                            r'\1 \2', regex=True)
        cc_data.loc[:, 'cc_clean'] = cc_data.loc[:, 'cc_clean'].str.replace('.', '', regex=True)
        cc_data.loc[:, 'cc_clean'] = cc_data.loc[:, 'cc_clean'].str.replace('\t', '', regex=True)
        cc_data.loc[:, 'cc_clean'] = cc_data.loc[:, 'cc_clean'].str.replace('[^0-9a-zA-Z\s]+', '', regex=True)
        # "3MONTHS" -> "3 MONTHS"
        cc_data.loc[:, 'cc_clean'] = cc_data.loc[:, 'cc_clean'].str.replace(r'([0-9])([a-zA-Z])', r'\1 \2', regex=True)
        cc_data['cc_clean'] = cc_data['cc_clean'].str.encode('utf-8')    # b'' showed up!!
        cc_data['cc_clean'] = cc_data['cc_clean'].astype(str)
        first_str = cc_data['cc_clean'].iloc[0]
        if first_str[0]=='b' and first_str[1]=='\'' and first_str[-1]=='\'':
            cc_data['cc_clean'] = cc_data['cc_clean'].str[2:-1]  # Remove "b' '"
        cc_data['cc_clean'] = cc_data['cc_clean'].str.lower()


        # Remove unwanted words and replace common mistakes
        print("Cleaning Data Step 4 of 5 - icd and clean the strings")
        pd.set_option('expand_frame_repr', False)
        for index in range(len(cc_data.index)):
            if not isinstance(cc_data['cc_clean'].iloc[index], str):
                print(f"{index}-th row of symptom has format errors")

        # Most important step: translate icd code and clean the unwanted words / uninterpretable icd codes
        cc_data['cc_clean'] = cc_data['cc_clean'].apply(icd_processor.replace_keywords)
        cc_data['cc_clean'] = cc_data['cc_clean'].apply(correct_processor.replace_keywords)
        cc_data['cc_clean'] = cc_data['cc_clean'].apply(remove_processor.replace_keywords)
        cc_data['cc_clean'] = cc_data['cc_clean'].apply(remove_tense)
        cc_data['cc_clean'] = cc_data['cc_clean'].apply(clean_list)

        # Filter out the invalid rows and output
        cc_data.loc[cc_data.loc[:, 'cc_clean'] == "", 'cc_clean'] = None
        cc_data.dropna(subset=['location_forscan', 'cc_clean'], inplace=True)
        if cc_data.empty:
            cleaned_chunks += 1
            continue


        cc_output_data = cc_data.loc[:, ['cc', 'cc_clean', 'icd_copy', 'VISITID', 'date',
                                         'time', 'sex', 'hospcode', 'agegroup']]
        cc_output_data.to_csv(output_file_name, index=False, header=False, mode='a', encoding='utf-8')

        # Write word to index dict
        print("Cleaning Data Step 5 of 5 - writing word-to-index dict")
        if word_index_dict_file!="":
            for i in range(len(cc_data)):
                string = cc_data['cc_clean'].iloc[i]
                word_list = re.split('_', string)
                for word in word_list:
                    if word not in seen_words:
                        new_line = pd.DataFrame([[word]], columns=['word'])
                        dict_df = pd.concat([dict_df, new_line], ignore_index=True)
                        seen_words.append(word)

        cleaned_chunks += 1

    # Make SearchGroups File
    # output_all_search_groups: the output file name, ie "..\data\search_group_10k.csv"
    if output_all_search_groups != "" and output_sg_dict != "":
        make_scan_search_group_file(search_group_file_names, search_group_attributes, output_all_search_groups,
                                lookup_combinations_dict, output_sg_dict)

    # Sort and write the word-to-index dictionary
    if word_index_dict_file != "":
        dict_df = dict_df.sort_values('word', ascending=True)
        dict_df['index'] = np.arange(len(dict_df))
        dict_df.to_csv(word_index_dict_file, sep=',', index=False)



# --- Helper functions for text processing ---
def clean_list(l):
    # Remove repetitive, number, 1-char strings
    # Remove the beginning digits of a string ("13MONTHS" -> "MONTHS")
    # For this version, if an icd code is attached directly to a word/icd code, we just throw it as noise.
    nulist = l.split(' ')
    ulist = []
    for x in nulist:
        if len(x)<=1 or x[-1].isdigit():
            continue
        if (x in ulist) or has_numbers(x):
            continue
        ulist.append(x)
    ustring = '_'.join(ulist)
    return ustring

def remove_first_digits(x):
    # "13MONTHS" -> "MONTHS"
    result = ""
    for i in range(len(x)):
        if not x[i].isdigit():
            result = x[i:]
    return result

def has_numbers(string):
    return any(char.isdigit() for char in string)

def remove_tense(string):
    nulist = string.split(' ')
    ulist = []
    for word in nulist:
        ulist.append(WordNetLemmatizer().lemmatize(word))
    ustring = ' '.join(ulist)
    return ustring

def read_attribute_search_groups(filename):
    f = open(filename)
    all_l_list = []
    for l in f.readlines():
        l_list = l.strip().split(',')
        l_list = [ll.strip() for ll in l_list]
        all_l_list.append(l_list)
    return all_l_list



def get_search_group_and_dict(input_file, attribute_column_names, search_group_file_names,
                output_all_search_groups, chunksize, sep, lookup_dict_path):
    # Creaing dictionaries needed for the location_id column
    # search_group_file_names: ['hospcode_searchgroups.csv', 'age_search_groups_rolling.csv']
    # attribute_column_names:  ['hospcode', 'agegroup']
    # lookup_combinations_dict: {('HOSP46', '40-44'): 0, ('HOSP46', '30-34'): 1, ('HOSP46', '70-74'): 2, ...}
    lookup_combinations_dict, unique_attribute_keys_dict = \
        make_location_search_groups_dict(search_group_file_names, attribute_column_names)
    all_unique_lookup_combinations_in_data = []

    # Load data
    chunksize = int(chunksize)
    reader = pd.read_csv(input_file, sep=sep, iterator=True, chunksize=chunksize, encoding='latin1')  # low_memory=False
    for cc_data in reader:
        for i in range(0, len(attribute_column_names)):
            cc_data = cc_data.loc[
                cc_data[attribute_column_names[i]].isin(unique_attribute_keys_dict[attribute_column_names[i]])]
            if cc_data.empty:
                continue

        # Create Location for Scan
        # The 'zipped_location_attributes' is 'hospcode' + 'agegroup'. That is, the "location" is not the geological
        # location/coordinates but a combination of the hopsital and the age group, which is in turned the "location"
        # used by the algorithm for the scan.
        cc_data['zipped_location_attributes'] = [tuple(cc_data.iloc[i][n] for n in attribute_column_names) for i in
                                                 range(0, len(cc_data))]

        # Then, using the dictionaries created earlier, the 'hospcode' + 'agegroup' location is mapped to a single number.
        # It is assumed that this number will be used by the spatial scanning model to compute the scan.
        cc_data['location_forscan'] = cc_data['zipped_location_attributes'].map(lookup_combinations_dict)
        all_unique_lookup_combinations_in_data.extend(cc_data['location_forscan'].unique().tolist())

    # Make SearchGroups File
    # output_all_search_groups: the output file name, ie "..\data\search_group_10k.csv"
    make_scan_search_group_file(search_group_file_names, attribute_column_names, output_all_search_groups,
                                lookup_combinations_dict, lookup_dict_path)



# Search Group Processing
def make_location_search_groups_dict(attribute_search_group_file_names, attribute_column_names):
    unique_attribute_keys = defaultdict(list)
    for i in range(0, len(attribute_column_names)):
        attribute = attribute_column_names[i]
        values_array = read_attribute_search_groups(attribute_search_group_file_names[i])
        values_list = list(set(x for l in values_array for x in l))
        unique_attribute_keys[attribute] = [v for v in values_list if v != '']
    allNames = unique_attribute_keys.keys()
    combinations_list = list(product(*(unique_attribute_keys[Name] for Name in allNames)))
    lookup_combinations_dict = dict([combinations_list[i], i] for i in range(0, len(combinations_list)))
    return lookup_combinations_dict, unique_attribute_keys


# Process Search Group (sg) Files Used To Create Combined Search Group File for Scan
def make_scan_search_group_file(attribute_search_group_file_names, attribute_column_names,
                                output_search_group_file_name, lookup_combinations_dict,
                                lookup_dict_path):
    """
    attribute_search_group_file_names: [hospcode_searchgroups.csv, age_search_groups_rolling.csv]
        The former is just HOSP1-50
        The latter is a combination of all possible age groups
    attribute_column_names: ['hospcode', 'agegroup']
    output_search_group_file_name: ie "..\data\search_group_10k.csv"
    lookup_combinations_dict: {('HOSP15', '20-24'): 837, ...}. One hospital code and one age range
    """
    # attribute0 is the hospital HOSP1-50, from attribute 1 on, are the number trials
    # attribute1 is the age group
    attribute_search_group_keys = defaultdict(list)
    for i in range(0, len(attribute_search_group_file_names)):
        attribute_search_group_keys["attribute" + str(i)] = read_attribute_search_groups(
            attribute_search_group_file_names[i])
    allsgNames = sorted(attribute_search_group_keys)
    attribute_search_group_list = list(product(*(attribute_search_group_keys[Name] for Name in allsgNames)))
    # attribute_search_group_list: [(['HOSP00'], ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44',
    # '45-49', '50-54', '', '', '', '', '', '', '', '', '', '', '', '']), ...]

    # Now map attribute_search_group_list to a combination of numbers via lookup_combinations_dict
    all_sg_number_string_list = []
    for sg in attribute_search_group_list:
        sg_number_string = ""
        for c in lookup_combinations_dict.keys():
            add_c = 0
            for i in range(0, len(attribute_column_names)):
                if c[i] in sg[i]:
                    add_c += 1
            if add_c == len(attribute_column_names):
                sg_number_string = sg_number_string + str(lookup_combinations_dict[c]) + "-"
        if sg_number_string != "":
            all_sg_number_string_list.append(sg_number_string.strip('-'))

    all_sg_number_string_list = pd.DataFrame(all_sg_number_string_list)
    all_sg_number_string_list.columns = ['location_ids']
    all_sg_number_string_list.to_csv(output_search_group_file_name, index=False)
    with open(lookup_dict_path, 'wb') as handle:
        pickle.dump(lookup_combinations_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_unique_values(input_file, attribute, output_file):
    print("Finding the unique values for this attribute that are present in the data...")
    all_unique_values = []
    reader = pd.read_csv(input_file, sep='\t', iterator=True, chunksize=200000, usecols=[attribute])
    for cc_data in reader:
        all_unique_values.extend(cc_data[attribute].unique().tolist())
    all_unique_values = list(set(all_unique_values))
    all_unique_values.sort()
    df = pd.DataFrame(all_unique_values)
    df.to_csv(output_file, index=False, header=False)



def main(args):
    search_group_attributes = args.search_group_attributes.split()
    search_group_file_names = args.search_group_file_names.split()
    if args.functionality=="clean_data":
        print(args.icd_map_file)
        clean_data(args.input_file, search_group_attributes, search_group_file_names,
                   args.start_date, args.end_date, args.output_file_name,
                   args.output_all_search_groups, args.chunksize, args.sep, args.output_sg_dict,
                   args.icd_map_file, args.output_word_index_dict_file, args.word_index_load_file,
                   args.correcting_misspell, args.remove_word_list, args.attributes_in_sequence)
    elif args.functionality=="get_search_group":
        get_search_group_and_dict(args.input_file, search_group_attributes, search_group_file_names,
                args.output_all_search_groups, args.chunksize, args.sep, args.output_sg_dict,
                args.icd_map_file)


if __name__ == '__main__':
    main(FLAGS)

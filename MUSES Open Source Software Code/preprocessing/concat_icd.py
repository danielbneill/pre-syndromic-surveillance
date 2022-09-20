""" Concatenate icd-10 indices:
Some icd-10 entries are as follow:
"S72136	s nondisplaced apophyseal fracture of unspecified femur  sequela"
But in the data the code is merged to "S72136s". So we need to concatenate the keys with the first
character in the value.
(From experiential observation, there usually is no single 'S72136' without the single character. If
you find one, prove me wrong and we will fix the code)
"""

import re
import pandas as pd

def largest_sim(symp_list):
    if len(symp_list)==1:
        return symp_list[0]
    min_len = len(min(symp_list, key=len))
    i = 0
    while i<min_len:
        first = symp_list[0][i]
        for j in range(1,len(symp_list)):
            if not symp_list[j][i]==first:
                return symp_list[0][:i]
        i+=1
    return symp_list[0][:i]

# Expand
icd10_ori  = "./dicts/icd10_to_text_list.txt"
icd10_copy = "./dicts/icd10_to_text_list_concat_.txt"
with open(icd10_ori, "r") as f1:
    with open(icd10_copy, "w") as f2:
        this_icd = ""
        symp_list = []
        for line in f1:
            value_list = re.split(' |\t', line)
            icd = value_list[0]
            second_str = value_list[1]
            if len(symp_list)>0 and icd!=this_icd:  # Going into a new icd, write the saved
                sim = largest_sim(symp_list)
                f2.write(this_icd + "\t" + sim + "\n")
                symp_list = []
            this_icd = icd
            if len(second_str)==1:
                symptom = ' '.join(value_list[2:])
                symp_list.append(symptom)
                branch_line = value_list[0] + value_list[1] + '\t' + symptom
                f2.write(branch_line)
            else:
                f2.write(line)

# Check for uniqueness
icd_map_list = pd.read_csv(icd10_copy, sep='\t')
dup = icd_map_list.duplicated(subset=['icd10'])
print("The following codes have duplicates:")
for i in range(len(icd_map_list)):
    if dup.iloc[i]:
        print(i,icd_map_list['icd10'].iloc[i])

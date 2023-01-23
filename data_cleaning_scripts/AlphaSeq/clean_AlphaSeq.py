#!/usr/bin/env python

# conda activate /gpfs/projects/bgmp/shared/groups/2022/cfr/lanl/envs/lanl_env

import pandas as pd 
import matplotlib.pyplot as plt


alpha_seq_data = pd.read_csv('/projects/bgmp/shared/groups/2022/cfr/lanl/raw_data/alpha_seq/MITLL_AAlphaBio_Ab_Binding_dataset.csv')

print(alpha_seq_data.head())

# look at unique values in all columns
poi_unique = alpha_seq_data['POI'].unique()
seq_unique = alpha_seq_data['Sequence'].unique()
target_unique = alpha_seq_data['Target'].unique()
assay_unique = alpha_seq_data['Assay'].unique() #1,2
replicate_unique = alpha_seq_data['Replicate'].unique() #1,2,3
affinity_unique = alpha_seq_data['Pred_affinity'].unique()
hc_unique = alpha_seq_data['HC'].unique()
lc_unique = alpha_seq_data['LC'].unique()
cdrh1_unique = alpha_seq_data['CDRH1'].unique()
cdrh2_unique = alpha_seq_data['CDRH2'].unique()
cdrh3_unique = alpha_seq_data['CDRH3'].unique()
cdrl1_unique = alpha_seq_data['CDRL1'].unique()
cdrl2_unique = alpha_seq_data['CDRL2'].unique()
cdrl3_unique = alpha_seq_data['CDRL3'].unique()

print('hc', len(hc_unique)) # 48928
print('lc', len(lc_unique)) # 56043
print('cdrh1', len(cdrh1_unique)) # 7868
print('cdrh2', len(cdrh2_unique)) # 18150
print('cdrh3', len(cdrh3_unique)) # 6799
print('cdrl1', len(cdrl1_unique)) # 19656 
print('cdrl2', len(cdrl2_unique)) # 5526
print('cdrl3', len(cdrl3_unique)) # 11325


# check unique lengths of entries to see how long sequences are; if there are missing entries
def len_checker(data, col):
    """
    Takes in dataframe column, returns unique lengths of
    entries in a set.
    """
    len_set = set()
    for item in data[col]:
        length = len(item)
        len_set.add(length)
    return len_set

# poi len set: {8, 9, 10, 11, 12, 14}
print('poi:', len_checker(alpha_seq_data, 'POI'))
# seq len set: {240, 249, 246}
print('sequence:', len_checker(alpha_seq_data, 'Sequence'))
# target len set: {9, 10}
print('target:', len_checker(alpha_seq_data, 'Target'))
# doesn't work for assay, replicate, or pred_affinity because they're integers not in a list
# HC len set: {117, 118, 119}
print('hc:', len_checker(alpha_seq_data, 'HC'))
# LC len set: {113, 115, 108}
print('lc:', len_checker(alpha_seq_data, 'LC'))
# cdrh1 len set: {10}
print('cdrh1:', len_checker(alpha_seq_data, 'CDRH1'))
# cdrh2: {16}
print('cdrh2:', len_checker(alpha_seq_data, 'CDRH2'))
# cdrh3: {8, 9, 10}
print('cdrh3:', len_checker(alpha_seq_data, 'CDRH3'))
# cdrl1: {17, 11, 14}
print('cdrl1:', len_checker(alpha_seq_data, 'CDRL1'))
# cdrl2: {13, 7}
print('cdrl2:', len_checker(alpha_seq_data, 'CDRL2'))
# cdrl3: {9, 11}
print('cdrl3:', len_checker(alpha_seq_data, 'CDRL3'))

# check for missing values in all columns; found only in Pred_affinity
print(alpha_seq_data['POI'].isna().sum()) # 0
print(alpha_seq_data['Sequence'].isna().sum()) # 0
print(alpha_seq_data['Target'].isna().sum()) # 0
print(alpha_seq_data['Assay'].isna().sum()) # 0
print(alpha_seq_data['Replicate'].isna().sum()) # 0
print('pred affinity:', alpha_seq_data['Pred_affinity'].isna().sum()) # 907561
print(alpha_seq_data['HC'].isna().sum()) # 0
print(alpha_seq_data['LC'].isna().sum()) # 0
print(alpha_seq_data['CDRH1'].isna().sum()) # 0
print(alpha_seq_data['CDRH2'].isna().sum()) # 0
print(alpha_seq_data['CDRH3'].isna().sum()) # 0
print(alpha_seq_data['CDRL1'].isna().sum()) # 0
print(alpha_seq_data['CDRL2'].isna().sum()) # 0
print(alpha_seq_data['CDRL3'].isna().sum()) # 0

# confirmed that total missing entries for dataframe are only in Pred_affinity column
print('total na values:', alpha_seq_data.isnull().sum().sum())  # 907561
# 1259701 - 907561 = 352140 usable entries (352139 not counting header line)

na_free = alpha_seq_data.dropna()
print('no seq NA len:', len(na_free))  # 352139 as expected

# check for duplicate sequences by putting sequences in set
sequence_set = set()
for seq in na_free['Sequence']:
    sequence_set.add(seq)
print('seq set:', len(sequence_set)) # 87807 unique sequences


# create new column with average affinity for each replicate
na_free['Mean_Affinity'] = na_free['Pred_affinity'].groupby(na_free['POI']).transform('mean')

# drop replicates using sequences
dedup_alpha_seq = na_free.drop_duplicates(subset='Sequence', keep='first')
print('dedup len:', len(dedup_alpha_seq)) # 87807


# write out NA-free averaged replicate data to clean csv
dedup_alpha_seq.to_csv('./clean_avg_alpha_seq.csv', index=False) 


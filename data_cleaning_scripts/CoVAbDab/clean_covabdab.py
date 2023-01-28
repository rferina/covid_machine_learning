#!/usr/bin/env python
"""
Data cleaner & splitter for CoV-AbDab dataset.
Author: Kaetlyn Gibson
"""
import os
import random
import pandas as pd

def dataLoader(CSV: str):
    """
    Given the CoV-AbDab dataset .csv, loads in the
    data into a pandas dataframe and returns the
    dataframe.
    """
    # == Load in data, only columns that we want ==
    # load data into pandas dataframe
    df = pd.read_csv(CSV)
    # acquire only columns that we want from the data
    df = df[['Name', 'Ab or Nb', 'Binds to','Protein + Epitope', 'Origin', 'VHorVHH', 'VL',
             'Heavy V Gene', 'Heavy J Gene', 'Light V Gene', 'Light J Gene', 'CDRH3',
             'CDRL3', 'Structures', 'ABB Homology Model (if no structure)']]
    print(f"[PRECLEANING]")
    print(f"\t> Number of entries before cleaning: {len(df)}")
    print(f"\t> done")
    return df


def dataCleaner(df, STANDARD_BT: str, TYPE: str):
    """
    Given the CoV-AbDab dataframe, 'Binds to' column standardizations,
    and type of 'Labels' column, returns the cleaned dataframe.
    'Binds to' column standardizations were made by printing out the
    unique values of the 'Binds to' column, and then manually fixing these values.
    """
    # == Isolate SARS-CoV1, SARS-CoV2, MERS-CoV, entries from 'Binds to' column ==
    print(f"[CLEANING 'Binds to']")
    binds_to = df['Binds to'].unique() # print this out so see unique values in column
    keep_cov, cov_var = [], []
    for entry in binds_to:
        # convert to upper in case of typos
        if 'SARS' in str(entry).upper():
            keep_cov.append(entry)
        if 'MERS' in str(entry).upper():
            keep_cov.append(entry)
        # isolating the list of unique variants
        split_entry = str(entry).split(';')
        for i in split_entry:
            if 'SARS' in str(i).upper():
                if i not in cov_var:
                    cov_var.append(i)
            elif 'MERS' in str(i).upper():
                if i not in cov_var:
                    cov_var.append(i)
    # filter out unwanted entries from original dataset 
    df = df[df['Binds to'].isin(keep_cov)]

    # == Further clean up of 'Binds to' column (standardizing names) ==
    # loaded the cov_var list into a sheets file, then fixed the typos manually for download as .tsv
    # load .tsv into a dictionary
    standard_bindsto_dict = {}
    with open(STANDARD_BT, 'r') as sf:
        for line in sf:
            if line != 'ORIGINAL\tSTANDARDIZED\n':
                line = line.strip().split('\t')
                # some were combined, such as 'SARS-CoV2_BetaSARS-CoV2_Gamma'
                if ',' in line[1]:
                    standard_bindsto_dict[line[0]] = line[1].split(',')
                else:
                    standard_bindsto_dict[line[0]] = line[1]
    # remove any non MERS or non SARS from entries in 'Binds to' column & standardize the data
    for entry in binds_to:
        sarsmers = []
        split_entry = str(entry).split(';')
        for n in range(len(split_entry)):
            if split_entry[n] in standard_bindsto_dict:
                # don't add variants 2x
                if split_entry[n] in ['SARS-CoV2_Omicron-BA4/5', 'SARS-CoV2_Omicron-BA4/BA5', \
                                      'SARS-CoV2_BetaSARS-CoV2_Gamma', 'SARS-CoV2_WTSARS-CoV1']:
                    for i in standard_bindsto_dict[split_entry[n]]:
                        if i not in sarsmers:
                            sarsmers.append(i)
                elif standard_bindsto_dict[split_entry[n]] not in sarsmers:
                    sarsmers.append(standard_bindsto_dict[split_entry[n]])
        sarsmers = ';'.join(str(i) for i in sarsmers)
        # replace old 'Binds to' unique entries with standardized unique entries
        df = df.assign(**{'Binds to':df['Binds to'].replace(entry, sarsmers)})

    # == Further clean up of 'Binds to' column (split weak binding into another column) ==
    # duplicate 'Binds to' column for isolation of weak binding
    weak_df, unweak_df, merscov_df, sarscov1_df, sarscov2_df, label_df =  df['Binds to'], df['Binds to'], df['Binds to'], df['Binds to'], df['Binds to'], df['Binds to']
    for entry in weak_df.unique():
        weak_li, unweak_li = [], []
        merscov, sarscov1, sarscov2 = 0, 0, 0
        split_entry = str(entry).split(';')
        for i in split_entry:
            # checking if weak binding or not
            if 'weak' in i:
                weak_li.append(i)
            else:
                unweak_li.append(i)
            # checking if any 'MERS-CoV'/'SARS-CoV1'/'SARS-CoV2' present
            if 'MERS-CoV' in i:
                merscov = 1
            if 'SARS-CoV1' in i:
                sarscov1 = 1
            if 'SARS-CoV2' in i:
                sarscov2 = 1
        weak = ';'.join(str(i) for i in weak_li)
        unweak = ';'.join(str(i) for i in unweak_li)
        # replace the values based on weak/unweak
        weak_df = weak_df.replace(entry, weak)
        unweak_df = unweak_df.replace(entry, unweak)
        merscov_df = merscov_df.replace(entry, merscov)
        sarscov1_df = sarscov1_df.replace(entry, sarscov1)
        sarscov2_df = sarscov2_df.replace(entry, sarscov2)
        # label creation
        if TYPE == 'binary':
            label_dict = { '010':0, # SARS-CoV1
                           '001':1, # SARS-Cov2
                           '100':0, # MERS-CoV
                           '011':1, # SARS-Cov1_SARS-CoV2
                           '110':0, # SARS-Cov1_MERS_CoV
                           '101':1, # SARS-Cov2_MERS_CoV
                           '111':1} # SARS-Cov1_SARS-CoV2_MERS_CoV (SARS FAMILY)
        elif TYPE == 'trinary':
            label_dict = { '010':1, # SARS-CoV1
                           '001':2, # SARS-Cov2
                           '100':1, # MERS-CoV
                           '011':3, # SARS-Cov1_SARS-CoV2
                           '110':1, # SARS-Cov1_MERS_CoV
                           '101':3, # SARS-Cov2_MERS_CoV
                           '111':3} # SARS-Cov1_SARS-CoV2_MERS_CoV (SARS FAMILY)
        else: #regular
            label_dict = { '010':1, # SARS-CoV1
                           '001':2, # SARS-Cov2
                           '100':3, # MERS-CoV
                           '011':4, # SARS-Cov1_SARS-CoV2
                           '110':5, # SARS-Cov1_MERS_CoV
                           '101':6, # SARS-Cov2_MERS_CoV
                           '111':7} # SARS-Cov1_SARS-CoV2_MERS_CoV (SARS FAMILY) 
        label = str(merscov) + str(sarscov1) + str(sarscov2)
        if label in label_dict:
            label_df = label_df.replace(entry, label_dict[label])
        else: # it should not do this! 
            break
    # apply new columns to original dataframe, df
    df = df.assign(**{'Binds to':unweak_df})
    df.insert(3, 'Binds to (weak)', weak_df)
    df.insert(4, 'Labels', label_df)
    print(f"\t> done")

    # == Isolate non human origin entries from 'Origin'column ==
    print(f"[CLEANING 'Origin']")
    origin = df['Origin'].unique()
    human = ['BA.1 convalescents', 'BA.2 convalescents', 'BA.5 convalescents', 
            'WT convalescents', 'WT vaccinees', 'SARS convalescents', 'Human']
    mouse = []
    # find which are human, mouse in origin
    for entry in origin:
        if 'b-cell' in str(entry).lower():
            if 'mouse' not in str(entry).lower():
                human.append(entry)
        if 'mouse' in str(entry).lower():
            if 'phage display' not in str(entry).lower():
                if 'chimeric' not in str(entry).lower():
                    mouse.append(entry)
        if 'mice' in str(entry).lower():
            mouse.append(entry)
    # this is the rest of the entries that are not included
    misc = [x for x in origin if x not in human+mouse]  
    # filter out non human and non mouse origin entries from original dataset 
    df = df[df['Origin'].isin(human+mouse)]
    # relabel human entries as 'Human' and mouse entries as 'Mouse'
    df['Origin'] = df['Origin'].replace(human, 'Human')
    df['Origin'] = df['Origin'].replace(mouse, 'Mouse')
    print(f"\t> done")

    # == Filter out data with no/incomplete sequences ('ND'/NaN/'') ==
    # & make sure no empty spaces at end of sequences
    print(f"[CLEANING 'VHorVHH','VL','CDRH3','CDRL3']")
    seq_dict = {'VHorVHH':[], 'VL':[], 'CDRH3':[], 'CDRL3':[]}
    for i in ['VHorVHH','VL','CDRH3','CDRL3']:
        df.drop(df[df[i] == 'ND'].index, inplace=True)
        df[i] = df[i].replace(r'^s*$', float('NaN'), regex = True)
        for j in df[i].unique():
            if ' ' in str(j):
                seq_dict[i].append(j)
        for j in seq_dict[i]:
            df[i] = df[i].replace(j, j.strip())
    df.dropna(subset = ['VHorVHH', 'VL', 'CDRH3', 'CDRL3'], inplace = True)
    # deduplicated dataframe
    df = df.drop_duplicates(subset=['VHorVHH','VL','CDRH3','CDRL3'], keep='first')
    df = df.drop_duplicates(subset=['VHorVHH','VL'], keep='first')
    df = df.drop_duplicates(subset=['CDRH3','CDRL3'], keep='first')
    print(f"\t> done")

    # == Clean Protein + Epitope column ==
    print(f"[CLEANING 'Protein + Epitope']")
    df.dropna(subset = ['Protein + Epitope'], inplace=True)
    df['Protein + Epitope'] = df['Protein + Epitope'].replace('S: NTD', 'S; NTD')
    df['Protein + Epitope'] = df['Protein + Epitope'].replace('S: RBD', 'S; RBD')
    pro_epi = df['Protein + Epitope'].unique()
    print(f"\t> done")

    # == Reset the indexing of the pandas dataframe since we removed entries ==
    df.reset_index(drop=True, inplace=True)
    return df


def dataSplit (df):
    """
    Given a dataframe, randomly subsamples the SARS-CoV2 data
    in a 2:1 ratio. Keeps the entirety of the non SARS-CoV2 data.

    Note!
    While this successfully splits the data, the results of the
    classification machine learning model suggests that alternative
    methods should be explored to mitigate 'Labels' imbalance.    
    """
    # == Separate indexes of dataset ==
    # loop through indexes + 'Labels' column
    # append the index based on the column values
    print(f"[SUBSETTING DATA DOWN]")
    set_dict = {}
    for index, label in zip(df.index, df['Labels']):
        key = str(label)
        if key not in set_dict:
            set_dict[key] = [index]
        else:
            set_dict[key].append(index)

    # == Randomly subsample dataset into new dataframe ==
    subset = []
    # add non sars-cov2 entries 
    subset += set_dict['0']
    # add downsampled sars-cov2 entries
    subset += random.sample(set_dict['1'], len(set_dict['0'])*2) # 2:1 ratio
    # sort indexes to be in ascending order
    subset.sort()
    # create new dataframe based on the indexes using .iloc
    subset_df = df.iloc[subset]
    print(f"\t> done")
    return subset_df


def dataStats (df, TYPE: str) -> None:
    """
    Given a dataframe and the type of 'Labels' column,
    prints out the distribution of SARS-CoV2 and non SARS-CoV2 
    entries between human and mouse.
    """
    # == Count ==
    set_dict = {}
    for index, label, species in zip(df.index, df['Labels'], df['Origin']):
        key = species+'_'+str(label)
        if key not in set_dict:
            set_dict[key] = 1
        else:
            set_dict[key] += 1

    # == Label ==
    if TYPE == 'binary':
        label_dict = {'Human_0':'non SARS-CoV2 entries', 'Human_1':'SARS-CoV2 entries',
                      'Mouse_0':'non SARS-CoV2 entries', 'Mouse_1':'SARS-CoV2 entries'}
    elif TYPE == 'trinary':
        label_dict = {'Human_1':'non SARS-CoV2 entries', 'Mouse_1':'non SARS-CoV2 entries',
                      'Human_2':'SARS-CoV2 entries', 'Mouse_2':'SARS-CoV2 entries',
                      'Human_3':'SARS-CoV2 cross-reaction entries', 'Mouse_3':'SARS-CoV2 cross-reaction entries'}
    else: # regular
        label_dict = {'Human_1':'SARS-CoV1 entries', 'Mouse_1':'SARS-CoV1 entries',
                      'Human_2':'SARS-CoV2 entries', 'Mouse_2':'SARS-CoV2 entries',
                      'Human_3':'MERS-CoV entries', 'Mouse_3':'MERS-CoV entries',
                      'Human_4':'SARS-Cov1, SARS-CoV2 entries', 'Mouse_4':'SARS-Cov1, SARS-CoV2 entries',
                      'Human_5':'SARS-Cov1, MERS-CoV entries', 'Mouse_5':'SARS-Cov1, MERS-CoV entries',
                      'Human_6':'SARS-Cov2, MERS-CoV entries', 'Mouse_6':'SARS-Cov2, MERS-CoV entries',
                      'Human_7':'SARS-Cov1, SARS-CoV2, MERS-CoV entries', 'Mouse_7':'SARS-Cov1, SARS-CoV2, MERS-CoV entries'}
    
    # == Print ==
    print(f"[**FINAL STATS**]")
    print(f"\t> Number of total entries: {len(df)}")
    print(f"\t  > Human")
    for i in label_dict:
        if 'Human' in i:
            if i in set_dict:
                print(f"\t\t > Number of {label_dict[i]}: {set_dict[i]}")
            else: 
                print(f"\t\t > Number of {label_dict[i]}: 0")
    print(f"\t  > Mouse")
    for i in label_dict:
        if 'Mouse' in i:
            if i in set_dict:
                print(f"\t\t > Number of {label_dict[i]}: {set_dict[i]}")
            else: 
                print(f"\t\t > Number of {label_dict[i]}: 0")
    return None


if __name__=='__main__':
    # == File pathing ==
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    ROOT_DIR = os.path.abspath(ROOT_DIR)
    #DATA_DIR = os.path.join(ROOT_DIR, 'raw_data/covabdab') # for Talapas
    DATA_DIR = os.path.join(ROOT_DIR, 'lanl/Data/covabdab') # for remote
    CSV = os.path.join(DATA_DIR, 'CoV-AbDab_031022.csv')
    STANDARD_BT = os.path.join(DATA_DIR, 'standardized_bindsto.tsv')
    SPLIT_DIR = os.path.join(DATA_DIR, 'split')
    TYPE = 'binary' # 'regular'/'binary'/'trinary'
    SPLIT = False # True = yes subset the data

    # == Run ==
    df = dataLoader(CSV)
    clean_df = dataCleaner(df, STANDARD_BT, TYPE)

    # == Write out to csv/stats ==
    if SPLIT:
        subset_df = dataSplit(clean_df)
        if TYPE == 'binary':
            subset_df.to_csv(os.path.join(SPLIT_DIR, 'covabdab_split_binary.csv'), index=False)
        elif TYPE == 'trinary':
            subset_df.to_csv(os.path.join(SPLIT_DIR, 'covabdab_split_trinary.csv'), index=False)
        else:
            subset_df.to_csv(os.path.join(SPLIT_DIR, 'covabdab_split.csv'), index=False)
        dataStats(subset_df, TYPE)
    else:
        if TYPE == 'binary':
            clean_df.to_csv(os.path.join(DATA_DIR, 'covabdab_clean_binary.csv'), index=False)
        elif TYPE == 'trinary':
            clean_df.to_csv(os.path.join(DATA_DIR, 'covabdab_clean_trinary.csv'), index=False)
        else:
            clean_df.to_csv(os.path.join(DATA_DIR, 'covabdab_clean.csv'), index=False)
        dataStats(clean_df, TYPE)

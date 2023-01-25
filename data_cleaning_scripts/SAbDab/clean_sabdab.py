#!/usr/bin/env python

import numpy as np
import pandas as pd 

### Initial Exploration
# Load full sabdab tsv into pandas dataframe
sabdab_df = pd.read_csv("sabdab_summary_all.tsv", sep = '\t')
print(sabdab_df.info(), '\n','\n','\n')

# All columns in the sabdab dataframe
sabdab_df_columns : list = ['pdb', 'Hchain', 'Lchain', 'model', 'antigen_chain', 'antigen_type', 'antigen_het_name', 'antigen_name', 'short_header', 'date', 
                 'compound', 'organism', 'heavy_species', 'light_species', 'antigen_species', 'authors', 'resolution', 'method', 'r_free', 'r_factor', 'scfv', 
                 'engineered', 'heavy_subclass', 'light_subclass', 'light_ctype', 'affinity', 'delta_g', 'affinity_method', 'temperature', 'pmid']

print(f"\t> Number of enteries in full database: {len(sabdab_df)}\n")

# Check for NAs in each columns
for i in sabdab_df_columns:
    print(f"NAs in {i}: {sabdab_df[i].isna().sum()}")

# Matches NAs by column total
print(f"\n\n\t> total num of NA values: {sabdab_df.isnull().sum().sum()}\n") # 45270 total NA values

# Which columns need to check for duplicate?? 
# PDB example
pdb_set = set()
for pdb in sabdab_df['pdb']:
    pdb_set.add(pdb)
print(f"\t> number of unique PDB codes: {len(pdb_set)}\n") # 6684 unique pdb codes 


### Organism Column
sabdab_df['organism'] = sabdab_df['organism'].str.lower()
organism = sabdab_df['organism'].unique()
human: list = []

# Filter out all organisms besides human
for entry in organism:
    if 'homo sapiens' in str(entry):
        human.append(entry)

sabdab_df =  sabdab_df[sabdab_df['organism'].isin(human)]
sabdab_df.reset_index(drop=True, inplace=True)
print(f"Cleaning 'organism' column\n")
print(f"\t> Number of homo sapiens enteries: {len(sabdab_df)}\n")


### Antigen_species column (precleaning the column to remove "None", and Nan)
print(f"Precleaning:\n {sabdab_df['antigen_species']}\n")
sabdab_df['antigen_species'] = sabdab_df['antigen_species'].replace(['None'],np.nan)
sabdab_df = sabdab_df[sabdab_df.antigen_species.notnull()]
print(f"Precleaning:\n {sabdab_df['antigen_species']}\n")
print(f'\t> Post cleaning antigen_species: {len(sabdab_df.index)} \n')
sabdab_df['antigen_species'] = sabdab_df['antigen_species'].str.lower()
antigen_species = sabdab_df['antigen_species']

# Dictionary of antigen species and counts, descending order  
species_dict : dict = {}
for species in antigen_species: 
    if species not in species_dict:
        species_dict[species] = 1
    elif species in species_dict:
        species_dict[species] += 1
 
sorted_by_val = sorted(species_dict.items(), key=lambda x:x[1], reverse=True)
sorted_species_dict = dict(sorted_by_val)

# print(sorted_species_dict)
# print(len(sorted_species_dict))
# print(sabdab_df.loc[sabdab_df['antigen_species'] == 'coronavirus'])


### Antigen_Species Column (isolate coronavirus enteries)
coronavirus_antigen: list = []

for entry in antigen_species:
    if 'coronavirus' in str(entry).lower():
        coronavirus_antigen.append(entry)

print(f"Cleaning 'antigen_species' column\n")
print(f"\t> Number of coronavirus antigen species: {len(coronavirus_antigen)}\n")
print(f"Coronavirues angtigen species list: \n {coronavirus_antigen}\n\n\n")

# Save the species list to an excel file to manually inpsect
coronavirus_antigen_df = pd.DataFrame(coronavirus_antigen)
coronavirus_antigen_df.to_excel("sabdab_coronavirus_species_list.xlsx")

# Filter unwanted antigen species from df
sabdab_df = sabdab_df[sabdab_df['antigen_species'].isin(coronavirus_antigen)]
# Reset index of pandas df after removing enteries
sabdab_df.reset_index(drop=True, inplace=True)

# Confirm 1220 coronavirus antigen species enteries remain
print(f"Nunmber of coronavirus antigen species remaining: {len(sabdab_df)}\n") 
print(f"{sabdab_df}\n")


### Antigen_Species Columns (standardize names)
# Isolate unique coronavirus variants
coronavirus_var : list = []
for i, entry in enumerate(coronavirus_antigen):
    if 'bat' in str(entry).lower():
        coronavirus_antigen[i] = 'bat_coronavirus'
        coronavirus_var.append('bat_coronavirus')
    elif 'coronavirus2' in str(entry).lower():
        coronavirus_antigen[i] = 'coronavirus2'
        coronavirus_var.append('coroavirus2')
    elif 'coronavirus' in str(entry).lower():
        coronavirus_antigen[i] = 'coronavirus1'
        coronavirus_var.append('coronavirus1')
    # replace old 'Binds to' unique entries with standardized unique entries

# Counts of coronavirus types
bat_coronavirus : int = 0
coronavirus1 : int = 0 
coronavirus2 : int = 0
for i in coronavirus_antigen:
    if i == 'bat_coronavirus':
        bat_coronavirus += 1
    elif i == 'coronavirus1':
        coronavirus1 += 1
    elif i == "coronavirus2":
        coronavirus2 += 1

print(f"\nStandardizing cornavirus antigens")
print(f"\t> bat_coronavirus count: {bat_coronavirus}")
print(f"\t> coronavirus count:  {coronavirus1}")
print(f"\t> coronavirus2 count:  {coronavirus2}")

# Write out cleaned data to new .csv file
sabdab_df.to_csv('./sabdab_df_clean.csv', index=False)
    
 

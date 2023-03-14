# Machine Learning Approach to a Novel Curated Covid-19 Antibody Database
### By Toni Brooks, Matt Esqueda, Rachel Ferina, and Kaetlyn Gibson

## Overview
We cleaned four publicly available Covid-19 antibody-antigen datasets to curate a MongoDB database for machine learning.
We developed a binary classification machine learning model to identify if a antibody is SARS-CoV-2 or not, as well as a regression machine learning model to predict antigen binding affinity.

## Database collection schemas

**AlphaSeq schema:** 

![as_schema](mongoDB_schemas/alphaseq_schema.png)

**Example of an AlphaSeq record that fits the collection's schema:**
![alphaseq](mongoDB_schemas/alphaseq_entry_example.png)

**Example of a CoV-AbDab record that fits the collection's schema:**
![covabdab](mongoDB_schemas/covabdab_entry_example.png)

**Example of an IMGT record that fits the collection's schema:**
![imgt](mongoDB_schemas/imgt_entry_example.png)

**Example of a SAbDab record that fits the collection's schema:**
![sabdab](mongoDB_schemas/sabdab_entry_example.png)


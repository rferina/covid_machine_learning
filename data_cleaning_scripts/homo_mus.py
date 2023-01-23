#!/usr/bin/env python

'''
This script is to determine the amount of each species and virus we have for the combined cleaned file 
'''

import pandas as pd 
import argparse
import re
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def get_args():
    parser = argparse.ArgumentParser(description="A program to convert cDNA sequences into protein sequences by searching for codons and translating into amino acids")
    parser.add_argument("-f", "--file", help="designates absolute file path to fasta file", type= str, required=True)
    parser.add_argument("-o", "--outfile", help="designates absolute file path to the output file", required=True)
    return parser.parse_args()
args=get_args()

species_dict={}
HIV_count= 0
Sars_count= 0
human_count_HIV =0
mouse_count_HIV =0
human_count_SARS =0
mouse_count_SARS =0


with open(args.file, 'r') as fasta_in, open(args.outfile, 'w') as fasta_out:
    for record in SeqIO.parse(fasta_in, 'fasta'):

        species=record.id.split("|")[1]
        if species in species_dict:
            species_dict[species] +=1 
        else:
            species_dict[species] =1

        if "HIV" in record.description:
            HIV_count +=1

            if "Homo" in record.id:
                human_count_HIV +=1
            if "Mus" in record.id:
                mouse_count_HIV +=1

        if "SARS" in record.description:
            Sars_count +=1

            if "Homo" in record.id:
                human_count_SARS +=1
            if "Mus" in record.id:
                mouse_count_SARS +=1


print(human_count_HIV)
print(mouse_count_HIV)
print(human_count_SARS)
print(mouse_count_SARS)
#print(species_dict)
        
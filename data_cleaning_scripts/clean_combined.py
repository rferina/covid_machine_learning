#!/usr/bin/env python

'This script is to remove unwanted species and cell types'

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

Homo_Mus_out = open('human_mouse.txt','w')

with open(args.file, 'r') as fasta_in, open(args.outfile, 'w') as fasta_out:
    for record in SeqIO.parse(fasta_in, 'fasta'):

        species=record.id.split("|")[1]
        if species in species_dict:
            species_dict[species] +=1 
        else:
            species_dict[species] =1
        
        if re.findall(r'Homo sapiens', record.description, re.I):
            if "Patent" not in record.description:
                if "T-cell" not in record.description:
                    if "B cell" not in record.description:
                        Homo_Mus_out.write(record.format("fasta"))
       
        if re.findall(r'Mus Musculus', record.description, re.I):
            if "Synthetic" not in record.description:
                Homo_Mus_out.write(record.format("fasta"))    

        if re.findall(r'Mus sp.', record.id, re.I):
            Homo_Mus_out.write(record.format("fasta"))  
Homo_Mus_out.close() 

        


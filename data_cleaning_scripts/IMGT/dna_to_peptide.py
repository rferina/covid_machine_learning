#!/usr/bin/env python3.10
' This script is to convert from cDNA to protein using BioPython '
import argparse
import re
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def get_args():
    parser = argparse.ArgumentParser(description="A program to convert cDNA sequences into protein sequences by searching for codons and translating into amino acids")
    parser.add_argument("-f", "--file", help="designates absolute file path to fasta file", required=True)
    parser.add_argument("-o", "--outfile", help="designates absolute file path to the output file", required=True)
    return parser.parse_args()
args=get_args()

rna_sequence=[]
translation_frame=[]

with open(args.file, 'r') as fasta_in, open(args.outfile, 'w') as fasta_out:

    for rna_seq in SeqIO.parse(args.file, 'fasta'):
        # we want to use both forward and reverse sequences
        rna_sequence = [rna_seq.seq, rna_seq.seq.reverse_complement()]
    
        # next we want to generate all possible translation frames
        for seq in rna_sequence:
            translation_frame = (i.translate(to_stop=True, stop_symbol="@") for i in rna_sequence)

            # # select the longest frame
            longest_frame = max(translation_frame, key=len)

            # # write a new record to an output file
            protein_record = SeqRecord(longest_frame, id=rna_seq.id, description= rna_seq.description)
            SeqIO.write(protein_record, fasta_out, 'fasta')


    

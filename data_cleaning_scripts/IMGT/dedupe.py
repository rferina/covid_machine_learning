#!/usr/bin/env python3.10

'This script is utilizing BioPython to remove redundant sequence entries from the dna_to_peptide output file'

import numpy
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

record_id = []
seen_record = set()

with open(args.file, 'r') as fasta_in, open(args.outfile, 'w') as fasta_out:
    for record in SeqIO.parse(args.file, 'fasta'): 
        if record.seq not in seen_record:
            seen_record.add(record.seq)
            record_id.append(record)
            SeqIO.write(record_id, "deduped", "fasta")

# (base) cat deduped | grep -E ^">" | wc -l     = 226

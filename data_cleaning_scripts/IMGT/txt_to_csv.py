#!/usr/bin/env python3.10

'This script is converting csv file with multiple lines per record to a csv file with one line for each record'

import numpy
import argparse
import re
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def get_args():
    parser = argparse.ArgumentParser(description="A program to convert .csv file into a one line .csv file using a defined one line function")
    parser.add_argument("-f", "--file", help="designates absolute file path to csv_insta file", required=True)
    parser.add_argument("-o", "--outfile", help="designates absolute file path to the output file", required=True)
    return parser.parse_args()
args=get_args()


def oneline_csv_insta():
    '''Makes csv sequences on one line. Writes out to the file. 
    Returns the number of records so they can be
    manually compared to the number of header lines in the output file,
    to confirm the output file is accurate.'''
    # make dict with headers as keys and sequences as values
    seq_dict = {}
    with open(args.file, 'r') as csv_in, open(args.outfile, 'w') as csv_out:
        line_count = 0
        for line in csv_in:
            line_count +=1
            line = line.strip('\n')
            # only get header lines
            if line[0] == '>':
                header_line = line
            # populate dict with seq lines (non-header lines)
            else:
                if header_line not in seq_dict:
                    seq_dict[header_line] = line
                else:
                    seq_dict[header_line] += line
    # write out to file
        for keys,vals in seq_dict.items():
            csv_out.write(str(keys) + ',' + str(vals) + '\n')
   
    return len(seq_dict)
oneline_csv_insta()

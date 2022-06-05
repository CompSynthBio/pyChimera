# Alon Diament, Tuller Lab, June 2022.

from collections import Iterable
import numpy as np


the_code = {
    'AAA': 'K',
    'AAC': 'N',
    'AAG': 'K',
    'AAT': 'N',
    'ACA': 'T',
    'ACC': 'T',
    'ACG': 'T',
    'ACT': 'T',
    'AGA': 'R',
    'AGC': 'S',
    'AGG': 'R',
    'AGT': 'S',
    'ATA': 'I',
    'ATC': 'I',
    'ATG': 'M',
    'ATT': 'I',
    'CAA': 'Q',
    'CAC': 'H',
    'CAG': 'Q',
    'CAT': 'H',
    'CCA': 'P',
    'CCC': 'P',
    'CCG': 'P',
    'CCT': 'P',
    'CGA': 'R',
    'CGC': 'R',
    'CGG': 'R',
    'CGT': 'R',
    'CTA': 'L',
    'CTC': 'L',
    'CTG': 'L',
    'CTT': 'L',
    'GAA': 'E',
    'GAC': 'D',
    'GAG': 'E',
    'GAT': 'D',
    'GCA': 'A',
    'GCC': 'A',
    'GCG': 'A',
    'GCT': 'A',
    'GGA': 'G',
    'GGC': 'G',
    'GGG': 'G',
    'GGT': 'G',
    'GTA': 'V',
    'GTC': 'V',
    'GTG': 'V',
    'GTT': 'V',
    'TAA': '*',
    'TAC': 'Y',
    'TAG': '*',
    'TAT': 'Y',
    'TCA': 'S',
    'TCC': 'S',
    'TCG': 'S',
    'TCT': 'S',
    'TGA': '*',
    'TGC': 'C',
    'TGG': 'W',
    'TGT': 'C',
    'TTA': 'L',
    'TTC': 'F',
    'TTG': 'L',
    'TTT': 'F',
}


def nt2aa(seq_nt):
    if is_str_iter(seq_nt):
        return [nt2aa(s) for s in seq_nt]

    seq_nt = seq_nt.upper()
    return ''.join([the_code[seq_nt[i:i+3]]
                    for i in range(0, len(seq_nt), 3)])


def nt2codon():
    pass


def codon2nt():
    pass


def rand_seq(n):
    return ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=n))


def compare_seq(seq1, seq2):
    """ list edits between 2 identically long sequences.
    """
    return [[i, s1, s2] for i, (s1, s2) in
            enumerate(zip(seq1, seq2)) if s1 != s2]


def is_str_iter(obj):
    return not isinstance(obj, str) and isinstance(obj, Iterable)

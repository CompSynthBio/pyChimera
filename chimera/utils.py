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
nt2codon_dict = {k: chr(i) for i, k in enumerate(the_code.keys())}


def nt2aa(seq_nt):
    if is_str_iter(seq_nt):
        return [nt2aa(s) for s in seq_nt]

    seq_nt = seq_nt.upper()
    return ''.join([the_code[seq_nt[i:i+3]]
                    for i in range(0, len(seq_nt), 3)])


def nt2codon(seq):
    """ converts a sequence in NT alphabet and returns a sequence in codon
        alphabet, which is defined as follows:
        the index of the character at a position equals the index of the codon in
        the sorted list of all 64 triplets.
        this allows for all string algorithms, including the Chimera approach, to
        work seemlessly.
        ignores partial codons.
        Alon Diament, Tuller Lab, August 2018 (MATLAB), June 2022 (Python). """

    if is_str_iter(seq):
        return [nt2codon(s) for s in seq]

    if not len(seq):
        return ''

    return ''.join([nt2codon_dict[seq[i:i+3]]
                    for i in range(0, len(seq), 3)])


def codon2nt(seq):
    if is_str_iter(seq):
        return [codon2nt(s) for s in seq]

    if not len(seq):
        return ''

    codon_list = list(nt2codon_dict.keys())

    return ''.join([codon_list[ord(c)] for c in seq])


def rand_seq(n):
    return ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=n))


def compare_seq(seq1, seq2):
    """ list edits between 2 identically long sequences.
    """
    return [[i, s1, s2] for i, (s1, s2) in
            enumerate(zip(seq1, seq2)) if s1 != s2]


def is_str_iter(obj):
    return not isinstance(obj, str) and isinstance(obj, Iterable)

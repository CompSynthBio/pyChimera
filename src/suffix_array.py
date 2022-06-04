# Alon Diament, Tuller Lab, June 2022.

from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import Pool

import numpy as np
pref_type = np.uint32  # sequence length of up to 4294967295
ref_index_type = np.uint16  # no. of sequences of up to 65535


def build_suffix_array(ref, pos_spec=True, n_jobs=None):
    """ returns a suffix array dict with the following keys:

        ref: list of strings
        ind: index in ref
        pos: position (from start) in string

        additional fields for position-specific Chimera:

        pos_from_stop: position (from stop) in string
        sorted_start: indices of suffixes sorted by `pos`
        sorted_stop: indices of suffixes sorted by `pos_from_stop`
    """
    # TODO: sequence assertion
    if type(ref) not in (list, np.array):
        ref = [ref]

    SA = []
    for i, text in enumerate(ref):
        SA.append(np.vstack(
            [build_single_suffix_array(text), len(text)*[i]]
            ))

    # merge-sort style
    with Pool(n_jobs) as pool:
        while len(SA) > 1:
            merged_SA = pool.starmap(
                merge_arrays, zip(SA[0::2], SA[1::2], repeat(ref)))
            if len(merged_SA) < len(SA)/2:
                merged_SA.append(SA[-1])
            SA = merged_SA
    SA = {'ref': ref,
          'ind': SA[0][1].astype(ref_index_type),
          'pos': SA[0][0].astype(pref_type)}

    if not pos_spec:
        return SA

    # additional fields for position-specific Chimera
    lens = np.array([len(r) for r in ref])
    SA['pos_from_stop'] = SA['pos'] - lens[SA['ind']]
    SA['sorted_start'] = np.argsort(SA['pos']).astype(pref_type)
    SA['sorted_stop'] = np.argsort(SA['pos_from_stop']).astype(pref_type)

    return SA


def search_array(key, SA, top=None, bottom=None):
    """ using binary search. returns the position where key should be inserted
        to maintain sorting. """
    if top is None:
        top = 0
    if bottom is None:
        bottom = SA['pos'].size - 1
    while top < bottom:
        mid = (top + bottom) // 2
        if is_key_greater_than(key, SA, mid):
            top = mid + 1
        else:
            bottom = mid

    nS = SA['pos'].size - 1
    if top < nS:
        top = mask_suffix(SA, top, +1)  # forward to next unmasked suffix
    if top == nS:
        top = mask_suffix(SA, top, -1)  # back to last unmasked suffix
    if top == nS and is_key_greater_than(key, SA, top):
        top += 1  # case: end of SA

    return top


def test_suffix_array(SA):
    # assert that all suffixes are sorted
    n = SA['pos'].size
    return [SA['ref'][SA['ind'][i1]][SA['pos'][i1]:] <=
            SA['ref'][SA['ind'][i2]][SA['pos'][i2]:]
            for i1, i2 in zip(range(0, n), range(1, n))]


def print_suffix_array(SA):
    return [get_suffix(SA, i)
            for i in range(SA['pos'].size)]


def is_key_greater_than(key, SA, i):
    return get_suffix(SA, i) < key


def get_suffix(SA, i):
    return SA['ref'][SA['ind'][i]][SA['pos'][i]:]


def mask_suffix(SA, ind, step, min_ind=0, max_ind=None):
    # getting the next suffix (up/down) in SA that isn'tx masked.
    if 'mask' not in SA:
        return ind
    if max_ind is None:
        max_ind = SA['pos'].size

    while not SA['mask'][ind] and (min_ind <= ind+step) and (ind+step < max_ind):
        ind = ind + step

    if not SA['mask'][ind]:
        return None

    return ind


def merge_arrays(SA1, SA2, ref):
    i1 = 0
    i2 = 0
    insert = 0

    n1 = SA1.shape[1]
    n2 = SA2.shape[1]
    outSA = np.zeros((2, n1 + n2), dtype=pref_type)

    suf1 = get_raw_suffix(SA1, i1, ref) 
    suf2 = get_raw_suffix(SA2, i2, ref)
    while (i1 < n1) and (i2 < n2):
        is_greater = suf1 < suf2
        if is_greater:
            outSA[:, insert] = SA1[:, i1]
            i1 += 1
            if i1 < n1:
                suf1 = get_raw_suffix(SA1, i1, ref)
        else:
            outSA[:, insert] = SA2[:, i2]
            i2 += 1
            if i2 < n2:
                suf2 = get_raw_suffix(SA2, i2, ref)
        insert += 1

    if i1 < n1:
        outSA[:, insert:] = SA1[:, i1:]
    if i2 < n2:
        outSA[:, insert:] = SA2[:, i2:]

    return outSA


def get_raw_suffix(SA, i, ref):
    return ref[SA[1, i]][SA[0, i]:]


# suffix_array_ManberMyers from:
# https://github.com/benfulton/Algorithmic-Alley/blob/master/AlgorithmicAlley/SuffixArrays/sa.py
def build_single_suffix_array(str):
    result = []
    def sort_bucket(str, bucket, order=1):
        d = defaultdict(list)
        for i in bucket:
            key = str[i:i+order]
            d[key].append(i)
        for k, v in sorted(d.items()):
            if len(v) > 1:
                sort_bucket(str, v, order*2)
            else:
                result.append(v[0])
        return result

    return sort_bucket(str, (i for i in range(len(str))))

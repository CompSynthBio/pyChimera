# Alon Diament, Tuller Lab, June 2022.

from collections import defaultdict
from itertools import repeat
import math
from multiprocessing.pool import Pool

import numpy as np
pref_type = np.uint32  # sequence length of up to 4294967295
ref_index_type = np.uint16  # no. of sequences of up to 65535

from .utils import is_str_iter


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
    if not is_str_iter(ref):
        ref = [ref]

    with Pool(n_jobs) as pool:
        # SA per reference sequence
        SA = pool.starmap(build_single_suffix_array, enumerate(ref))
        # aggregate, merge-sort style
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

    return SA


def longest_prefix(key, SA, max_len=np.inf):
    """ standard LCS search, with an added homolog sequence
        filter (masking) based on the maximal allowed prefix `max_len`.
    """
    max_len = min([len(key), max_len])

    where = search_suffix(key, SA)
    where = min([max([0, where]), SA['pos'].size-1])

    n = np.inf  # prefix length
    while n > max_len:
        if np.isfinite(n):
            # get here in the >1 iter when exceeding max_len
            SA['homologs'].add(SA['ind'][pind])

        nei = get_neighbors(SA, where)  # adjacent suffixes
        if not len(nei):
            pind = -1
            pref = ''
            return pref, pind

        nei_count = [count_common(key, SA, n) for n in nei]
        nei_max = np.argmax(nei_count)
        pind = nei[nei_max]  # prefix index in SA
        n = nei_count[nei_max]

    pref = key[:nei_count[nei_max]]  # prefix string
    return pref, pind


def get_all_nt_blocks(pref_aa, SA_aa, ref_nt):
    """ finds the all *NT* prefixes in `SA_aa` that codes the given
            AA prefix `pref_aa`.
    """
    n = len(pref_aa)
    left = search_suffix(pref_aa, SA_aa)
    right = search_suffix(pref_aa + '~', SA_aa)

    i_prefix = [i for i in range(left, right)
                if not is_suffix_masked(SA_aa, i)]

    all_blocks = [get_nt_prefix(SA_aa, ref_nt, i, n)
                  for i in i_prefix]

    return all_blocks, i_prefix


def select_window(SA, win_params, pos_start, pos_stop):
    """ filter only relevant suffixes, i.e.:
        1. starting at the same distance from start as `pos_start`, within
        half a window size; or:
        2. starting at the same distance from stop as `pos_stop`, within
        half a window size.

        Alon Diament, Tuller Lab, Januray 2018 (MATLAB), June 2022 (Python).
    """
    max_dist = win_params['size'] / 2

    if win_params['by_start']:
        win_start = range(math.ceil(pos_start - max_dist + win_params['center']),
                          math.ceil(pos_start + max_dist + win_params['center']))  # relative to start codon
        SA['win_start'] = set(win_start)

    if win_params['by_stop']:
        win_stop = range(math.ceil(pos_stop - max_dist + win_params['center']),
                         math.ceil(pos_stop + max_dist + win_params['center']))  # relative to stop codon
        SA['win_stop'] = set(win_stop)


def search_suffix(key, SA, top=None, bottom=None):
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
        top = skip_masked_suffix(SA, top, +1)  # forward to next unmasked suffix
    if top == nS and is_suffix_masked(SA, top):
        top = skip_masked_suffix(SA, top, -1)  # back to last unmasked suffix
    if is_key_greater_than(key, SA, top):
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


def get_neighbors(SA, ind):
    # it would seem that in order for masking to work properly we need to
    # look both up (new) and down (legacy) the found suffix.
    nS = SA['pos'].size
    lo = skip_masked_suffix(SA, max([0, ind-1]), -1)  # next lower neighbor
    hi = skip_masked_suffix(SA, min([nS-1, ind+1]), +1)  # next upper neighbor

    neis = [n for n in [lo, ind, hi]
            if not is_suffix_masked(SA, n)]
    return neis


def count_common(key, SA, i):
    suf = get_suffix(SA, i)[:len(key)]
    key = key[:len(suf)]
    if key == suf:
        return len(key)
    for j, (a, b) in enumerate(zip(key, suf)):
        if a != b:
            break
    return j


def is_key_greater_than(key, SA, i):
    return get_suffix(SA, i) < key


def get_suffix(SA, i):
    return SA['ref'][SA['ind'][i]][SA['pos'][i]:]


def get_nt_prefix(SA_aa, ref_nt, i, n):
    seq = ref_nt[SA_aa['ind'][i]]
    loc = SA_aa['pos'][i]
    return seq[3*loc : 3*(loc+n)]


def skip_masked_suffix(SA, ind, step, min_ind=0, max_ind=None):
    # getting the next suffix (up/down) in SA that isn't masked.
    if max_ind is None:
        max_ind = SA['pos'].size

    while is_suffix_masked(SA, ind) and \
            (min_ind <= ind+step) and (ind+step < max_ind):
        ind = ind + step

    return ind


def is_suffix_masked(SA, ind):
    if 'win_start' not in SA and 'win_stop' not in SA:
        return SA['ind'][ind] in SA['homologs']

    in_win = False
    if 'win_start' in SA:
        in_win |= SA['pos'][ind] in SA['win_start']
    if 'win_stop' in SA:
        in_win |= SA['pos_from_stop'][ind] in SA['win_stop']

    return (not in_win) or (SA['ind'][ind] in SA['homologs'])


def merge_arrays(SA1, SA2, ref):
    i1 = 0
    i2 = 0
    insert = 0

    n1 = SA1.shape[1]
    n2 = SA2.shape[1]
    if n1 == 0:
        return SA2
    if n2 == 0:
        return SA1
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


def build_single_suffix_array(i, str):
    return np.vstack([build_suffix_array_ManberMyers(str), len(str)*[i]])


# suffix_array_ManberMyers from:
# https://github.com/benfulton/Algorithmic-Alley/blob/master/AlgorithmicAlley/SuffixArrays/sa.py
def build_suffix_array_ManberMyers(str):
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

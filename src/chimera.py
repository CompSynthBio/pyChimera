# Alon Diament, Tuller Lab, June 2022.

import numpy as np
import pandas as pd

from .suffix_array import build_suffix_array, longest_prefix, most_freq_prefix
from .utils import *


def calc_cARS(key, SA, max_len=np.inf, max_pos=1):
    """ compute the ChimeraARS (Average Repetitive Substring, Zur and Tuller,
        2015) for a given key.
        includes heuristics for homologous sequences (Diament et al., 2019).

        max_len: if provided, cARS will detect single substrings/blocks that
            are larger than [max_len] and filter the entire gene of origin.
        max_pos: if provided, cARS will detect genes that occur in a fraction
            of positions that is larger than [max_pos] and filter them.

        Alon Diament / Tuller Lab, July 2015 (MATLAB), June 2022 (Python).
    """
    n = len(key)
    cars_vec = np.zeros(n)
    cars_origin = -np.ones(n, dtype=np.int)
    SA['mask'] = np.zeros(SA['pos'].size, dtype=bool)  # empty mask

    pos_queue = np.ones(n, dtype=bool)
    while np.any(pos_queue):
        pos = np.flatnonzero(pos_queue)[0]
        block, pid = longest_prefix(key[pos:], SA, max_len)
        cars_origin[pos] = SA['ind'][pid]
        cars_vec[pos] = len(block)

        if not len(block):
            raise Exception('empty substring at {}, suffix starts with: "{}"'
                            .format(pos, key[pos:pos+10]))

        same = cars_origin == cars_origin[pos]
        if (np.mean(same) > max_pos) and (n > 1):
            # mask origin ref in SA and reset all related positions
            SA['mask'][SA['ind'] == cars_origin[pos]] = True
            pos_queue[same] = True
            cars_origin[same] = -1
        else:
            pos_queue[pos] = False  # success

    cars = np.mean(cars_vec)

    return cars


def calc_cMap(target_aa, SA_aa, ref_nt, max_len=np.inf, max_pos=1):
    """ compute an optimal NT sequence for a target AA sequence based on
        the ChimeraMap (Zur and Tuller, 2015) algorithm.
        includes heuristics for homologous sequences (Diament et al., 2019).

        max_len: if provided, cMap will detect single substrings/blocks that
            are larger than [max_len] and filter the entire gene of origin.
        max_pos: if provided, cMap will detect genes that occur in a fraction
            of positions that is larger than [max_pos] and filter them.

        Alon Diament / Tuller Lab, July 2015 (MATLAB), June 2022 (Python).
    """
    n = len(target_aa)
    B = []  # Chimera blocks
    cmap_origin = -np.ones((2, n), dtype=np.int)
    SA_aa['mask'] = np.zeros(SA_aa['pos'].size, dtype=bool)  # empty mask

    pos = 0  # position in target
    while pos < n:
        block_aa = longest_prefix(target_aa[pos:], SA_aa, max_len)[0]
        if not len(block_aa):
            raise Exception('empty block at {}, suffix starts with: "{}"'
                            .format(pos, target_aa[pos:pos+10]))

        gene, loc, block = most_freq_prefix(block_aa, SA_aa, ref_nt)
        m = len(block_aa)
        cmap_origin[0, pos:pos+m] = gene
        cmap_origin[1, pos:pos+m] = len(B)
        B.append([gene, loc, block])

        same = cmap_origin[0] == cmap_origin[0, pos]
        if (np.mean(same) > max_pos) and (n > 1):
            # mask origin ref in SA and reset all related positions
            SA_aa['mask'][SA_aa['ind'] == cmap_origin[0, pos]] = True
            # backtrack: remove all blocks that appear after the
            # first occurrence of origin ref 
            pos = np.flatnonzero(same)[0]
            blk = cmap_origin[1, pos] - 1
            B = B[:blk + 1]
            cmap_origin[:, pos:] = -1
        else:
            pos += m

    target_opt = ''.join([b[2] for b in B])

    if nt2aa(target_opt) != target_aa:
        raise Exception('non-syonymous optimization')

    return target_opt

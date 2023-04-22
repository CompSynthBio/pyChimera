# Alon Diament, Tuller Lab, June 2022.

from itertools import repeat, starmap
from multiprocessing.pool import Pool

import numpy as np
# import pandas as pd

from .suffix_array import longest_prefix, most_freq_nt_prefix, get_all_nt_blocks, select_window
from .utils import is_str_iter, nt2aa

def_win_params = {'size': 40, 'center': 0, 'by_start': True, 'by_stop': True}


def calc_cARS(key, SA, win_params=None, max_len=np.inf, max_pos=1, return_vec=False, n_jobs=None):
    """ compute the ChimeraARS (Average Repetitive Substring, Zur and Tuller,
        2015) for a given key. when `win_params` is given, compute the
        position-specific ChimeraARS (Diament et al., 2019).
        includes heuristics for homologous sequences (Diament et al., 2019).

        win_params: an optional dictionary with any of the fields
            (size, center, by_start, by_stop). when providing a partial dict
            default values are set (40, 0, True, True).
        max_len: if provided, cARS will detect single substrings/blocks that
            are larger than [max_len] and filter the entire gene of origin.
        max_pos: if provided, cARS will detect genes that occur in a fraction
            of positions that is larger than [max_pos] and filter them.
        return_vec: if True, return the cARS vector instead of the mean.
        n_jobs: number of parallel jobs to run. if None, use all available cores.

        Alon Diament / Tuller Lab, July 2015 (MATLAB), June 2022 (Python).
    """
    if is_str_iter(key):
        with Pool(n_jobs) as pool:
            args = zip(key, repeat(SA), repeat(win_params),
                       repeat(max_len), repeat(max_pos), repeat(return_vec),
                       repeat(1))
            if n_jobs is None or n_jobs > 1:
                return pool.starmap(calc_cARS, args)
            else:
                return list(starmap(calc_cARS, args))

    win_params = init_win_params(win_params)
    SA.pop('win_start', None)
    SA.pop('win_stop', None)

    n = len(key)
    if n == 0:
        return np.nan
    cars_vec = np.zeros(n)
    cars_origin = -np.ones(n, dtype=np.int)
    SA['homologs'] = set()  # empty mask

    pos_queue = np.ones(n, dtype=bool)
    while np.any(pos_queue):
        pos = np.flatnonzero(pos_queue)[0]
        if win_params is not None:
            select_window(SA, win_params, pos, pos-n)

        block, pid = longest_prefix(key[pos:], SA, max_len)
        cars_origin[pos] = SA['ind'][pid]
        cars_vec[pos] = len(block)

        if not len(block):
            print('empty substring at {}/{}, suffix starts with: "{}"'
                  .format(pos, len(key), key[pos:pos+10]))
            cars_origin[pos] = -1
            pos_queue[pos] = False
            continue

        same = cars_origin == cars_origin[pos]
        if (np.mean(same) > max_pos) and (n > 1):
            # mask origin ref in SA and reset all related positions
            SA['homologs'].add(cars_origin[pos])
            pos_queue[same] = True
            cars_origin[same] = -1
        else:
            pos_queue[pos] = False  # success

    if return_vec:
        return cars_vec

    return np.mean(cars_vec)


def calc_cMap(target_aa, SA_aa, ref_nt, win_params=None, max_len=np.inf, max_pos=1,
              block_select='most_freq', n_jobs=None):
    """ compute an optimal NT sequence for a target AA sequence based on
        the ChimeraMap (Zur and Tuller, 2015) algorithm. when `win_params`
        is given, compute the position-specific ChimeraMap (Diament et al., 2019).
        includes heuristics for homologous sequences (Diament et al., 2019).

        win_params: an optional dictionary with any of the fields
            (size, center, by_start, by_stop). when providing a partial dict
            default values are set (40, 0, True, True).
        max_len: if provided, cMap will detect single substrings/blocks that
            are larger than [max_len] and filter the entire gene of origin.
        max_pos: if provided, cMap will detect genes that occur in a fraction
            of positions that is larger than [max_pos] and filter them.
        block_select: the method to select among synonymous sequence blocks.
            the 'all' method outputs all equally-optimal unique synonymous
            blocks, and in this case the `max_pos` param is ignored. value
            in {'most_freq', 'all'}, defaults to 'most_freq'.
        n_jobs: number of parallel jobs to run. if None, use all available cores.

        Alon Diament / Tuller Lab, July 2015 (MATLAB), June 2022 (Python).
    """
    if is_str_iter(target_aa):
        with Pool(n_jobs) as pool:
            args = zip(target_aa, repeat(SA_aa), repeat(ref_nt), repeat(win_params),
                       repeat(max_len), repeat(max_pos), repeat(1))
            if n_jobs is None or n_jobs > 1:
                return pool.starmap(calc_cMap, args)
            else:
                return list(starmap(calc_cMap, args))

    win_params = init_win_params(win_params)
    SA_aa.pop('win_start', None)
    SA_aa.pop('win_stop', None)

    n = len(target_aa)
    B = []  # Chimera blocks
    cmap_origin = -np.ones((2, n), dtype=np.int)
    SA_aa['homologs'] = set()  # empty mask

    pos = 0  # position in target
    while pos < n:
        if win_params is not None:
            select_window(SA_aa, win_params, pos, pos-n)

        block_aa = longest_prefix(target_aa[pos:], SA_aa, max_len)[0]
        m = len(block_aa)
        if m == 0:
            raise ValueError('empty block at {}/{}, suffix starts with: "{}"'
                             .format(pos, len(target_aa), target_aa[pos:pos+10]))

        if block_select == 'most_freq':
            gene, loc, block = most_freq_nt_prefix(block_aa, SA_aa, ref_nt)
        elif block_select == 'all':
            B.append(np.unique(get_all_nt_blocks(block_aa, SA_aa, ref_nt)[0]))
            pos += m
            continue
        else:
            raise ValueError('block_select must be in {"most_freq", "all"}')

        cmap_origin[0, pos:pos+m] = gene
        cmap_origin[1, pos:pos+m] = len(B)
        B.append([gene, loc, block])

        same = cmap_origin[0] == cmap_origin[0, pos]
        if (np.mean(same) > max_pos) and (n > 1):
            # mask origin ref in SA and reset all related positions
            SA_aa['homologs'].add(cmap_origin[0, pos])
            # backtrack: remove all blocks that appear after the
            # first occurrence of origin ref
            pos = np.flatnonzero(same)[0]
            blk = cmap_origin[1, pos] - 1
            B = B[:blk + 1]
            cmap_origin[:, pos:] = -1
        else:
            pos += m

    if block_select == 'all':
        return B

    target_opt = ''.join([b[2] for b in B])

    if nt2aa(target_opt) != target_aa:
        raise ValueError('non-syonymous optimization')

    return target_opt  #, pd.DataFrame(B, columns=['gene', 'loc', 'block'])


def init_win_params(win_params):
    """ add missing params to the dictionary in place. """
    if win_params is None:
        return
    for k, v in def_win_params.items():
        if k in win_params:
            continue
        win_params[k] = v
    return win_params

# Alon Diament, Tuller Lab, June 2022.
from collections import Counter
from itertools import repeat, starmap
from multiprocessing.pool import Pool

import numpy as np
# import pandas as pd

from .suffix_array import longest_prefix, get_all_nt_blocks, select_window
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
    cars_origin = -np.ones(n, dtype=int)
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
              block_select='most_freq', n_seqs=1, min_blocks=1, return_vec=False, n_jobs=None):
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
                       repeat(max_len), repeat(max_pos), repeat(block_select), repeat(n_seqs),
                       repeat(min_blocks), repeat(return_vec), repeat(1))
            if n_jobs is None or n_jobs > 1:
                return pool.starmap(calc_cMap, args)
            else:
                return list(starmap(calc_cMap, args))

    win_params = init_win_params(win_params)
    SA_aa.pop('win_start', None)
    SA_aa.pop('win_stop', None)

    n = len(target_aa)
    SA_aa['homologs'] = set()  # empty mask

    pos_pass = False

    while not pos_pass:
        all_blocks = []
        cmap_origin = np.zeros(len(SA_aa["ind"]), dtype=int)
        pos = 0  # position in target
        while pos < n:
            if win_params is not None:
                select_window(SA_aa, win_params, pos, pos - n)

            block_aa = longest_prefix(target_aa[pos:], SA_aa, max_len)[0]
            if len(block_aa) == 0:
                raise ValueError('empty block at {}/{}, suffix starts with: "{}"'
                                 .format(pos, len(target_aa), target_aa[pos:pos + 10]))

            prev_blocks = np.empty(shape=0)
            prev_block_aa = ""
            while len(block_aa) > 0:
                blocks, i_prefixes = get_all_nt_blocks(block_aa, SA_aa, ref_nt)

                if len(set(prev_blocks)) > len(set(blocks)):  # if shortening the aa block reduced the number of nt blocks, use the longer aa block
                    block_aa = prev_block_aa
                    blocks = prev_blocks
                    break

                if len(block_aa) == 1 or len(set(blocks)) >= min_blocks:
                    break

                prev_block_aa = block_aa
                prev_blocks = blocks
                block_aa = block_aa[:-1]

            if block_select == 'most_freq':
                blocks = [b[0] for b in Counter(sorted(blocks)).most_common(n_seqs)]
            else:
                blocks = list(set(blocks))

            cmap_origin[[SA_aa['ind'][pid] for pid in i_prefixes]] += len(block_aa)
            all_blocks.append(blocks)
            pos += len(block_aa)

        homologs = np.argwhere(cmap_origin / n > max_pos).flatten()
        SA_aa['homologs'].update(homologs)
        if homologs.size == 0:
            pos_pass = True

    if return_vec:
        return all_blocks

    i_b = 0  # block index
    target_opt = n_seqs * [""]
    for blocks in all_blocks:
        n_b = len(blocks)
        for i_seq in range(n_seqs):
            i = ((i_seq if i_b % 2 else int(n_b * i_seq / n_seqs)) + i_b) % n_b
            target_opt[i_seq] += blocks[i]
        if n_b != 1:
            i_b += 1

    if (np.array(nt2aa(target_opt)) != target_aa).any():
        raise ValueError('non-syonymous optimization')

    return target_opt


def init_win_params(win_params):
    """ add missing params to the dictionary in place. """
    if win_params is None:
        return
    for k, v in def_win_params.items():
        if k in win_params:
            continue
        win_params[k] = v
    return win_params

# Alon Diament, Tuller Lab, June 2022.

import numpy as np

from .suffix_array import build_suffix_array, longest_prefix


def calc_cARS(key, SA, max_len=np.inf, max_pos=1):
    """ compute the ChimeraARS (Average Repeatetive Substring, Zur and Tuller
        2014) for a given key. an optimized implementaion.

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
    err = False

    pos_queue = np.ones(n, dtype=bool)
    while np.any(pos_queue):
        pos = np.flatnonzero(pos_queue)[0]
        substring, pid = longest_prefix(key[pos:], SA, max_len)
        cars_origin[pos] = SA['ind'][pid]
        cars_vec[pos] = len(substring)

        if not len(substring):
            print('empty substring at {}'.format(pos))
            cars_origin[pos] = -1
            pos_queue[pos] = False  # skip
            err = True

        same = cars_origin == cars_origin[pos]
        if (np.mean(same) > max_pos) and (n > 1):
            # mask origin ref in SA and reset all positions
            SA['mask'][SA['ind'] == cars_origin[pos]] = True
            pos_queue[same] = True
            cars_origin[same] = -1
        else:
            pos_queue[pos] = False  # success

    cars = np.mean(cars_vec)

    return cars

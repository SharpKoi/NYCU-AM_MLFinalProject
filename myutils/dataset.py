from itertools import chain
from typing import Sequence, Iterable

import numpy as np


def flatten(ndlist: Sequence):
    while isinstance(ndlist[0], list):
        ndlist = list(chain.from_iterable(ndlist))

    return ndlist


def pad_sequences(data: Iterable, value, max_length: int = -1):
    if max_length < 0:
        max_length = np.max([len(seq) for seq in data])

    for seq in data:
        pad_size = max_length - len(seq)
        if pad_size > 0:
            seq.extend([value] * pad_size)

    return data

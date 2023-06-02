import sys
from functools import reduce

import numpy as np
import scipy.stats as st


def tensor_size(tensor):
    """
    Compute size of feature map at CONTENT_LAYER
    """
    from operator import mul

    return reduce(mul, (d for d in tensor.shape[1:]), 1)

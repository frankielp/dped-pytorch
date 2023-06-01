import scipy.stats as st
import numpy as np
import sys

from functools import reduce

def tensor_size(tensor):
    '''
    Compute size of feature map at CONTENT_LAYER
    '''
    from operator import mul
    return reduce(mul, (d for d in tensor.shape[1:]), 1)
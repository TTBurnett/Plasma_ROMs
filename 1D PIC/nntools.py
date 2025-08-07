import numpy as np

def normalize(array, axis=None):
    '''
    Returns:
    1) normalized array
    2) mean
    3) standard deviation
    '''
    mean = np.mean(array, axis=axis, keepdims=axis!=None)
    std = np.std(array, axis=axis, keepdims=axis!=None)
    normalized = (array - mean) / std
    return normalized, mean, std

def shift_to_zero_one(array):
    '''
    Returns:
    1) shifted array
    2) shift
    3) scaling
    '''
    max_val = array.max()
    min_val = array.min()
    scaling = (max_val - min_val)
    shifted_array = (array - min_val) / scaling
    return shifted_array, min_val, scaling

def unnormalize(array, mean, std):
    return array*std + mean
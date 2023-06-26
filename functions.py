# Functions (methods) for the MacDermott and other models calculation. Author: Kubeyev A.

import numpy as np


def coarsenArray(original_array, numElems):
    '''
    Function coarsens (downsizes) supplied original array into equally spaced number of elements
    '''

    idx = np.round(np.linspace(0, len(original_array) - 1, numElems)).astype(int)
    downsized_array = original_array[idx]
    return downsized_array


def coarsenArrayMixed(original_array, numElems, takefirst = 5):
    '''
    Function coarsens (downsizes) supplied original array into equally spaced number of elements. It always takes first 5
    array items as it is and corrsens the rest.
    '''

    first_five = original_array[ :takefirst+1]

    rest_array = original_array[takefirst+1 : ]
    idx = np.round(np.linspace(0, len(rest_array) - 1, numElems)).astype(int)
    downsized_array = rest_array[idx]

    final_array = np.concatenate((first_five, downsized_array), axis=0)

    return final_array

def coarsenArrayLog(original_array, numElems, exp=10):
    '''
        Function takes fine array and coarsens it by taking 1st and logarithmically specified numElems for the coarse array
    '''
    first_item_array = np.array([original_array[0]])
    numElems = numElems - 1

    rest_array = original_array[1:]
    idx = np.round(logDistr(1, len(rest_array) - 1, exp=exp, n=numElems)).astype(int)
    downsized_array = rest_array[idx]


    final_array = np.concatenate((first_item_array, downsized_array), axis=0)

    assert len(final_array) == numElems + 1

    return final_array


def coarsenArrayLog2(original_array, numElems, exp=10):
    '''
        Function takes fine array and coarsens it by taking first 3 items and logarithmically specified numElems for the coarse array
    '''
    first_item_array = original_array[0:3]
    numElems = numElems - 3

    rest_array = original_array[3:]
    idx = np.round(logDistr(1, len(rest_array) - 1, exp=exp, n=numElems)).astype(int)
    downsized_array = rest_array[idx]


    final_array = np.concatenate((first_item_array, downsized_array), axis=0)

    assert len(final_array) == numElems + 3

    return final_array


def logDistr(x1, x2, exp, n):
    '''
     DESCRIPTION:
       Create a logarithmic distribution. Works only on 1D array (vector), to extend for 2D see your MATLAB implementation

     PARAMETERS:
       n       -  number of entries
       exp     -  exponentialliness
       x1, x2  -  minimum and maximum limits of the data you have. Can be vectors.

     RETURNS:
       y - logarithmic distribution, a row vector of 'n' log spaced points between x1 and x2.

     AUTHOR:
     Aidan (Amanzhol) Kubeyev

    '''

    mat = np.linspace(0, np.log10(exp + 1), n)
    y = (x2 - x1) / exp * (10**mat - 1) + x1

    return y




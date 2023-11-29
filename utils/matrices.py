# matrices.py
import cupy as cp

def oplu_moments_mat(fc_hl_units):
    """ Generates auxiliary matrices used for the computation of moments of OPLU

    Function that generates as global variables the matrices required in the oplu_moments.py file to reduce overhead.
    Each matrix depends only on the number of units in the layer that it will be used. Each matrix is a dictionary which
    each entry's name is the number of units that the matrix can be used for. Then, in the case of a NN with different
    widths, one can use the size of the layer stored in the BjorckLinear object to access the required matrices

    Args:
        fc_hl_units: list in which each entry contains the number of units in the respective hidden layer

    Returns:

    """
    global mat1, mat2, mat3, mat4
    mat1 = {}
    mat2 = {}
    mat3 = {}
    mat4 = {}

    for units in fc_hl_units:
        if '{}'.format(units) not in mat1:
            idx_aux = cp.arange((units / 2).astype(int))

            # identity matrix
            mat1['{}'.format(units)] = cp.identity(units)

            # pairwise row permutation matrix
            aux = cp.zeros((units,units))
            aux[idx_aux*2,idx_aux*2 + 1] = 1
            aux[idx_aux*2 + 1,idx_aux*2] = 1
            mat2['{}'.format(units)] = aux

            aux = cp.zeros((units, (units / 2).astype(int)))
            aux[idx_aux * 2, idx_aux] = 1
            aux[idx_aux * 2 + 1, idx_aux] = -1
            mat3['{}'.format(units)] = aux

            aux = cp.zeros((units, (units / 2).astype(int)))
            aux[idx_aux * 2, idx_aux] = 1
            aux[idx_aux * 2 + 1, idx_aux] = 1
            mat4['{}'.format(units)] = aux

import cupy as cp

def power_iteration(A, init_u=None, n_iters=10, return_uv=True):
    """
    Power iteration for matrix
    """
    shape = list(A.shape)
    # shape[-2] = shape[-1]
    shape[-1] = 1
    shape = tuple(shape)
    if init_u is None:
        u = cp.randn(*shape, dtype=A.dtype, device=A.device)
    else:
        assert tuple(init_u.shape) == shape, (init_u.shape, shape)
        u = init_u
    for _ in range(n_iters):
        v = A.transpose(-1, -2) @ u
        v /= v.norm(dim=-2, keepdim=True)
        u = A @ v
        u /= u.norm(dim=-2, keepdim=True    
    s = (u.transpose(-1, -2) @ A @ v).squeeze(-1).squeeze(-1)
    if return_uv:
        return u, s, v
    return 


def bcop_mat(cv_hl_kernels):
    """ Generates auxiliary matrices used for the computation of BCO

    Function that generates as global variables the matrices required in the bcop.py file to reduce overhead.
    Each matrix depends only on the number of kernels in the layer that it will be used. Each matrix is a dictionary which
    each entry's name is the number of kernels that the matrix can be used for. Then, in the case of a NN with different
    widths, one can use the size of the layer stored in the BjorckLinear object to access the required matrices

    Args:
        cv_hl_kernels: list in which each entry contains the number of kernels in the respective hidden layer

    Returns:

    """
    global mat1, mat2, mat3, mat4
    mat1 = {}
    mat2 = {}
    mat3 = {}
    mat4 = {}

    for kernels in cv_hl_kernels:
        if '{}'.format(kernels) not in mat1:
            aux1 = cp.ones((2, 2, kernels, kernels))
            aux2 = cp.zeros((2, 2, kernels, kernels))

            mat = aux1.copy()
            mat[0,1,:,:] *= -1
            mat[1,0,:,:] *= -1

            #[1,-1][-1,1] matrix
            mat1['{}'.format(kernels)] = mat

            mat = aux2.copy()
            mat[0,1,:,:] += 1
            mat[1,1,:,:] -= 1
            #[0,1][0,-1] matrix
            mat2['{}'.format(kernels)] = mat

            mat = aux2.copy()
            mat[1,0,:,:] += 1
            mat[1,1,:,:] -= 1
            #[0,0][1,-1] matrix
            mat3['{}'.format(kernels)] = mat
            
            mat = aux2.copy()
            mat[1,1,:,:] += 1
            #[0,0][0,1] matrix
            mat4['{}'.format(kernels)] = mat

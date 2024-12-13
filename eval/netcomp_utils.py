"""
**********
Eigenstuff & Matrix Helpers
**********

Functions for calculating eigenstuff of graphs, matrices associated with graphs,
and linear algebraic helper functions.
"""

from scipy import sparse as sps
import numpy as np
from scipy.sparse import linalg as spla
from numpy import linalg as la
from scipy.sparse import issparse


_eps = 10 ** (-10)  # a small parameter

######################
## Helper Functions ##
######################


def _flat(D):
    """Flatten column or row matrices, as well as arrays."""
    if issparse(D):
        raise ValueError("Cannot flatten sparse matrix.")
    d_flat = np.array(D).flatten()
    return d_flat


def _pad(A, N):
    """Pad A so A.shape is (N,N)"""
    n, _ = A.shape
    if n >= N:
        return A
    else:
        if issparse(A):
            # thrown if we try to np.concatenate sparse matrices
            side = sps.csr_matrix((n, N - n))
            bottom = sps.csr_matrix((N - n, N))
            A_pad = sps.hstack([A, side])
            A_pad = sps.vstack([A_pad, bottom])
        else:
            side = np.zeros((n, N - n))
            bottom = np.zeros((N - n, N))
            A_pad = np.concatenate([A, side], axis=1)
            A_pad = np.concatenate([A_pad, bottom])
        return A_pad


########################
## Matrices of Graphs ##
########################


def degree_matrix(A):
    """Diagonal degree matrix of graph with adjacency matrix A

    Parameters
    ----------
    A : matrix
        Adjacency matrix

    Returns
    -------
    D : SciPy sparse matrix
        Diagonal matrix of degrees.
    """
    n, m = A.shape
    degs = _flat(A.sum(axis=1))
    D = sps.spdiags(degs, [0], n, n, format="csr")
    return D


def laplacian_matrix(A, normalized=False):
    """Diagonal degree matrix of graph with adjacency matrix A

    Parameters
    ----------
    A : matrix
        Adjacency matrix
    normalized : Bool, optional (default=False)
        If true, then normalized laplacian is returned.

    Returns
    -------
    L : SciPy sparse matrix
        Combinatorial laplacian matrix.
    """
    n, m = A.shape
    D = degree_matrix(A)
    L = D - A
    if normalized:
        degs = _flat(A.sum(axis=1))
        rootD = sps.spdiags(np.power(degs, -1 / 2), [0], n, n, format="csr")
        L = rootD * L * rootD
    return L


def _eigs(M, which="SR", k=None):
    """Helper function for getting eigenstuff.

    Parameters
    ----------
    M : matrix, numpy or scipy sparse
        The matrix for which we hope to get eigenstuff.
    which : string in {'SR','LR'}
        If 'SR', get eigenvalues with smallest real part. If 'LR', get largest.
    k : int
        Number of eigenvalues to return

    Returns
    -------
    evals, evecs : numpy arrays
        Eigenvalues and eigenvectors of matrix M, sorted in ascending or
        descending order, depending on 'which'.

    See Also
    --------
    numpy.linalg.eig
    scipy.sparse.eigs
    """
    n, _ = M.shape
    if k is None:
        k = n
    if which not in ["LR", "SR"]:
        raise ValueError("which must be either 'LR' or 'SR'.")
    M = M.astype(float)
    if issparse(M) and k < n - 1:
        evals, evecs = spla.eigs(M, k=k, which=which)
    else:
        try:
            M = M.todense()
        except:
            pass
        evals, evecs = la.eig(M)
        # sort dem eigenvalues
        inds = np.argsort(evals)
        if which == "LR":
            inds = inds[::-1]
        else:
            pass
        inds = inds[:k]
        evals = evals[inds]
        evecs = np.matrix(evecs[:, inds])
    return np.real(evals), np.real(evecs)


def normalized_laplacian_eig(A, k=None):
    """Return the eigenstuff of the normalized Laplacian matrix of graph
    associated with adjacency matrix A.

    Calculates via eigenvalues if

    K = D^(-1/2) A D^(-1/2)

    where `A` is the adjacency matrix and `D` is the diagonal matrix of
    node degrees. Since L = I - K, the eigenvalues and vectors of L can
    be easily recovered.

    Parameters
    ----------
    A : NumPy matrix
        Adjacency matrix of a graph

    k : int, 0 < k < A.shape[0]-1
        The number of eigenvalues to grab.

    Returns
    -------
    lap_evals : NumPy array
       Eigenvalues of L

    evecs : NumPy matrix
       Columns are the eigenvectors of L

    Notes
    -----
    This way of calculating the eigenvalues of the normalized graph laplacian is
    more numerically stable than simply forming the matrix L = I - K and doing
    numpy.linalg.eig on the result. This is because the eigenvalues of L are
    close to zero, whereas the eigenvalues of K are close to 1.

    References
    ----------

    See Also
    --------
    nx.laplacian_matrix
    nx.normalized_laplacian_matrix
    """
    n, m = A.shape
    ##
    ## TODO: implement checks on the adjacency matrix
    ##
    degs = _flat(A.sum(axis=1))
    # the below will break if
    inv_root_degs = [d ** (-1 / 2) if d > _eps else 0 for d in degs]
    inv_rootD = sps.spdiags(inv_root_degs, [0], n, n, format="csr")
    # build normalized diffusion matrix
    K = inv_rootD * A * inv_rootD
    evals, evecs = _eigs(K, k=k, which="LR")
    lap_evals = 1 - evals
    return np.real(lap_evals), np.real(evecs)

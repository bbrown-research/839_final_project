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
from networkx import Graph, adjacency_matrix


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


# pulled straight from:
# https://github.com/peterewills/NetComp/blob/master/netcomp/distance/exact.py


def _edit_dist(A1, A2):
    """The edit distance between graphs, defined as the number of changes one
    needs to make to put the edge lists in correspondence.

    Parameters
    ----------
    A1, A2 : NumPy matrices
        Adjacency matrices of graphs to be compared

    Returns
    -------
    dist : float
        The edit distance between the two graphs
    """
    dist = np.abs((A1 - A2)).sum() / 2
    return dist


def edit_dist(G1: Graph, G2: Graph):
    return _edit_dist(adjacency_matrix(G1), adjacency_matrix(G2))


# code ripped from:
# https://github.com/peterewills/NetComp/blob/master/netcomp/distance/exact.py
# and nearby files


def _spectral_dist(A1, A2, k=None, p=2, kind="laplacian"):
    """The lambda distance between graphs, which is defined as

        d(G1,G2) = norm(L_1 - L_2)

    where L_1 is a vector of the top k eigenvalues of the appropriate matrix
    associated with G1, and L2 is defined similarly.

    Parameters
    ----------
    A1, A2 : NumPy matrices
        Adjacency matrices of graphs to be compared

    k : Integer
        The number of eigenvalues to be compared

    p : non-zero Float
        The p-norm is used to compare the resulting vector of eigenvalues.

    kind : String , in {'laplacian','laplacian_norm','adjacency'}
        The matrix for which eigenvalues will be calculated.

    Returns
    -------
    dist : float
        The distance between the two graphs

    Notes
    -----
    The norm can be any p-norm; by default we use p=2. If p<0 is used, the
    result is not a mathematical norm, but may still be interesting and/or
    useful.

    If k is provided, then we use the k SMALLEST eigenvalues for the Laplacian
    distances, and we use the k LARGEST eigenvalues for the adjacency
    distance. This is because the corresponding order flips, as L = D-A.

    References
    ----------

    See Also
    --------
    netcomp.linalg._eigs
    normalized_laplacian_eigs

    """
    # ensure valid k
    n1, n2 = [A.shape[0] for A in [A1, A2]]
    N = min(n1, n2)  # minimum size between the two graphs
    if k is None or k > N:
        k = N
    if kind is "laplacian":
        # form matrices
        L1, L2 = [laplacian_matrix(A) for A in [A1, A2]]
        # get eigenvalues, ignore eigenvectors
        evals1, evals2 = [_eigs(L)[0] for L in [L1, L2]]
    elif kind is "laplacian_norm":
        # use our function to graph evals of normalized laplacian
        evals1, evals2 = [normalized_laplacian_eig(A)[0] for A in [A1, A2]]
    elif kind is "adjacency":
        evals1, evals2 = [_eigs(A)[0] for A in [A1, A2]]
        # reverse, so that we are sorted from large to small, since we care
        # about the k LARGEST eigenvalues for the adjacency distance
        evals1, evals2 = [evals[::-1] for evals in [evals1, evals2]]
    else:
        raise ValueError(
            "Invalid kind, choose from 'laplacian', "
            "'laplacian_norm', and 'adjacency'."
        )
    dist = la.norm(evals1[:k] - evals2[:k], ord=p)
    return dist


def spectral_dist(G1: Graph, G2: Graph, p=2, kind="laplacian"):
    return _spectral_dist(adjacency_matrix(G1), adjacency_matrix(G2), p=p, kind=kind)

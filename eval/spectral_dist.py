# code ripped from:
# https://github.com/peterewills/NetComp/blob/master/netcomp/distance/exact.py
# and nearby files

from numpy import linalg as la
from netcomp_utils import laplacian_matrix, _eigs, normalized_laplacian_eig


def spectral_dist(A1, A2, k=None, p=2, kind="laplacian"):
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

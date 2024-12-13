import networkx as nx
from edit_dist import edit_dist
from spectral_dist import spectral_dist
from random import seed


if __name__ == "__main__":
    seed(123)

    G1, G2 = [nx.erdos_renyi_graph(10, 1 / 2) for _ in range(2)]  # 2 random graphs
    A1, A2 = [nx.adjacency_matrix(G) for G in [G1, G2]]

    print()
    print("Should be 0")
    print(edit_dist(A1, A1))
    print(spectral_dist(A1, A1, kind="laplacian_norm"))

    print()
    print("Should be non-zero")
    print(edit_dist(A1, A2))
    print(spectral_dist(A1, A2, kind="laplacian_norm"))

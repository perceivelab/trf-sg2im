import dgl
import numpy as np
import torch


def path(m):
    """ Returns a path matrix """
    p = [list(row) for row in m]
    n = len(p)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            if p[j][i]:
                for k in range(0, n):
                    if p[j][k] == 0:
                        p[j][k] = p[i][k]
    return p


def hsu(m):
    """ Transforms a given directed acyclic graph into its minimal equivalent """
    n = len(m)
    for j in range(n):
        for i in range(n):
            if m[i][j]:
                for k in range(n):
                    if m[j][k]:
                        m[i][k] = 0


def get_minimal_graph(g):
    p = path(g)
    hsu(p)
    return p


def triplets_to_adj_matrix(triplets):
    triplets = np.array(triplets, dtype='int').copy()
    triplets_concatenated = np.array(np.concatenate(
        [triplets[:, :1], triplets[:, 2:]], axis=1))
    N = int(np.max(triplets_concatenated) + 1)
    grid = np.zeros((N, N), dtype='uint8')
    for k in range(triplets_concatenated.shape[0]):
        i, j = triplets_concatenated[k]
        grid[i, j] = 1
    return grid.tolist()


def matrix_to_triplets(m, rel_idx):
    rows, cols = np.where(np.array(m, dtype='uint8') == 1)
    rels = np.ones(len(rows)) * rel_idx
    return np.stack([rows, rels, cols], axis=1)


def reduce_transitive_edges(triplets, p_keep=0.5):
    if len(triplets) < 3:
        return triplets
    mat = triplets_to_adj_matrix(triplets)
    min_graph = get_minimal_graph(mat)
    prob_mat = np.random.uniform(0, 1, (len(mat), len(mat)))
    new_mat = (prob_mat * (np.array(mat) - np.array(min_graph)) >
               (1 - p_keep)).astype('uint8') + np.array(min_graph)
    new_triplets = matrix_to_triplets(new_mat, triplets[0][1])
    return new_triplets


def get_labels_from_graph(graph):
    labels = []
    subgraphs = dgl.unbatch(graph)
    for subgraph in subgraphs:
        labels.append(subgraph.ndata['feat'])
    labels = torch.stack(labels).to(graph.device)
    return labels

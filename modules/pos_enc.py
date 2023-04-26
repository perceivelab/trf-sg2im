import math

import dgl
import numpy as np
import torch
from scipy import sparse as sp
from torch import nn


class SinePositionalEncoding(nn.Module):

    def __init__(self, emb_size, dropout=0.1, max_len=10):
        super(SinePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x.to(self.pe.device)
        x = x * math.sqrt(self.emb_size)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        DGL implementation
    """
    n_nodes = g.number_of_nodes()
    pad_size = pos_enc_dim + 1 - n_nodes
    pad_size = max(0, pad_size)

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float).toarray()
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1)
                 ** -0.5, dtype=float).toarray()
    L = np.eye(n_nodes) - N @ A @ N
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    EigVec = np.pad(EigVec, (0, pad_size))[:n_nodes]
    g.ndata['lap_pos_enc'] = torch.from_numpy(
        EigVec[:, 1:pos_enc_dim+1]).float()

    return g

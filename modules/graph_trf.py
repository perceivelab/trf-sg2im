import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from trainers.t_base import _init_weights
from utils.layers import build_mlp
from utils.model import MultiHeadAttentionLayer


def get_lap_pos_enc(graph):
    # Implementation from graphtransformer
    lap_pos_enc = graph.ndata['lap_pos_enc']
    sign_flip = torch.rand(lap_pos_enc.size(1), device=graph.device)
    sign_flip[sign_flip >= 0.5] = 1.0
    sign_flip[sign_flip < 0.5] = -1.0
    lap_pos_enc = lap_pos_enc * sign_flip.unsqueeze(0)
    return lap_pos_enc


"""
    Graph Transformer Layer with edge features
    
"""


class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False, mixed_precision=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.mixed_precision = mixed_precision

        self.attention = MultiHeadAttentionLayer(
            in_dim, out_dim//num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e, amp=False):
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)

        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h  # residual connection
            e = e_in1 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h  # for second residual connection
        e_in2 = e  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # residual connection
            e = e_in2 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads, self.residual)


"""
    Graph Transformer with edge features
    
"""


class GraphTransformerNet(nn.Module):
    def __init__(self,
                 n_objs,
                 n_rels,
                 emb_size,
                 n_heads,
                 n_enc_layers,
                 pos_enc_dim,
                 dropout=0.1,
                 layer_norm=False,
                 batch_norm=True,
                 residual=True,
                 skip_edge_feat=False,
                 mixed_precision=False,
                 lap_pos_enc=True):
        super().__init__()
        out_dim = emb_size

        if lap_pos_enc:
            pos_enc_dim = pos_enc_dim
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, emb_size)

        self.lap_pos_enc = lap_pos_enc
        self.skip_edge_feat = skip_edge_feat
        self.embedding_h = nn.Embedding(n_objs, emb_size)

        if not skip_edge_feat:
            self.embedding_e = nn.Embedding(n_rels, emb_size)
        else:
            self.embedding_e = nn.Linear(1, emb_size)

        self.in_feat_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(emb_size, emb_size, n_heads, dropout, layer_norm,
                                  batch_norm, residual, mixed_precision=mixed_precision)
            for _ in range(n_enc_layers-1)])
        self.layers.append(GraphTransformerLayer(emb_size, out_dim, n_heads, dropout,
                           layer_norm, batch_norm, residual, mixed_precision=mixed_precision))

        self.box_regressor = nn.Sequential(
            build_mlp([emb_size, emb_size //
                      2, emb_size // 4], 'gelu'),
            nn.Linear(emb_size // 4, 4),
            nn.SiLU()
        )
        self.class_regressor = build_mlp(
            [emb_size, emb_size // 2, n_objs])

        self.box_regressor.apply(_init_weights)
        self.class_regressor.apply(_init_weights)

    def forward(self, g):

        h = g.ndata['feat']
        e = g.edata['feat']
        h_lap_pos_enc = get_lap_pos_enc(g)
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.skip_edge_feat:  # edge feature set to 1
            e = torch.ones(e.size(0), 1, device=h.device)
        e = self.embedding_e(e)

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)

        h = rearrange(h, '(b s) e -> b s e', b=g.batch_size)
        boxes = self.box_regressor(h)
        logits = self.class_regressor(h)
        return boxes, logits

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss

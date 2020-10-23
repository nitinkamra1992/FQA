import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

from itertools import chain, permutations

import sys
sys.path.append('./')
from utils.model_utils import Reshape


############## Graph functions ##############


def get_fc_edge_indices(sizes, device):
    """ Get fully connected edge_index for nodes in a batch with the i-th
        element of batch having sizes[i] nodes.
    Args:
        sizes: List with N items with each item being the #nodes in each
            batch element
    Returns:
        edge_index: torch.tensor of shape (2, E) with all directed edges
            all nodes belonging to the same batch element (no self-loops)
    """
    L = [[(s,t) for s, t in permutations(range(sum(sizes[:i]), sum(sizes[:i])+sizes[i]), 2)] for i in range(len(sizes))]
    edge_index = torch.tensor(list(chain.from_iterable(L)), device=device)
    edge_index = edge_index.transpose(0, 1) if edge_index.dim() == 2 else torch.zeros(2, 0, dtype=torch.long, device=device)
    return edge_index


def get_dist_based_edge_mask(node_attr, dist_threshold=0., edge_index=None):
    """ Get distance threshold based edge types.
    Args:
        node_attr: A node_attr tensor of shape (N,2)
        dist_threshold (default = 0): Distance threshold; <0 means all edges,
            =0 means no edges, >0 applies a threshold for edges.
        edge_index: Edge index of shape (2,E). If not provided,
            a fully-connected index of shape (2,N*(N-1)) is assumed.
    Returns:
        edge_mask: Tensor of shape (E,) with binary 0/1 types for
            all directed edges (excluding self-loops)
    """
    N, D = node_attr.shape
    device = node_attr.device
    f_type = torch.cuda.FloatTensor if device.type.startswith('cuda') else torch.FloatTensor
    assert D == 2, "Node attribute must only contain 2 coordinates"

    # Get edge index
    if edge_index is None:
        edge_index = get_fc_edge_indices([N], device)
    E = edge_index.shape[1]

    if E > 0:
        if dist_threshold < 0.:
            edge_mask = torch.ones(E, device=device)
        elif dist_threshold == 0.:
            edge_mask = torch.zeros(E, device=device)
        elif dist_threshold > 0.:
            s, t = edge_index
            # Get pairwise distances
            DM = torch.sqrt(torch.sum((node_attr[s] - node_attr[t]) ** 2, dim=1))
            # Impose distance based threshold
            edge_mask = (DM <= dist_threshold).type(f_type)
    else:
        edge_mask = torch.zeros(E, device=device)
    return edge_mask


############## Helper functions ##############


def postfix_dict_keys(D, postfix, separator=''):
    K = list(D.keys()) # Get key list first to avoid bug (don't add new postfixed keys while reading D.keys() in loop)
    for k in K:
        D[k + separator + postfix] = D.pop(k)
    return D


############## Neural nets ##############


class FQALayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super(FQALayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.n_q = kwargs['n_q']
        self.d_qk = kwargs['d_qk']
        self.d_v = kwargs['d_v']
        self.att_dim = kwargs['att_dim']
        self.n_hk_q = kwargs.get('n_hk_q', 0)
        self.flags = kwargs['flags']

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.k_gen = nn.Sequential(
                        nn.Linear(4*self.input_dim + 4*self.hidden_dim, self.n_q * self.d_qk),
                        Reshape((self.n_q, self.d_qk))
                    )        
        self.q_gen = nn.Sequential(
                        nn.Linear(4*self.input_dim + 4*self.hidden_dim, self.n_q * self.d_qk),
                        Reshape((self.n_q, self.d_qk))
                    )
        self.vy_gen = nn.Sequential(
                        nn.Linear(self.input_dim + self.hidden_dim, (self.hidden_dim + self.input_dim + (self.n_q + self.n_hk_q) * self.d_v)//2),
                        nn.ReLU(),
                        nn.Linear((self.hidden_dim + self.input_dim + (self.n_q + self.n_hk_q) * self.d_v)//2, (self.n_q + self.n_hk_q) * self.d_v),
                        nn.ReLU(),
                        Reshape(((self.n_q + self.n_hk_q), self.d_v))
                    )
        self.vn_gen = nn.Sequential(
                        nn.Linear(self.input_dim + self.hidden_dim, (self.hidden_dim + self.input_dim + (self.n_q + self.n_hk_q) * self.d_v)//2),
                        nn.ReLU(),
                        nn.Linear((self.hidden_dim + self.input_dim + (self.n_q + self.n_hk_q) * self.d_v)//2, (self.n_q + self.n_hk_q) * self.d_v),
                        nn.ReLU(),
                        Reshape(((self.n_q + self.n_hk_q), self.d_v))
                    )
        self.bias = nn.Parameter(torch.zeros(self.n_q), requires_grad=True)
        self.concat = Reshape(((self.n_q + self.n_hk_q) * self.d_v,))
        self.att_gen = nn.Linear((self.n_q + self.n_hk_q) * self.d_v, self.att_dim)
        self.compress_att = nn.Linear(self.att_dim, self.hidden_dim)

    def forward(self, h, m, edge_index, edge_mask, p, **kwargs):
        if 'nointeract' in self.flags:
            return torch.zeros_like(h), {}

        s_idx, r_idx = edge_index
        num_agents = p.shape[0]
        eps = 1e-8

        # Generate sr inputs
        p_s = p[s_idx]
        p_r = p[r_idx]
        h_s = h[s_idx]
        h_r = h[r_idx]
        m_s = m[s_idx, 0] * edge_mask
        m_r = m[r_idx, 0] * edge_mask

        p_sr = p_s - p_r
        h_sr = h_s - h_r
        p_sr_norm = torch.sqrt(torch.sum(p_sr * p_sr, dim=1, keepdim=True)).repeat(1, p_sr.shape[1])
        p_sr_norm[p_sr_norm < eps] = eps
        h_sr_norm = torch.sqrt(torch.sum(h_sr * h_sr, dim=1, keepdim=True)).repeat(1, h_sr.shape[1])
        h_sr_norm[h_sr_norm < eps] = eps
        p_sr_unit = p_sr / p_sr_norm
        h_sr_unit = h_sr / h_sr_norm

        if self.n_hk_q > 0: # Extract velocity estimates if human-knowledge queries are needed
            v = kwargs['v_est']
            v_s = v[s_idx]
            v_r = v[r_idx]
            v_sr = v_s - v_r
            v_sr_norm = torch.sqrt(torch.sum(v_sr * v_sr, dim=1, keepdim=True)).repeat(1, v_sr.shape[1])
            v_sr_norm[v_sr_norm < eps] = eps
            v_sr_unit = v_sr / v_sr_norm

        sr_input = torch.cat((p_s, p_r, p_sr, p_sr_unit, h_s, h_r, h_sr, h_sr_unit), dim=1).detach()
        if 'nodec' in self.flags:
            sr_red_input = torch.cat((p_s, h_s), dim=1)
        else:
            sr_red_input = torch.cat((p_sr, h_s), dim=1)

        # Generate K, Q, Vy, Vn
        K = self.k_gen(sr_input)
        Q = self.q_gen(sr_input)
        Vy = self.vy_gen(sr_red_input)
        Vn = self.vn_gen(sr_red_input)

        # Get if-else decisions
        dec = self.sigmoid(torch.sum(K * Q, dim=2) + self.bias)
        if self.n_hk_q > 0: # Add human-knowledge query decisions
            C = 3.0 # Large constant
            next_p_sr = p_sr + v_sr
            hkq_list = [
                self.sigmoid(C * torch.sum(v_sr_unit * p_sr_unit, dim=1)).detach(),
                self.sigmoid(torch.sum(p_sr * p_sr, dim=1) - 0.2**2).detach(),
                self.sigmoid(torch.sum(next_p_sr * next_p_sr, dim=1) - 0.2**2).detach(),
                self.sigmoid(torch.sum(p_sr * p_sr, dim=1) - 0.4**2).detach(),
                self.sigmoid(torch.sum(next_p_sr * next_p_sr, dim=1) - 0.4**2).detach(),
            ]
            assert self.n_hk_q <= len(hkq_list), "Required number of human-knowledge queries ({}) is more than provided ({})".format(self.n_hk_q, len(hkq_list))
            hkq = torch.stack(hkq_list[:self.n_hk_q], dim=1)
            dec = torch.cat((dec, hkq), dim=1)
        # Apply ablations if flagged
        if 'nodec' in self.flags:
            dec = torch.ones_like(dec)
        dec_expand = dec.unsqueeze(2).repeat(1, 1, self.d_v)

        # Generate responses from decisions
        resp = dec_expand * Vy + (1.0 - dec_expand) * Vn

        # Concatenate and generate attention
        concat_att = self.concat(resp)
        att_sr = self.relu(self.att_gen(concat_att))

        # Aggregate attention from all senders        
        att_m_sr = att_sr * m_s.unsqueeze(1).repeat(1, self.att_dim)
        att, _ = scatter_max(att_m_sr, r_idx, dim=0, dim_size=num_agents, fill_value=0.)

        # Transform from attention dimensions to hidden state dimensions
        att = self.relu(self.compress_att(att))
        # Mask agents not present
        att = att * m[:,0].unsqueeze(1).repeat(1, self.hidden_dim)
        return att, {'dec': dec}


class FQA(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super(FQA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = kwargs['n_layers']
        assert self.n_layers > 0, "Number of layers should be a positive integer."

        self.fqa_layers = nn.ModuleList([FQALayer(self.input_dim, self.hidden_dim, **kwargs) for _ in range(self.n_layers)])

    def forward(self, h, m, edge_index, edge_mask, p, **kwargs):
        _h = h

        s_idx, r_idx = edge_index
        info = {'edge_mask': edge_mask * m[s_idx, 0] * m[r_idx, 0]}
        for i, att_layer in enumerate(self.fqa_layers):
            att, layer_info = att_layer(_h, m, edge_index, edge_mask, p, **kwargs)
            _h = _h + att
            info.update(postfix_dict_keys(layer_info, 'l'+str(i), separator='_'))
        return att, info


class VelPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(VelPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.relu = nn.ReLU()
        self.lin_in1 = nn.Linear(self.input_dim + self.hidden_dim*2, 48)
        self.lin_in2 = nn.Linear(48, self.hidden_dim)
        self.lin_out1 = nn.Linear(self.hidden_dim, 16)
        self.lin_out2 = nn.Linear(16, self.output_dim)

    def forward(self, p, h, att):
        x = torch.cat((p, h, att), dim=1)
        h_next = h + self.lin_in2(self.relu(self.lin_in1(x)))
        y = self.lin_out2(self.relu(self.lin_out1(h_next)))
        return y, h_next

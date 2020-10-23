import pdb
import torch
import torch.nn as nn

import sys
sys.path.append('./')
from models.BaseModel import BaseModel
from models.FQA.utils import FQA, VelPredictor, get_fc_edge_indices, get_dist_based_edge_mask


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__()
        self.state_dim = 4
        self.input_dim = kwargs.get('input_dim', 2)
        self.output_dim = kwargs.get('output_dim', 2)
        self.hidden_dim = kwargs.get('hidden_dim', 32)
        self.dist_threshold = kwargs.get('dist_threshold', -1.0)
        self.rtype = kwargs.get('rtype', 'lstm')
        self.attention_params = kwargs.get('attention_params', {})
        self.use_vel = kwargs.get('use_vel', True)
        self.use_mask = kwargs.get('use_mask', False)

        self.FQA_block = FQA(self.input_dim, self.hidden_dim, **self.attention_params)
        self.vel_predictor = VelPredictor(self.input_dim, self.hidden_dim, self.output_dim)

        if self.rtype == 'lstm':
            self.rnn_cell = nn.LSTMCell(self.input_dim, self.hidden_dim)
        elif self.rtype == 'gru':
            self.rnn_cell = nn.GRUCell(self.input_dim, self.hidden_dim)
        elif self.rtype == 'rnn_tanh':
            self.rnn_cell = nn.RNNCell(self.input_dim, self.hidden_dim, nonlinearity='tanh')
        elif self.rtype == 'rnn_relu':
            self.rnn_cell = nn.RNNCell(self.input_dim, self.hidden_dim, nonlinearity='relu')

    def get_FQA_vel(self, s, h, m, sizes):
        # Generate edges
        edge_index = get_fc_edge_indices(sizes, s.device)
        edge_mask = get_dist_based_edge_mask(s[:,:2], self.dist_threshold, edge_index)
        # Compute multi-agent attention
        att, info = self.FQA_block(h, m, edge_index, edge_mask, s[:,:2], v_est=s[:,2:])
        # Update h and predict velocities for nodes
        vel, h = self.vel_predictor(s[:,:2], h, att)
        return vel, h, info

    def forward_one_step(self, x, s, h, m, m_prev, **kwargs):
        # Extract sizes of batch elements
        sizes = kwargs.get('sizes', [s.shape[0]])
        c = kwargs.get('c', None)
        
        # Compute temporary updated state
        s_temp = torch.zeros_like(s)
        v_update = m_prev * m
        s_temp[:,:2] = x * m + s[:,:2] * (1 - m)
        s_temp[:,2:] = (x - s[:,:2]) * v_update + s[:,2:] * (1 - v_update)

        # Compute updated hidden state
        if self.rtype == 'lstm':
            h_next, c_next = self.rnn_cell(s_temp[:,:2], (h, c))
        else:
            h_next = self.rnn_cell(s_temp[:,:2], h)

        # Incorporate interactions
        del_v, h_next, info = self.get_FQA_vel(s_temp, h_next, m, sizes)

        # Predict s_next and del_x
        del_x = del_v * v_update
        if self.use_vel:
            del_x = del_x + s_temp[:,2:]
        s_next = torch.zeros_like(s_temp)
        s_next[:,:2] = s_temp[:,:2]
        s_next[:,2:] = del_x
        if self.rtype == 'lstm':
            return del_x, s_next, h_next, c_next, info
        else:
            return del_x, s_next, h_next, info

    def forward(self, sources, masks, sizes, burn_in_steps=-1):
        N, T, D = sources.shape
        assert D == 2, "D must be 2"
        device = sources.device

        # Initial state
        s = torch.zeros(N, self.state_dim, device=device)
        h = torch.zeros(N, self.hidden_dim, device=device)
        if self.rtype == 'lstm':
            c = torch.zeros(N, self.hidden_dim, device=device)
        m_prev = torch.zeros_like(sources[:, 0, :])
        m = m_prev

        # Loop over time
        preds = []
        infos = []
        for t in range(T):
            # Get input
            if burn_in_steps <= 0 or t < burn_in_steps:
                if self.use_mask:
                    m_prev = m
                    m = masks[:, t, :]
                else:
                    m_prev = torch.ones(N, D, device=device)
                    m = m_prev
                x = sources[:, t, :]
            else:
                if self.use_mask:
                    m_prev = m
                    m = masks[:, burn_in_steps, :]
                else:
                    m_prev = torch.ones(N, D, device=device)
                    m = m_prev
                x = y

            # Take one forward step
            if self.rtype == 'lstm':
                del_x, s, h, c, info = self.forward_one_step(x, s, h, m, m_prev, sizes=sizes, c=c)
            else:                
                del_x, s, h, info = self.forward_one_step(x, s, h, m, m_prev, sizes=sizes)
            
            # Predict output (rel_out assumed)
            y = x + del_x
            preds.append(y)
            infos.append(info)

        # Process infos
        info_dict = {}
        for k in infos[0].keys():
            info_dict[k] = torch.stack([infos[t][k] for t in range(T)], dim=1)
        return torch.stack(preds, dim=1), info_dict


class FQAModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(FQAModel, self).__init__(Net, *args, **kwargs)

    def get_batch_losses(self, preds, targets, *args, **kwargs):
        s_masks = kwargs.get('s_masks', torch.ones_like(preds).to(self.device))
        t_masks = kwargs.get('t_masks', torch.ones_like(preds).to(self.device))
        mse_loss = (self.mse_loss_fn(preds, targets) * s_masks * t_masks).sum()
        mae_loss = (self.mae_loss_fn(preds, targets) * s_masks * t_masks).sum()
        numel = (s_masks * t_masks).sum()
        losses = {
            'loss': {'numel': numel, 'val': mse_loss},
            'mse_loss': {'numel': numel, 'val': mse_loss},
            'mae_loss': {'numel': numel, 'val': mae_loss},
        }
        return losses

    def run_batch(self, batch, mode='train', store_losses=True, store_debug=False):
        # Do mode-based setup
        if mode == 'train':
            self.net.train()
            torch.set_grad_enabled(True)
        elif mode == 'eval':
            self.net.eval()
            torch.set_grad_enabled(False)

        # Batchify tensors
        sources = torch.cat([d[0] for d in batch], 0).to(self.device)
        targets = torch.cat([d[1] for d in batch], 0).to(self.device)
        s_masks = torch.cat([d[2] for d in batch], 0).to(self.device)
        t_masks = torch.cat([d[3] for d in batch], 0).to(self.device)
        sizes = [d[0].size(0) for d in batch]

        # Get burn_in_steps
        seq_length = sources.size(1)
        burn_in_steps = self.get_burn_in_steps(seq_length, mode=mode)

        # Make prediction
        preds, info_dict = self.net(sources, masks=s_masks, sizes=sizes, burn_in_steps=burn_in_steps)

        # Compute loss
        losses = self.get_batch_losses(preds, targets, s_masks=s_masks, t_masks=t_masks)

        # Do backprop
        if mode == 'train':
            self.optimizer.zero_grad()
            mean_loss = losses['loss']['val'] / losses['loss']['numel']
            mean_loss.backward()
            self.optimizer.step()

        # Unbatchify preds from 3D tensor to list of 2D tensors
        preds = torch.split(preds, sizes, dim=0)

        # Convert losses to float values
        for loss, entry in losses.items():
            losses[loss] = {k: v.detach().cpu().numpy().item() for k, v in losses[loss].items()}

        # Return losses and preds
        batch_info = {}
        if store_losses:
            batch_info['losses'] = losses
        if store_debug:
            batch_info['debug'] = {
                'preds': preds,
                'sizes': sizes
            }
            batch_info['debug'].update(info_dict)
        return batch_info

Model = FQAModel
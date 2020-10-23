import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./')
from models.BaseModel import BaseModel


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__()
        self.input_size = kwargs.get('input_size', 2)
        self.embed_size = kwargs.get('embed_size', 64)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.output_size = kwargs.get('output_size', 2)
        self.rel_out = kwargs.get('rel_out', True)
        self.rtype = kwargs.get('rtype', 'lstm')
        self.use_mask = kwargs.get('use_mask', False)

        self.input_layer = nn.Linear(self.input_size, self.embed_size)
        if self.rtype == 'lstm':
            self.rnn_cell = nn.LSTMCell(self.embed_size, self.hidden_size)
        elif self.rtype == 'gru':
            self.rnn_cell = nn.GRUCell(self.embed_size, self.hidden_size)
        elif self.rtype == 'rnn_tanh':
            self.rnn_cell = nn.RNNCell(self.embed_size, self.hidden_size, nonlinearity='tanh')
        elif self.rtype == 'rnn_relu':
            self.rnn_cell = nn.RNNCell(self.embed_size, self.hidden_size, nonlinearity='relu')
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward_one_step(self, x_t, h_t, c_t=None, **kwargs):
        embed = F.relu(self.input_layer(x_t))
        if self.rtype == 'lstm':
            h_t, c_t = self.rnn_cell(embed, (h_t, c_t))
            y_t = self.output_layer(h_t)
            return y_t, h_t, c_t
        else:
            h_t = self.rnn_cell(embed, h_t)
            y_t = self.output_layer(h_t)
            return y_t, h_t

    def forward(self, x, **kwargs):
        burn_in_steps = kwargs.get('burn_in_steps', -1)
        mask = kwargs.get('mask', None) if self.use_mask else None
        num_agents, seq_length, _ = x.size()

        h_t = torch.zeros(num_agents, self.hidden_size, device=x.device)
        if self.rtype == 'lstm':
            c_t = torch.zeros(num_agents, self.hidden_size, device=x.device)

        pred = []
        for t in range(seq_length):
            if burn_in_steps <= 0 or t < burn_in_steps:
                if self.use_mask:
                    x_t = torch.cat((x[:,t,:], mask[:,t,:]), dim=1)
                else:
                    x_t = x[:,t,:]
            else:
                if self.use_mask:
                    x_t = torch.cat((y_t, x_t[:,2:]), dim=1)
                else:
                    x_t = y_t

            # Take one step forward
            if self.rtype == 'lstm':
                y_t, h_t, c_t = self.forward_one_step(x_t, h_t, c_t)
            else:
                y_t, h_t = self.forward_one_step(x_t, h_t)

            if self.rel_out:
                if self.use_mask:
                    y_t = x_t[:,:2] + y_t
                else:
                    y_t = x_t + y_t

            pred.append(y_t)

        return torch.stack(pred, dim=1)


class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(Net, *args, **kwargs)

    def get_batch_losses(self, pred, target, *args, **kwargs):
        s_mask = kwargs.get('s_mask', torch.ones_like(pred).to(self.device))
        t_mask = kwargs.get('t_mask', torch.ones_like(pred).to(self.device))
        mse_loss = (self.mse_loss_fn (pred, target) * s_mask * t_mask).sum()
        mae_loss = (self.mae_loss_fn (pred, target) * s_mask * t_mask).sum()
        numel = (s_mask * t_mask).sum()
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

        source = torch.cat([d[0] for d in batch], 0).to(self.device)
        target = torch.cat([d[1] for d in batch], 0).to(self.device)
        s_mask = torch.cat([d[2] for d in batch], 0).to(self.device)
        t_mask = torch.cat([d[3] for d in batch], 0).to(self.device)

        sizes = [d[0].size(0) for d in batch]

        # Get burn_in_steps
        seq_length = source.size(1)
        burn_in_steps = self.get_burn_in_steps(seq_length, mode=mode)

        # Make prediction
        pred = self.net(source, mask=s_mask, burn_in_steps=burn_in_steps)

        # Compute loss
        losses = self.get_batch_losses(pred, target, s_mask=s_mask, t_mask=t_mask)

        # Do backprop
        if mode == 'train':
            self.optimizer.zero_grad()
            mean_loss = losses['loss']['val'] / losses['loss']['numel']
            mean_loss.backward()
            self.optimizer.step()

        # Re-format pred from 3D tensor to list of 2D tensors
        pred = torch.split(pred, sizes, dim=0)

        # Convert losses to float values
        for loss, entry in losses.items():
            losses[loss] = {k: v.detach().cpu().numpy().item() for k, v in losses[loss].items()}

        # Return losses and preds
        batch_info = {}
        if store_losses:
            batch_info['losses'] = losses
        if store_debug:
            batch_info['debug'] = {'preds': pred}
        return batch_info

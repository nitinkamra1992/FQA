import pdb
import torch
import torch.nn as nn

import sys
sys.path.append('./')
from models.BaseModel import BaseModel


class Net(nn.Module):
    ''' Inertia model
    '''
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__()
        self.use_mask = kwargs.get('use_mask', False)
        self.state_dim = 4
        self.A = nn.Parameter(torch.tensor([[0., 0., -1., 0.],
                                            [0., 0., 0., -1.],
                                            [0., 0., 0., 0.],
                                            [0., 0., 0., 0.]]), requires_grad=False)
        self.B = nn.Parameter(torch.tensor([[1., 0., 1., 0.],
                                            [0., 1., 0., 1.]]), requires_grad=False)
        self.C = nn.Parameter(torch.tensor([[0., 0.],
                                            [0., 0.],
                                            [1., 0.],
                                            [0., 1.]]), requires_grad=False)

    def forward_one_step(self, x, s, s_app, **kwargs):
        # Compute updated state
        s_temp = torch.mm(s, self.A) + torch.mm(x, self.B)
        s_next = torch.zeros_like(s)
        s_next[:,:2] = s_temp[:,:2]
        s_next[:,2:4] = s[:,2:4] * s_app + s_temp[:,2:4] * (1 - s_app)
        # Predict next step
        del_x = torch.mm(s_next, self.C)
        return del_x, s_next

    def forward(self, source, mask, burn_in_steps=-1):
        N, T, D = source.shape
        device = source.device
        # Initial state
        s = torch.zeros(N, self.state_dim, device=device)
        s_app = torch.zeros_like(source[:, 0, :])
        m_prev = torch.zeros_like(source[:, 0, :])
        m = m_prev
        # Loop over time
        pred = []
        for t in range(T):
            # Get input
            if burn_in_steps <= 0 or t < burn_in_steps:
                if self.use_mask:
                    m_prev = m
                    m = mask[:, t, :]
                    s_app = (1 - m_prev) * m
                x = source[:, t, :]
            else:
                if self.use_mask:
                    m_prev = m
                    s_app = (1 - m_prev) * m
                x = y

            # Take one forward step
            del_x, s = self.forward_one_step(x, s, s_app)
            
            # Predict output
            y = x + del_x
            pred.append(y)
        return torch.stack(pred, dim=1)


class InertiaModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(InertiaModel, self).__init__(Net, *args, **kwargs)

    def get_batch_losses(self, preds, targets, *args, **kwargs):
        B = len(preds)
        default_mask = [torch.ones_like(preds[i]).to(self.device) for i in range(B)]
        s_masks = kwargs.get('s_masks', default_mask)
        t_masks = kwargs.get('t_masks', default_mask)
        
        mse_loss = torch.zeros(1, device=self.device)
        mae_loss = torch.zeros(1, device=self.device)
        numel = 0
        for pred, target, s_mask, t_mask in zip(preds, targets, s_masks, t_masks):
            mse_loss += (self.mse_loss_fn(pred, target) * s_mask * t_mask).sum()
            mae_loss += (self.mae_loss_fn(pred, target) * s_mask * t_mask).sum()
            numel += (s_mask * t_mask).sum()

        losses = {
            'loss': {'numel': numel, 'val': mse_loss},
            'mse_loss': {'numel': numel, 'val': mse_loss},
            'mae_loss': {'numel': numel, 'val': mae_loss},
        }
        return losses

    def run_batch(self, data, mode='train', store_losses=True, store_debug=False):
        # Do mode-based setup
        if mode == 'train':
            self.net.train()
            torch.set_grad_enabled(True)
        elif mode == 'eval':
            self.net.eval()
            torch.set_grad_enabled(False)

        # Looping over batch elements
        B = len(data)
        sources = []
        targets = []
        s_masks = []
        t_masks = []
        preds = []
        for d in data:
            # Extract sources, targets and masks
            source = d[0].to(self.device)
            target = d[1].to(self.device)
            s_mask = d[2].to(self.device)
            t_mask = d[3].to(self.device)

            sources.append(source)
            targets.append(target)
            s_masks.append(s_mask)
            t_masks.append(t_mask)

            # Get prediction
            seq_length = source.size(1)
            burn_in_steps = self.get_burn_in_steps(seq_length, mode=mode)
            pred = self.net(source, s_mask, burn_in_steps=burn_in_steps)
            preds.append(pred)

        # Compute batch loss
        losses = self.get_batch_losses(preds, targets, s_masks=s_masks, t_masks=t_masks)

        # No backprop needed

        # Convert losses to float values
        for loss, entry in losses.items():
            losses[loss] = {k: v.detach().cpu().numpy().item() for k, v in losses[loss].items()}

        # Return losses and preds
        batch_info = {}
        if store_losses:
            batch_info['losses'] = losses
        if store_debug:
            batch_info['debug'] = {'preds': preds}
        return batch_info

Model = InertiaModel
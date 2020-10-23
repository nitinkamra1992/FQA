import pdb
import time
import importlib.util
import os
import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import sys
sys.path.append('./')

from utils.model_utils import optim_list, count_parameters
from utils.misc_utils import create_directory, get_by_dotted_path, add_record, get_records, log_record_dict
from utils.plot_utils import create_curve_plots


############################ Base model class ###########################


class BaseModel(object):

    def __init__(self, Net, device, global_records, config):
        # Initializations
        self.device = device
        self.global_records = global_records
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize network
        self.net = Net(**self.config['net'])

        # Then load its params if available
        if self.config['net'].get('saved_params_path', None) is not None:
            self.load_net(self.config['net']['saved_params_path'])

        # Initialize optimizer
        self.setup_optimizer()

        # Initialize learning rate scheduler
        self.setup_lr_scheduler()

        # Transfer network to device
        self.net.to(self.device)
        self.logger.info(self.net)
        self.logger.info("Number of parameters: %d" % (count_parameters(self.net)))

        # Losses for all models (more can be defined in derived models if needed)
        self.mse_loss_fn = nn.MSELoss(reduction='none')
        self.mae_loss_fn = nn.L1Loss(reduction='none')

        # Initialize epoch number
        self.epoch = 0

    # Abstract method to implement
    def run_batch(self, batch, mode='train', store_losses=True, store_debug=False):
        raise NotImplementedError()

    def setup_optimizer(self):
        self.optimizer = None
        if 'optimizer' in self.config:
            optim = optim_list[self.config['optimizer']['name']]
            self.optimizer = optim(self.net.parameters(), **self.config['optimizer']['params'])

    def setup_lr_scheduler(self):
        self.scheduler = None
        if self.config['optimizer'].get('scheduler'):
            self.scheduler = lr_scheduler.StepLR(self.optimizer, **self.config['optimizer']['scheduler'])
            self.logger.debug('Using LR scheduler: '+ str(self.config['optimizer']['scheduler']))

    def step_lr_scheduler(self):
        if self.scheduler:
            self.scheduler.step()
            self.logger.debug("Learning rate: %s" % (','.join([str(lr) for lr in self.scheduler.get_lr()])))

    def fit(self, tr_loader, val_loader, *args, **kwargs):
        # Initialize params
        if 'max_epoch' in kwargs:
            max_epoch = kwargs['max_epoch']
        else:
            assert 'max_epoch' in self.config['train']['stop_crit'], "max_epoch not specified in config['train']['stop_crit']"
            max_epoch = self.config['train']['stop_crit']['max_epoch']

        if 'min_epoch' in kwargs:
        	min_epoch = kwargs['min_epoch']
        elif 'min_epoch' in self.config['train']['stop_crit']:
        	min_epoch = self.config['train']['stop_crit']['min_epoch']
        else:
        	min_epoch = max_epoch // 2

        if 'max_patience' in kwargs:
            max_patience = kwargs['max_patience']
        elif 'max_patience' in self.config['train']['stop_crit']:
            max_patience = self.config['train']['stop_crit']['max_patience']
        else:
            max_patience = None
        if max_patience is not None:
            assert max_patience > 0, "max_patience should be positive"
            self.logger.debug('Early stopping enabled with max_patience = {}'.format(max_patience))
        else:
            self.logger.debug('Early stopping disabled since max_patience not specified.')

        # Train epochs
        best_valid_loss = np.inf
        best_valid_epoch = 0
        early_break = False
        for epoch in range(max_epoch):
            self.logger.info('\n' + 40 * '%' + '  EPOCH {}  '.format(epoch) + 40 * '%')
            self.epoch = epoch

            # Perform LR scheduler step
            self.step_lr_scheduler()

            # Run train epoch
            t = time.time()
            epoch_records = self.run_epoch(tr_loader, 'train', epoch, *args, **kwargs)

            # Log and print train epoch records
            log_record_dict('train', epoch_records, self.global_records)
            self.print_record_dict(epoch_records, 'Train', time.time() - t)
            self.global_records['result'].update({
                'final_train_loss': epoch_records['loss'],
                'final_train_epoch': epoch
            })

            if val_loader is not None:
                # Run valid epoch
                t = time.time()
                epoch_records = self.run_epoch(val_loader, 'eval', epoch, *args, **kwargs)

                # Log and print valid epoch records
                log_record_dict('valid', epoch_records, self.global_records)
                self.print_record_dict(epoch_records, 'Valid', time.time() - t)
                self.global_records['result'].update({
                    'final_valid_loss': epoch_records['loss'],
                    'final_valid_epoch': epoch
                })

                # Check for early-stopping
                if epoch_records['loss'] < best_valid_loss:
                    best_valid_loss = epoch_records['loss']
                    best_valid_epoch = epoch
                    self.global_records['result'].update({
                        'best_valid_loss': best_valid_loss,
                        'best_valid_epoch': best_valid_epoch
                    })
                    self.logger.info('    Best validation loss improved to {:.8f}'.format(best_valid_loss))
                    self.save_net(os.path.join(self.config['outdir'], 'best_valid_params.ptp'))

                if (epoch > min_epoch) and (max_patience is not None) and (best_valid_loss < np.min(get_records('valid.loss', self.global_records)[-max_patience:])):
                    early_break = True

            # Produce plots
            plots = self._plot_helper(epoch_records) # Needs epoch_records for names of logged losses
            if plots is not None:
                for k, v in plots.items():
                    create_curve_plots(k, v, self.config['outdir'])

            # Save net
            self.save_net(os.path.join(self.config['outdir'], 'final_params.ptp'))

            # Save results
            pickle.dump(self.global_records, file=open(os.path.join(self.config['outdir'], self.config['record_file']), 'wb'))

            # Early-stopping
            if early_break:
                self.logger.warning('Early Stopping because validation loss did not improve for {} epochs'.format(max_patience))
                break

    def run_epoch(self, data_loader, mode, epoch, *args, **kwargs):
        epoch_losses = {}
        num_batches = len(data_loader)
        
        # Iterate over batches
        for batch_idx, batch in enumerate(data_loader):
            # Eval options
            store_losses = self.config['eval'].get('store_losses', True)
            store_debug = self.config['eval'].get('store_debug', False)
            
            # Run the batch
            batch_info = self.run_batch(batch, mode, store_losses, store_debug)
            batch_losses = batch_info['losses'] if store_losses else None
            batch_debug = batch_info['debug'] if store_debug else None
            
            # Log stuff
            log = self.config['log_interval']
            if batch_idx % log == 0:
                loss_vals = ''
                if batch_losses is not None:
                    for loss in batch_losses:
                        mean_loss = batch_losses[loss]['val'] / batch_losses[loss]['numel'] if batch_losses[loss]['numel'] != 0 else batch_losses[loss]['val']
                        loss_vals = loss_vals + ', {}: {:.8f}'.format(loss, mean_loss)
                self.logger.debug('{} epoch: {} [{}/{} ({:.0f}%)]{}'.format(
                        mode, epoch, (batch_idx + 1), num_batches,
                        100.0 * (batch_idx + 1.0) / num_batches, loss_vals))

            # Populate epoch losses
            for k, v in batch_losses.items():
                if k in epoch_losses:
                    epoch_losses[k]['val'] += v['val']
                    epoch_losses[k]['numel'] += v['numel']
                else:
                    epoch_losses[k] = v

            # Save preds to files
            if batch_debug:
                save_dict = {}
                debug_dir = os.path.join(self.config['outdir'], 'debug')
                create_directory(debug_dir)
                save_path = os.path.join(debug_dir, 'epoch{}_batch{}.npz'.format(epoch, batch_idx))

                # Store source, target, smask and tmask per batch element
                for i in range(len(batch)):
                    save_dict[str(i) + '_source'] = np.copy(batch[i][0].detach().cpu().numpy())
                    save_dict[str(i) + '_target'] = np.copy(batch[i][1].detach().cpu().numpy())
                    save_dict[str(i) + '_smask'] = np.copy(batch[i][2].detach().cpu().numpy())
                    save_dict[str(i) + '_tmask'] = np.copy(batch[i][3].detach().cpu().numpy())
                # Store debug data
                for k, v in batch_debug.items():                    
                    if type(v) in [list, tuple]: # assume that v has items per-batch-element
                        assert len(v) == len(batch), "Only per-batch-element lists are allowed for debug tensors"
                        for i in range(len(v)):
                            try: # v[i] is tensor
                                save_dict[str(i) + '_' + k] = np.copy(v[i].detach().cpu().numpy())
                            except: # v[i] is not a tensor
                                save_dict[str(i) + '_' + k] = v[i]
                    else: # assume a single item for the whole batch
                        try: # v is a tensor
                            save_dict[k] = np.copy(v.detach().cpu().numpy())
                        except: # v is not a tensor
                            save_dict[k] = v
                np.savez_compressed(save_path, **save_dict)

        # Return epoch records
        epoch_records = {}
        for k, v in epoch_losses.items():
            epoch_records[k] = v['val'] if v['numel'] == 0. else v['val'] / float(v['numel'])
        return epoch_records

    def evaluate(self, data_loader, *args, **kwargs):
        # Run eval
        t = time.time()
        epoch_records = self.run_epoch(data_loader, 'eval', 0, *args, **kwargs)

        # Log and print epoch records
        log_record_dict('Eval', epoch_records, self.global_records)
        self.print_record_dict(epoch_records, 'Eval', time.time() - t)
        self.global_records['result'].update({
            'loss': epoch_records['loss'],
        })

    def save_net(self, filename):
        torch.save(self.net.state_dict(), filename)
        self.logger.info('params saved to {}'.format(filename))

    def load_net(self, filename):
        self.logger.info('Loading params from {}'.format(filename))
        self.net.load_state_dict(torch.load(filename), strict=False)

    def print_record_dict(self, record_dict, usage, t_taken):
        loss_str = ''
        for k, v in record_dict.items():
            loss_str = loss_str + ' {}: {:.8f}'.format(k, v)
        self.logger.info('{}:{} took {:.3f}s'.format(
                usage, loss_str, t_taken))

    def _plot_helper(self, record_dict):
        plots = {}
        for loss in record_dict.keys():
            plots[loss] = {
                'train': get_records('train.' + loss, self.global_records),
                'valid': get_records('valid.' + loss, self.global_records)
            }
        return plots

    def get_burn_in_steps(self, seq_length, mode='train'):
        """ Linearly decrease burn_in_steps each epoch (if dynamic_burn_in allowed) """
        burn_in_steps = self.config[mode].get('burn_in_steps', -1)
        dynamic_burn_in = self.config[mode].get('dynamic_burn_in', mode=='train')
        if dynamic_burn_in and burn_in_steps > 0:
            burn_in_steps = max(burn_in_steps, seq_length - self.epoch)

        return burn_in_steps

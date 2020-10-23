import pdb
import os
import time
import numpy as np
import argparse
import pickle

import sys
sys.path.append('./')
from utils.misc_utils import create_directory


########################## Helper functions ###########################


def get_losses(pred, target, s_mask, t_mask, *args, **kwargs):
    br = kwargs['burn_in_steps']
    no_filter = kwargs.get('no_filter', False)
    mask = s_mask * t_mask

    # Filtering on agents
    if not no_filter:
        m = mask[:,:,0]
        # Remove agents present for less than half the trajectory
        m[np.sum(m, axis=1) / m.shape[1] < 1/2] = 0
        # Remove agents not present at time == br - 3
        if br >= 3:
            m[m[:,br-3] == 0] = 0
        mask = np.tile(np.expand_dims(m,2), (1,1,2))

    # MSE and MAE
    mse_loss = np.sum(((pred - target) * mask)**2)
    mae_loss = np.sum(np.abs(pred - target) * mask)
    numel = mask.sum() / mask.shape[-1]

    # Pre and post burn-in MSE and MAE
    if br > 0:
        pre_pred = pred[:,:br]
        pre_target = target[:,:br]
        pre_mask = mask[:,:br]
        pre_mse_loss = np.sum(((pre_pred - pre_target) * pre_mask)**2)
        pre_mae_loss = np.sum(np.abs(pre_pred - pre_target) * pre_mask)
        pre_numel = pre_mask.sum() / pre_mask.shape[-1]

        post_pred = pred[:,br:]
        post_target = target[:,br:]
        post_mask = mask[:,br:]
        post_mse_loss = np.sum(((post_pred - post_target) * post_mask)**2)
        post_mae_loss = np.sum(np.abs(post_pred - post_target) * post_mask)
        post_numel = post_mask.sum() / post_mask.shape[-1]
    else:
        pre_mse_loss = mse_loss
        pre_mae_loss = mae_loss
        pre_numel = numel

        post_mse_loss = 0.
        post_mae_loss = 0.
        post_numel = 0

    # FDE
    fde_cols = mask.shape[1] - 1 - mask[:,::-1,0].argmax(axis=1)
    fde_rows = range(mask.shape[0])
    fde_loss = np.sum(np.sqrt(np.sum(((pred[fde_rows, fde_cols, :] - target[fde_rows, fde_cols, :]) * mask[fde_rows, fde_cols, :])**2, axis=1)))
    numel_fde = np.sum(mask[fde_rows, fde_cols, :])

    losses = {
        'mse_loss': {'numel': numel, 'val': mse_loss},
        'mae_loss': {'numel': numel, 'val': mae_loss},
        'pre_mse_loss': {'numel': pre_numel, 'val': pre_mse_loss},
        'pre_mae_loss': {'numel': pre_numel, 'val': pre_mae_loss},
        'post_mse_loss': {'numel': post_numel, 'val': post_mse_loss},
        'post_mae_loss': {'numel': post_numel, 'val': post_mae_loss},
        'fde_loss': {'numel': numel_fde, 'val': fde_loss}
    }
    return losses


########################## Evaluation fn ###########################


def eval(args):
    # Assign and create directories
    if args.predsdir is None:
        args.predsdir = os.path.join(args.datadir, 'debug')
    if args.configpkl is None:
        args.configpkl = os.path.join(args.datadir, 'config.pkl')
    if args.outfile is None:
        args.outfile = os.path.join(args.datadir, 'metrics.pkl')
    else:
        create_directory(os.path.dirname(args.outfile)) # Creates only if it does not already exist
 
    # Read config pkl
    with open(args.configpkl, 'rb') as f:
        config = pickle.load(f)

    # Get dataset de-normalizing scale
    if args.scalefile is None:
        args.scalefile = os.path.join(os.path.dirname(config['datapath']), 'scale.txt')
    with open(args.scalefile) as f:
        scale = float(f.readline())

    # Get batch size
    if config['mode'] == 'eval':
        batch_size = config[config['eval'].get('usage', 'test')].get('batch_size', -1)
    else: # config['mode'] == 'train'
        batch_size = config[config['mode']].get('batch_size', -1)

    # Get burn_in_steps
    burn_in_steps = config[config['mode']].get('burn_in_steps',-1)
    if burn_in_steps is None:
        burn_in_steps = -1

    # Dictionary for summing up losses
    total_losses = {
        'mse_loss': {'numel': 0, 'val': 0.},
        'mae_loss': {'numel': 0, 'val': 0.},
        'pre_mse_loss': {'numel': 0, 'val': 0.},
        'pre_mae_loss': {'numel': 0, 'val': 0.},
        'post_mse_loss': {'numel': 0, 'val': 0.},
        'post_mae_loss': {'numel': 0, 'val': 0.},
        'fde_loss': {'numel': 0, 'val': 0.}
    }
    # Parse the preds directory batchwise
    for preds_file in os.listdir(args.predsdir):
        preds_npz = np.load(os.path.join(args.predsdir, preds_file))
        # Eval each sequence
        for i in range(batch_size):
            try:
                source = preds_npz[str(i) + '_source']
                target = preds_npz[str(i) + '_target']
                pred = preds_npz[str(i) + '_preds']
                s_mask = preds_npz[str(i) + '_smask']
                t_mask = preds_npz[str(i) + '_tmask']

                # Get losses for this sequence
                losses = get_losses(pred * scale, target * scale, s_mask, t_mask,
                                burn_in_steps=burn_in_steps, no_filter=args.no_filter)
                # Accumulate in total_losses
                for k in total_losses:
                    total_losses[k]['numel'] += losses[k]['numel']
                    total_losses[k]['val'] += losses[k]['val']
            except KeyError as e:
                print('WARNING: ' + str(e))

    # Final losses
    final_losses = {}
    for k in total_losses.keys():
        final_losses[k] = total_losses[k]['val'] / total_losses[k]['numel'] if total_losses[k]['numel'] > 0 else total_losses[k]['val']
    final_losses['rmse_loss'] = np.sqrt(final_losses['mse_loss'])
    final_losses['pre_rmse_loss'] = np.sqrt(final_losses['pre_mse_loss'])
    final_losses['post_rmse_loss'] = np.sqrt(final_losses['post_mse_loss'])

    # Print and store to file
    print(final_losses)
    with open(args.outfile, 'wb') as f:
        pickle.dump(final_losses, file=f)


############################## Main code ###############################


if __name__ == '__main__':
    # Initial time
    t_init = time.time()

    # Create parser
    parser = argparse.ArgumentParser(description='Generate final evaluation metrics')

    # Add arguments
    parser.add_argument('-d', '--datadir',
        help='Top-level directory where eval results are stored')
    parser.add_argument('-p', '--predsdir',
        default=None,
        help='Directory where preds are stored (default: <datadir>/debug/)')
    parser.add_argument('-o', '--outfile',
        default=None,
        help='Output pickle file path (default: <datadir>/metrics.pkl)')
    parser.add_argument('-c', '--configpkl',
        default=None,
        help='Config file in pkl format (default: <datadir>/config.pkl)')
    parser.add_argument('-s', '--scalefile',
        default=None,
        help='File with dataset scale (default: Checks dataset directory extracted from config for "scale.txt")')
    parser.add_argument('-nf','--no-filter',
        help='''If not specified, the script filters out agents for which either:
                       (1) trajectory_len < (1/2)*(episode_len), or
                       (2) not present at time=burn_in_steps-3''', action='store_true')

    # Parse arguments
    args = parser.parse_args()

    # Call the viz method
    eval(args)

    # Final time
    t_final = time.time()
    print('Program finished in {} secs.'.format(t_final - t_init))
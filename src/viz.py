import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import os
import time
import numpy as np
import argparse
import pickle

import sys
sys.path.append('./')
from utils.misc_utils import create_directory

########################## Inputs ######################################

mode_dir = 'eval_test'
models = [
          'VanillaLSTM/run1', 
          'FQA/AEdge/run1',
         ]
model_names = ['VLSTM', 'FQA']

########################## Visualization fns ###########################


def viz_traj(source, target, pred, mask=None, metadata=None):
    """ Visualize trajectories of entities
    """
    viz_at_time_t(source, target, pred, None, mask, metadata)

def generate_points(source, target):
    (x,y) = source
    (u,v) = target - source
    return [x,y,u,v]

def viz_at_time_t(source, target, pred, time=None, mask=None,
                  metadata={'args':None, 'burn_in_steps':-1, 'savefile':None}):
    """ Plots entity trajectory/location
    
    Args:
        source(N, T, D): ground-truth location array for time t=t
                         where N = Number of agents, T = Trajectory length,
                         D = Number of dimentions in entity's location        
        target (N, T, D): expected ground-truth location array for time t=t+1
        pred (N, T, D): predicted location array for time t=t+1
        time (int): if None, generate plots for entire trajectory, else generate plots for t=time
        mask(N, T, D): mask array
        metadata(dict): metadata.args: argument dict
                        metadata.burn_in_steps: burn_in_steps used for current dataset
                        metadata.savefile: file name to save plots
    """
    N, T, D = source.shape
    args = metadata['args']
    fig = metadata['fig']
    ax_idx = metadata['ax_idx']
    total_ax = metadata['total_ax']
    c_target = c_pred = metadata['color']

    # Open and configure new figure
    ax = fig.add_subplot(1,total_ax, ax_idx)
    #ax.set_title(metadata['ax_title'], fontdict={'fontsize':15}, y=-0.01) #alernative
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_xlabel(metadata['ax_title'], fontdict={'fontsize':15})
    
    # Generate per entity quiver plot
    for n in range(N):
        # Pick up mask
        mask_n = mask[n,:,0]
        
        if (not args.no_filter) and (time is None):
            if np.sum(mask_n) / mask_n.shape[0] < 1/2:
                continue
            br = metadata['burn_in_steps']
            if br >=0 and (br < 3 or mask_n[br-3] == 0):
                continue
        
        # Generate trajectory for entity = n
        gt, predt = [], []
        if time is None:
            for t in range(T):
                if mask_n[t] != 0:
                    gt.append(generate_points(source[n,t,:], target[n,t,:]))
                    if metadata['burn_in_steps'] <= 0 or t < metadata['burn_in_steps']:
                        predt.append(generate_points(source[n,t,:], pred[n,t,:]))
                    else:
                        predt.append(generate_points(pred[n,t-1,:], pred[n,t,:]))

        # Plots for n-th entity
        gt, predt = np.array(gt), np.array(predt)
        if gt.shape[0] == 0:
            pt_x, pt_y, pred_x, pred_y = np.array([]), np.array([]), np.array([]), np.array([])
            gt_x, gt_y, target_x, target_y = np.array([]), np.array([]), np.array([]), np.array([])
        else:
            pt_x, pt_y, pred_x, pred_y = predt[:,0], predt[:,1], predt[:,2], predt[:,3]
            gt_x, gt_y, target_x, target_y = gt[:,0], gt[:,1], gt[:,2], gt[:,3]

        br = int(sum(mask_n[:metadata['burn_in_steps']]))
        # plot entity locations during rollout mode
        if br < 0:
            print("NRI style doesn't support next-step prediction mode")
            return
        traj_len = pt_x.shape[0]
        s = np.linspace(10, 40, traj_len)
        line13_1 = ax.scatter(gt_x[:br], gt_y[:br], color=(c_pred[n]+c_target[n])/2, marker='o', s= s[:br], alpha=0.3)
        if  traj_len - br > 0:
            line13_2 = ax.scatter(pt_x[br-1:]+pred_x[br-1:], pt_y[br-1:]+pred_y[br-1:],
                                   color=(c_pred[n]+c_target[n])/2, marker='o', s= s[br-1:], alpha=0.8)
    # Display and save figure
    if metadata['savefile'] is not None:
        fig.savefig(metadata['savefile'], bbox_inches='tight')
    
def viz(args):
    # Create output directory
    if args.outdir is None:
        args.outdir = os.path.join('results',args.dataset,'viz_all', mode_dir, 'viz')
    create_directory(args.outdir) # Creates only if it does not already exist
        
    # get input locations
    input_datadir = [os.path.join('results',args.dataset,m, mode_dir, 'debug') for m in models]

    # create metadata structure
    metadata = {}
    metadata['args'] = args
    metadata['total_ax'] = len(input_datadir) + 1
        
    # get burn_in_steps value
    configpkl_file = os.path.join('results',args.dataset,models[0], mode_dir, 'config.pkl')
    with open(configpkl_file,"rb") as f:
        config_args = pickle.load(f)        
    metadata['burn_in_steps'] = config_args[config_args['mode']].get('burn_in_steps',-1)
    if metadata['burn_in_steps'] is None: metadata['burn_in_steps'] = -1
    
    #get batch size
    if config_args['mode'] == 'eval':
        batch_size = config_args[config_args['eval'].get('usage','test')].get('batch_size',-1)
    else:
        batch_size = config_args[config_args['mode']].get('batch_size',-1)

    # Parse the preds directory
    fig_dir = {}
    for preds_file in os.listdir(input_datadir[0]):
        for key in fig_dir:
            fig, _ = fig_dir[key]
            plt.close(fig)

        for d, datadir in enumerate(input_datadir):
            print(datadir)
            preds_npz = np.load(os.path.join(datadir, preds_file))

            # Visualize each sequence
            for i in range(batch_size):
                if str(i)+'_source' not in preds_npz:
                    print("{} file's batch smaller than expected. got batch_size = {}".format(preds_file, i))
                    break
                source = preds_npz[str(i) + '_source']
                target = preds_npz[str(i) + '_target']
                pred = preds_npz[str(i) + '_preds']
                smask = preds_npz[str(i) + '_smask']
                tmask = preds_npz[str(i) + '_tmask']

                pos = preds_file.find('.npz')

                # Trajectory visualization
                if tuple((preds_file, i)) not in fig_dir:
                    fig, ax = plt.subplots(nrows=1, ncols=0, sharex=False, sharey=False, figsize=(24, 24/metadata['total_ax']))
                    fig.subplots_adjust(wspace=0, hspace=0)
                    # Generate colors per agent
                    N = source.shape[0]                    
                    if args.nba_mode:
                        assert N >= 11, "Less than 11 players found!"
                        color = np.concatenate((0.2 * np.ones((11,3), dtype=np.float64), np.ones((N-11,3), dtype=np.float64)), axis=0)
                        color[0,1] = 1.0 # Green for ball
                        color[1:6,0] = 1.0 # Red team
                        color[6:11,2] = 1.0 # Blue team
                    else:
                        h, s, v = np.random.random(N), np.ones(N), np.ones(N)
                        color = matplotlib.colors.hsv_to_rgb(np.stack((h, s, v), axis=1))
                    
                    fig_dir[tuple((preds_file, i))] = (fig, color)
                    
                    metadata['savefile'] = None
                    metadata['fig'], metadata['color'] = fig, color
                    metadata['ax_idx'] = 1
                    metadata['ax_title'] = '(a) Ground truth'
                    viz_traj(source, target, source, smask * tmask, metadata)

                if d == len(input_datadir)-1:
                    savefile = os.path.join(args.outdir, preds_file[:pos] + '_seq{}.png'.format(i))
                    metadata['savefile'] = savefile
                else:
                    metadata['savefile'] = None
                metadata['fig'], metadata['color'] = fig_dir[tuple((preds_file, i))]
                metadata['ax_idx'] = d+2
                metadata['ax_title'] = '(' + chr(ord('a')+metadata['ax_idx']-1) + ') '
                metadata['ax_title'] += model_names[d]
                viz_traj(source, target, pred, smask * tmask, metadata)

############################## Main code ###############################


if __name__ == '__main__':
    # Initial time
    t_init = time.time()

    # Create parser
    parser = argparse.ArgumentParser(description='Visualize trajectories')

    # Add arguments
    parser.add_argument('-o', '--outdir',
        default=None,
        help='Output directory for results (default: same as datadir)')
    parser.add_argument('-d', '--dataset',
        default='ethucy',
        help='dataset dir name in results')
    parser.add_argument('--nba_mode',
                        action='store_true',
                        help='Display only first 11 agents with ball + 2 teams of 5 players each')
    parser.add_argument('-nf','--no-filter',
        help='''if not mentioned, it will filter any agent satisfying one of the following:
                        1) trajectory_len < (1/2)*(episode_len))
                        2) not present at time=burn_in_rate-3''', action='store_true')

    # Parse arguments
    args = parser.parse_args()
    
    # Call the viz method
    viz(args)

    # Final time
    t_final = time.time()
    print('Program finished in {} secs.'.format(t_final - t_init))

import pdb
import os
import importlib.util
import time
import dill as pickle
import shutil
import numpy as np
import torch
import logging

import sys
sys.path.append('./')
from utils.customargparse import CustomArgumentParser, args_to_dict
from utils.misc_utils import create_directory

logger = logging.getLogger(__name__)

############################## Main method ###############################


def run(config):
    # Extract configs
    cfg_tr = config['train']
    cfg_val = config['valid']
    cfg_te = config['test']
    cfg_net = config['net']
    cfg_eval = config['eval']

    # Device: cpu/gpu
    if 'device' in config:
        if config['device'].startswith('cuda') and torch.cuda.is_available():
            device = torch.device(config['device'])
            # BUG: Setting default_tensor_type gives a CUDA initialization
            #      error when using multiple workers in data loader.
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # BUG: Setting default_tensor_type gives a CUDA initialization
            #      error when using multiple workers in data loader.
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
    logger.info('Using device: {}'.format(device))

    # Seeding
    if config['seed'] is not None:
        seed = int(config['seed'])
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Generate global records dictionary
    global_records = {'info': {}, 'result': {}}
    
    # Import model
    spec = importlib.util.spec_from_file_location('model', config['modelfile'])
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    model = model_mod.Model(device, global_records, config)

    # Save configs and models
    pickle.dump(config, file=open(os.path.join(config['outdir'], 'config.pkl'), 'wb'))
    shutil.copy(src=config['configfile'], dst=os.path.join(config['outdir'], 'configfile.py'))
    shutil.copy(src=config['modelfile'], dst=os.path.join(config['outdir'], 'modelfile.py'))
    shutil.copy(src=config['datapath'], dst=os.path.join(config['outdir'], 'dataset.py'))

    if cfg_net['saved_params_path'] is not None:
    	shutil.copy(src=cfg_net['saved_params_path'], dst=os.path.join(config['outdir'], 'init_weights.ptp'))

    # Import data and create data loaders
    logger.info("Using dataset: {}".format(config['datapath']))
    spec = importlib.util.spec_from_file_location('data', config['datapath'])
    data_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_mod)

    ## Train data
    logger.debug("Loading training data...")
    tr_data = config['train']['data']
    tr_dataset = data_mod.get_dataset('tr', **tr_data, burn_in_steps=cfg_tr['burn_in_steps'])
    tr_loader = torch.utils.data.DataLoader(tr_dataset,
                    batch_size=cfg_tr['batch_size'], shuffle=cfg_tr['shuffle'],
                    **config['data_loader'])
    logger.debug("  Training set size: {}".format(len(tr_dataset)))

    ## Test data
    logger.debug("Loading test data...")
    te_data = config['test']['data']
    te_dataset = data_mod.get_dataset('te', **te_data, burn_in_steps=cfg_eval['burn_in_steps'])
    te_loader = torch.utils.data.DataLoader(te_dataset,
                    batch_size=cfg_te['batch_size'], shuffle=cfg_te['shuffle'],
                    **config['data_loader'])
    logger.debug("  Test set size: {}".format(len(te_dataset)))

    ## Validation data
    logger.debug("Loading validation data...")
    val_data = config['valid']['data']
    val_dataset = data_mod.get_dataset('val', **val_data, burn_in_steps=cfg_eval['burn_in_steps'])
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=cfg_val['batch_size'], shuffle=cfg_val['shuffle'],
                        **config['data_loader'])
    else:
        val_loader = None
    logger.debug("  Validation set size: {}".format(len(val_dataset)))

    # Train mode
    if config['mode'] == 'train':
        # Train model
        tic = time.time()
        results = model.fit(tr_loader, val_loader)
        toc = time.time()
        global_records['tr_time'] = toc - tic

        # Save results
        pickle.dump(global_records, file=open(os.path.join(config['outdir'], config['record_file']), 'wb'))

    # Eval mode
    elif config['mode'] == 'eval':
        # Decide data usage
        assert cfg_eval['usage'] in ['train', 'valid', 'test'], 'usage should be one of [train, valid, test]'
        if cfg_eval['usage'] == 'train':
            data_loader = tr_loader
        elif cfg_eval['usage'] == 'test':
            data_loader = te_loader
        elif cfg_eval['usage'] == 'valid':
            data_loader = val_loader

        # Evaluate on data
        tic = time.time()
        results = model.evaluate(data_loader)
        toc = time.time()
        global_records['eval_time'] = toc - tic

        # Save results
        pickle.dump(global_records, file=open(os.path.join(config['outdir'], config['record_file']), 'wb'))


############################## Main code ###############################


if __name__ == '__main__':
    # Initial time
    t_init = time.time()

    # Create parser
    parser = CustomArgumentParser(description='Run exps')

    # Add arguments
    parser.add_argument('-c', '--configfile',
        help='Input file for parameters, constants and initial settings')
    parser.add_argument('-d', '--datapath',
        help='Path of python module to load the dataset in Torch format')
    parser.add_argument('-m', '--modelfile',
        help='Input file for model')
    parser.add_argument('-o', '--outdir',
        help='Output directory for results')
    parser.add_argument('-mode', '--mode',
        choices=['train', 'eval'], default='train',
        help='Mode: [train, eval]')
    parser.add_argument('-v', '--verbose',
        help='Increase output verbosity', action='store_true')

    # Parse arguments
    args = parser.parse_args()
    config = args_to_dict(args)

    # Create output directory if it does not exist
    create_directory(config['outdir'])

    # Set up logging
    log_file = os.path.join(config['outdir'], config['logging']['log_file'])
    log_fmt = config['logging']['fmt']
    log_level = config['logging']['level']
    log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='w')]
    logging.basicConfig(format=log_fmt, level=logging.INFO, handlers=log_handlers)
    logging.getLogger('models').setLevel(log_level)
    logger.setLevel(log_level)
    
    # Call the main method
    run_results = run(config)

    # Final time
    t_final = time.time()
    logger.info('Program finished in {} secs.'.format(t_final - t_init))

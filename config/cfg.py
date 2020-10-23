config = {
    'device': 'cpu',                                # device to run on
    'log_interval': 10,                             # Logging interval for debugging (None, int)
    'record_file': 'records.pkl',                   # File to save global records dictionary

    # config for logging
    'logging': {
        'log_file': 'run.log',                      # log file name
        'fmt': '%(asctime)s: %(message)s',          # logging format
        'level': 'DEBUG',                           # logger level
    },

    # config to load and save networks
    'net': {
        'use_mask': True,                           # Whether to use the mask for predictions, if provided
        'saved_params_path': None                   # path to load saved weights in a loaded network
    },

    # config for data loader
    'data_loader': {
        'num_workers': 0,                           # Number of parallel CPU workers
        'pin_memory': False,                        # Copy tensors into CUDA pinned memory before returning them
        'collate_fn': lambda x: x,                  # collate_fn to get minibatch as list instead of tensor
    },

    # Random seed for numpy, torch and cuda (None, int)
    'seed': 1,

    'optimizer': {
        'name': 'adam',
        'params': {
            'lr': 1e-3,
        },
        'scheduler': {
            'step_size': 5,                         # Period of learning rate decay
            'gamma': 0.8,                           # Multiplicative factor of learning rate decay
        }
    },

    # config to control training
    'train': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'stop_crit':{
            'max_patience': 10,                     # patience for early stopping (int, None)
            'min_epoch': 50,                        # minimum epochs to run for (int)
            'max_epoch': 100                        # maximum epochs to run for (int)
        },        
        'batch_size': 32,
        'shuffle': True,
        'burn_in_steps': 8,                         # Number of burn in steps before rollout (int)
        'dynamic_burn_in': True                     # Dynamically vary burn_in_steps (default: True)
    },
    # config to control validation
    'valid': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 32,
        'shuffle': False
    },
    # config to control testing
    'test': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 32,
        'shuffle': False
    },
    # config to control evaluation
    'eval': {
        'usage': 'test',                            # what dataset to use {train, valid, test}
        'store_losses': True,                       # store losses while evaluating
        'store_debug': False,                       # store debug outputs while evaluating
        'burn_in_steps': 8,                         # Number of burn in steps before rollout (int)
        'dynamic_burn_in': False                    # Dynamically vary burn_in_steps (default: False)
    }
}

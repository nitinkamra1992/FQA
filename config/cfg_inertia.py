from config.cfg import config

cfg = {
    # config to load and save networks
    'net': {
        'use_mask': True,                           # Whether to use the mask for predictions, if provided
        'saved_params_path': None                   # path to load saved weights in a loaded network
    },

    'optimizer': {
        'name': 'sgd',
        'params': {
            'lr': 1e-3,
            'momentum': 0.0
        },
    },

    # config to control evaluation
    'eval': {
        'usage': 'test',                            # what dataset to use {train, valid, test}
        'store_losses': True,                       # store losses while evaluating
        'store_debug': True,                        # store debug outputs while evaluating
        'burn_in_steps': 8,                         # Number of burn in steps before rollout (int)
        'dynamic_burn_in': False                    # Dynamically vary burn_in_steps (default: False)
    }
}

config.update(cfg)
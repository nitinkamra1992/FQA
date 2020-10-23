from config.cfg import config

cfg = {
    'device': 'cuda:0',                             # device to run on

    # config to load and save networks
    'net': {
        'input_size': 4,                            # Input shape of network
        'embed_size': 32,
        'hidden_size': 64,                          # Hidden shape of network
        'output_size': 2,                           # Output shape of network
        'rtype': 'lstm',                            # Recurrent core type: ['rnn_tanh', 'rnn_relu', 'gru', 'lstm']
        'use_mask': True,                           # Whether to use the mask for predictions, if provided
        'saved_params_path': None                   # path to load saved weights in a loaded network
    },

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
}

config.update(cfg)
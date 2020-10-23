from config.cfg import config

cfg = {
    'device': 'cuda:0',                             # device to run on

    # config to load and save networks
    'net': {
        'input_dim': 2,                             # Input dimension of network
        'output_dim': 2,                            # Output dimension of network
        'hidden_dim': 32,                           # Hidden dimension of recurrent core
        'dist_threshold': -1.0,                     # Distance threshold
        'rtype': 'lstm',                            # Recurrent core type: ['rnn_tanh', 'rnn_relu', 'gru', 'lstm']
        'use_vel': True,                            # Use direct velocity in the final prediction
        'attention_params': {
            'n_layers': 1,                          # Number of layers
            'n_q': 8,                               # Number of queries;
            'd_qk': 4,                              # Dimension of queries and keys
            'd_v': 6,                               # Dimension of values
            'att_dim': 32,                          # Dimension of attention output
            'n_hk_q': 0,                            # Number of human-knowledge queries to use
            'flags': [],                            # Flags for ablations; Use nodec or nointeract
        },
        'use_mask': True,                           # Whether to use the mask for predictions, if provided
        'saved_params_path': None                   # Path to load saved weights in a loaded network
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
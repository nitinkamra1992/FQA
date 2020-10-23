import pdb
import pickle
import os
from statistics import mean, stdev

import sys
sys.path.append('./')
from utils.misc_utils import create_directory

######## Helper methods #######

def aggregate_metrics_to_str(aggregate_metrics, indent=0, print_vals=False):
    metric_str = ''
    for metric, metric_dict in aggregate_metrics.items():
        if print_vals:
            metric_str = metric_str + '\t' * indent + '{} = {}: {:.3f} \u00B1 {:.4f}, [{:.3f}, {:.3f}]'.format(metric, metric_dict['vals'], metric_dict['mean'], metric_dict['stddev'], metric_dict['min'], metric_dict['max']) + '\n'
        else:
            metric_str = metric_str + '\t' * indent + '{}: {:.3f} \u00B1 {:.4f}, [{:.3f}, {:.3f}]'.format(metric, metric_dict['mean'], metric_dict['stddev'], metric_dict['min'], metric_dict['max']) + '\n'
    return metric_str

######## Metric display ########

# Output path
outdir = 'results'
outfile = 'metrics.txt'
create_directory(outdir)

# Datasets and models
# dsets = ['ethucy', 'collisions', 'ngsim', 'charged', 'nba'] # Uncomment this line if nba data is available too
dsets = ['ethucy', 'collisions', 'ngsim', 'charged']
models = [
          'InertiaModel', 'VanillaLSTM', \
          'FQA/DCE', 'FQA/AEdge', \
          'FQA/AEdge_nodec', \
          'FQA/AEdge_nointeract', \
          'FQA/FQA_add_hk1' \
         ]
metrics = ['rmse_loss']
evalset = 'eval_test'
runs = range(1, 6)

# Collect metrics
f = open(os.path.join(outdir, outfile), 'w')
for dset in dsets:
    for model in models:
        # Dictionary to hold aggregate statistics for all metrics
        aggregate_metric = {
            k: {
                'vals': [],
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'stddev': 0.0
            } for k in metrics
        }
        # Inspect all runs and accumulate metric values
        for run in runs:
            metrics_path = 'results/{}/{}/run{}/{}/metrics.pkl'.format(dset, model, run, evalset)
            try:
                metric = pickle.load(open(metrics_path, 'rb'))
                for k in metrics:
                    aggregate_metric[k]['vals'].append(metric[k])
            except Exception as e:
                print(e)
                continue
        # Compute aggregate statistics
        for k in metrics:
            vals = aggregate_metric[k]['vals']
            if len(vals) > 1:
                aggregate_metric[k]['mean'] = mean(vals)
                aggregate_metric[k]['stddev'] = stdev(vals)
                aggregate_metric[k]['min'] = min(vals)
                aggregate_metric[k]['max'] = max(vals)
            else:
                print('len(vals) = {} should be greater than 1'.format(len(vals)))
        # Print and save metric string
        metric_str = '{}, {}:\n{}'.format(dset, model, aggregate_metrics_to_str(aggregate_metric, 1))
        print(metric_str)
        f.write(metric_str)
f.close()
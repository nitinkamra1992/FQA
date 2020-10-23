#!/bin/bash

#############################################

# Run ID (changes the random seed)

# i=1

#############################################

# InertiaModel

## collisions
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_inertia.py -d data/collisions/collisions.py -m models/InertiaModel.py -o results/collisions/InertiaModel/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/collisions/InertiaModel/run${i}/eval_test/ &

## ethucy
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_inertia.py -d data/ethucy/ethucy.py -m models/InertiaModel.py -o results/ethucy/InertiaModel/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ethucy/InertiaModel/run${i}/eval_test/ &

## ngsim
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_inertia.py -d data/ngsim/ngsim.py -m models/InertiaModel.py -o results/ngsim/InertiaModel/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ngsim/InertiaModel/run${i}/eval_test/ &

## charged
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_inertia.py -d data/charged/charged.py -m models/InertiaModel.py -o results/charged/InertiaModel/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/charged/InertiaModel/run${i}/eval_test/ &

## nba
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_inertia.py -d data/nba/nba.py -m models/InertiaModel.py -o results/nba/InertiaModel/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 12 --eval.burn_in_steps 12 &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/nba/InertiaModel/run${i}/eval_test/ &

#############################################

# Vanilla LSTM

## collisions
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_vlstm.py -d data/collisions/collisions.py -m models/VanillaLSTM.py -o results/collisions/VanillaLSTM/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.embed_size 32 --net.hidden_size 64 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_vlstm.py -d data/collisions/collisions.py -m models/VanillaLSTM.py -o results/collisions/VanillaLSTM/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.embed_size 32 --net.hidden_size 64 --net.saved_params_path results/collisions/VanillaLSTM/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/collisions/VanillaLSTM/run${i}/eval_test/ &

## ethucy
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_vlstm.py -d data/ethucy/ethucy.py -m models/VanillaLSTM.py -o results/ethucy/VanillaLSTM/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.embed_size 32 --net.hidden_size 64 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_vlstm.py -d data/ethucy/ethucy.py -m models/VanillaLSTM.py -o results/ethucy/VanillaLSTM/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.embed_size 32 --net.hidden_size 64 --net.saved_params_path results/ethucy/VanillaLSTM/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ethucy/VanillaLSTM/run${i}/eval_test/ &

## ngsim
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_vlstm.py -d data/ngsim/ngsim.py -m models/VanillaLSTM.py -o results/ngsim/VanillaLSTM/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.embed_size 32 --net.hidden_size 64 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_vlstm.py -d data/ngsim/ngsim.py -m models/VanillaLSTM.py -o results/ngsim/VanillaLSTM/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.embed_size 32 --net.hidden_size 64 --net.saved_params_path results/ngsim/VanillaLSTM/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ngsim/VanillaLSTM/run${i}/eval_test/ &

## charged
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_vlstm.py -d data/charged/charged.py -m models/VanillaLSTM.py -o results/charged/VanillaLSTM/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.embed_size 32 --net.hidden_size 64 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_vlstm.py -d data/charged/charged.py -m models/VanillaLSTM.py -o results/charged/VanillaLSTM/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.embed_size 32 --net.hidden_size 64 --net.saved_params_path results/charged/VanillaLSTM/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/charged/VanillaLSTM/run${i}/eval_test/ &

## nba
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_vlstm.py -d data/nba/nba.py -m models/VanillaLSTM.py -o results/nba/VanillaLSTM/run${i}/train/ -mode train -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.embed_size 32 --net.hidden_size 64 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_vlstm.py -d data/nba/nba.py -m models/VanillaLSTM.py -o results/nba/VanillaLSTM/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.embed_size 32 --net.hidden_size 64 --net.saved_params_path results/nba/VanillaLSTM/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/nba/VanillaLSTM/run${i}/eval_test/ &

#############################################

# FQA

## collisions
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/collisions/collisions.py -m models/FQA/FQA.py -o results/collisions/FQA/DCE/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold 0.5 &
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/collisions/collisions.py -m models/FQA/FQA.py -o results/collisions/FQA/AEdge/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/collisions/collisions.py -m models/FQA/FQA.py -o results/collisions/FQA/DCE/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold 0.5 --net.saved_params_path results/collisions/FQA/DCE/run${i}/train/best_valid_params.ptp --eval.store_debug True &
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/collisions/collisions.py -m models/FQA/FQA.py -o results/collisions/FQA/AEdge/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.saved_params_path results/collisions/FQA/AEdge/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/collisions/FQA/DCE/run${i}/eval_test/ &
# python3 src/generate_eval_metrics.py -d results/collisions/FQA/AEdge/run${i}/eval_test/ &

## ethucy
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ethucy/ethucy.py -m models/FQA/FQA.py -o results/ethucy/FQA/DCE/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold 0.5 &
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ethucy/ethucy.py -m models/FQA/FQA.py -o results/ethucy/FQA/AEdge/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ethucy/ethucy.py -m models/FQA/FQA.py -o results/ethucy/FQA/DCE/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold 0.5 --net.saved_params_path results/ethucy/FQA/DCE/run${i}/train/best_valid_params.ptp --eval.store_debug True &
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ethucy/ethucy.py -m models/FQA/FQA.py -o results/ethucy/FQA/AEdge/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.saved_params_path results/ethucy/FQA/AEdge/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ethucy/FQA/DCE/run${i}/eval_test/ &
# python3 src/generate_eval_metrics.py -d results/ethucy/FQA/AEdge/run${i}/eval_test/ &

## ngsim
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ngsim/ngsim.py -m models/FQA/FQA.py -o results/ngsim/FQA/DCE/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold 0.5 &
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ngsim/ngsim.py -m models/FQA/FQA.py -o results/ngsim/FQA/AEdge/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ngsim/ngsim.py -m models/FQA/FQA.py -o results/ngsim/FQA/DCE/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold 0.5 --net.saved_params_path results/ngsim/FQA/DCE/run${i}/train/best_valid_params.ptp --eval.store_debug True &
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ngsim/ngsim.py -m models/FQA/FQA.py -o results/ngsim/FQA/AEdge/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.saved_params_path results/ngsim/FQA/AEdge/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ngsim/FQA/DCE/run${i}/eval_test/ &
# python3 src/generate_eval_metrics.py -d results/ngsim/FQA/AEdge/run${i}/eval_test/ &

## charged
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/charged/charged.py -m models/FQA/FQA.py -o results/charged/FQA/DCE/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold 0.5 &
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/charged/charged.py -m models/FQA/FQA.py -o results/charged/FQA/AEdge/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/charged/charged.py -m models/FQA/FQA.py -o results/charged/FQA/DCE/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold 0.5 --net.saved_params_path results/charged/FQA/DCE/run${i}/train/best_valid_params.ptp --eval.store_debug True &
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/charged/charged.py -m models/FQA/FQA.py -o results/charged/FQA/AEdge/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.saved_params_path results/charged/FQA/AEdge/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/charged/FQA/DCE/run${i}/eval_test/ &
# python3 src/generate_eval_metrics.py -d results/charged/FQA/AEdge/run${i}/eval_test/ &

## nba
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/nba/nba.py -m models/FQA/FQA.py -o results/nba/FQA/DCE/run${i}/train/ -mode train -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.hidden_dim 32 --net.dist_threshold 0.5 &
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/nba/nba.py -m models/FQA/FQA.py -o results/nba/FQA/AEdge/run${i}/train/ -mode train -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.hidden_dim 32 --net.dist_threshold -1.0 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/nba/nba.py -m models/FQA/FQA.py -o results/nba/FQA/DCE/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.hidden_dim 32 --net.dist_threshold 0.5 --net.saved_params_path results/nba/FQA/DCE/run${i}/train/best_valid_params.ptp --eval.store_debug True &
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/nba/nba.py -m models/FQA/FQA.py -o results/nba/FQA/AEdge/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.saved_params_path results/nba/FQA/AEdge/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/nba/FQA/DCE/run${i}/eval_test/ &
# python3 src/generate_eval_metrics.py -d results/nba/FQA/AEdge/run${i}/eval_test/ &

#############################################

# FQA human-knowledge tests with human-knowledge addition

## collisions
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/collisions/collisions.py -m models/FQA/FQA.py -o results/collisions/FQA/FQA_add_hk1/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.n_q 8 --net.attention_params.n_hk_q 1 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/collisions/collisions.py -m models/FQA/FQA.py -o results/collisions/FQA/FQA_add_hk1/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.n_q 8 --net.attention_params.n_hk_q 1 --net.saved_params_path results/collisions/FQA/FQA_add_hk1/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/collisions/FQA/FQA_add_hk1/run${i}/eval_test/ &

## ethucy
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ethucy/ethucy.py -m models/FQA/FQA.py -o results/ethucy/FQA/FQA_add_hk1/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.n_q 8 --net.attention_params.n_hk_q 1 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ethucy/ethucy.py -m models/FQA/FQA.py -o results/ethucy/FQA/FQA_add_hk1/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.n_q 8 --net.attention_params.n_hk_q 1 --net.saved_params_path results/ethucy/FQA/FQA_add_hk1/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ethucy/FQA/FQA_add_hk1/run${i}/eval_test/ &

## ngsim
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ngsim/ngsim.py -m models/FQA/FQA.py -o results/ngsim/FQA/FQA_add_hk1/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.n_q 8 --net.attention_params.n_hk_q 1 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ngsim/ngsim.py -m models/FQA/FQA.py -o results/ngsim/FQA/FQA_add_hk1/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.n_q 8 --net.attention_params.n_hk_q 1 --net.saved_params_path results/ngsim/FQA/FQA_add_hk1/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ngsim/FQA/FQA_add_hk1/run${i}/eval_test/ &

## charged
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/charged/charged.py -m models/FQA/FQA.py -o results/charged/FQA/FQA_add_hk1/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.n_q 8 --net.attention_params.n_hk_q 1 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/charged/charged.py -m models/FQA/FQA.py -o results/charged/FQA/FQA_add_hk1/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.n_q 8 --net.attention_params.n_hk_q 1 --net.saved_params_path results/charged/FQA/FQA_add_hk1/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/charged/FQA/FQA_add_hk1/run${i}/eval_test/ &

## nba
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/nba/nba.py -m models/FQA/FQA.py -o results/nba/FQA/FQA_add_hk1/run${i}/train/ -mode train -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.n_q 8 --net.attention_params.n_hk_q 1 &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/nba/nba.py -m models/FQA/FQA.py -o results/nba/FQA/FQA_add_hk1/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.n_q 8 --net.attention_params.n_hk_q 1 --net.saved_params_path results/nba/FQA/FQA_add_hk1/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/nba/FQA/FQA_add_hk1/run${i}/eval_test/ &

#############################################

# FQA ablations

# NoDec

## collisions
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/collisions/collisions.py -m models/FQA/FQA.py -o results/collisions/FQA/AEdge_nodec/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nodec]' &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/collisions/collisions.py -m models/FQA/FQA.py -o results/collisions/FQA/AEdge_nodec/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nodec]' --net.saved_params_path results/collisions/FQA/AEdge_nodec/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/collisions/FQA/AEdge_nodec/run${i}/eval_test/ &

## ethucy
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ethucy/ethucy.py -m models/FQA/FQA.py -o results/ethucy/FQA/AEdge_nodec/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nodec]' &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ethucy/ethucy.py -m models/FQA/FQA.py -o results/ethucy/FQA/AEdge_nodec/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nodec]' --net.saved_params_path results/ethucy/FQA/AEdge_nodec/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ethucy/FQA/AEdge_nodec/run${i}/eval_test/ &

## ngsim
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ngsim/ngsim.py -m models/FQA/FQA.py -o results/ngsim/FQA/AEdge_nodec/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nodec]' &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ngsim/ngsim.py -m models/FQA/FQA.py -o results/ngsim/FQA/AEdge_nodec/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nodec]' --net.saved_params_path results/ngsim/FQA/AEdge_nodec/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ngsim/FQA/AEdge_nodec/run${i}/eval_test/ &

## charged
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/charged/charged.py -m models/FQA/FQA.py -o results/charged/FQA/AEdge_nodec/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nodec]' &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/charged/charged.py -m models/FQA/FQA.py -o results/charged/FQA/AEdge_nodec/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nodec]' --net.saved_params_path results/charged/FQA/AEdge_nodec/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/charged/FQA/AEdge_nodec/run${i}/eval_test/ &

## nba
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/nba/nba.py -m models/FQA/FQA.py -o results/nba/FQA/AEdge_nodec/run${i}/train/ -mode train -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nodec]' &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/nba/nba.py -m models/FQA/FQA.py -o results/nba/FQA/AEdge_nodec/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nodec]' --net.saved_params_path results/nba/FQA/AEdge_nodec/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/nba/FQA/AEdge_nodec/run${i}/eval_test/ &

# NoIntr

## collisions
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/collisions/collisions.py -m models/FQA/FQA.py -o results/collisions/FQA/AEdge_nointeract/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nointeract]' &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/collisions/collisions.py -m models/FQA/FQA.py -o results/collisions/FQA/AEdge_nointeract/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nointeract]' --net.saved_params_path results/collisions/FQA/AEdge_nointeract/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/collisions/FQA/AEdge_nointeract/run${i}/eval_test/ &

## ethucy
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ethucy/ethucy.py -m models/FQA/FQA.py -o results/ethucy/FQA/AEdge_nointeract/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nointeract]' &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ethucy/ethucy.py -m models/FQA/FQA.py -o results/ethucy/FQA/AEdge_nointeract/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nointeract]' --net.saved_params_path results/ethucy/FQA/AEdge_nointeract/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ethucy/FQA/AEdge_nointeract/run${i}/eval_test/ &

## ngsim
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ngsim/ngsim.py -m models/FQA/FQA.py -o results/ngsim/FQA/AEdge_nointeract/run${i}/train/ -mode train -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nointeract]' &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/ngsim/ngsim.py -m models/FQA/FQA.py -o results/ngsim/FQA/AEdge_nointeract/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 8 --eval.burn_in_steps 8 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nointeract]' --net.saved_params_path results/ngsim/FQA/AEdge_nointeract/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/ngsim/FQA/AEdge_nointeract/run${i}/eval_test/ &

## charged
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/charged/charged.py -m models/FQA/FQA.py -o results/charged/FQA/AEdge_nointeract/run${i}/train/ -mode train -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nointeract]' &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/charged/charged.py -m models/FQA/FQA.py -o results/charged/FQA/AEdge_nointeract/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 10 --eval.burn_in_steps 10 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nointeract]' --net.saved_params_path results/charged/FQA/AEdge_nointeract/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/charged/FQA/AEdge_nointeract/run${i}/eval_test/ &

## nba
#### Train
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/nba/nba.py -m models/FQA/FQA.py -o results/nba/FQA/AEdge_nointeract/run${i}/train/ -mode train -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nointeract]' &
#### Eval_test
# python3 src/run.py --seed ${i} -c config/cfg_FQA.py -d data/nba/nba.py -m models/FQA/FQA.py -o results/nba/FQA/AEdge_nointeract/run${i}/eval_test/ -mode eval -v --train.burn_in_steps 12 --eval.burn_in_steps 12 --net.hidden_dim 32 --net.dist_threshold -1.0 --net.attention_params.flags '[nointeract]' --net.saved_params_path results/nba/FQA/AEdge_nointeract/run${i}/train/best_valid_params.ptp --eval.store_debug True &
#### Eval metrics
# python3 src/generate_eval_metrics.py -d results/nba/FQA/AEdge_nointeract/run${i}/eval_test/ &

#############################################

# Visualizations

# python3 src/viz.py -o results/viz/collisions/ -d collisions &
# python3 src/viz.py -o results/viz/ethucy/ -d ethucy &
# python3 src/viz.py -o results/viz/ngsim/ -d ngsim &
# python3 src/viz.py -o results/viz/charged/ -d charged &
# python3 src/viz.py -o results/viz/nba/ -d nba --nba_mode &

#############################################
#!/usr/bin/env bash
cd ../

for seed in 15 49 124 ; do
    tmux new "conda activate DeepFlarePred;  read" ';' split 'python main_TCN_Liu.py \
    --seed $seed'
done


#!/usr/bin/env bash
cd ../

for dataset in Liu Liu_train ; do
    for seed in 15 124 49; do
        python main_TCN_Liu.py \
        --seed $seed \
        --dataset $dataset
    done
done


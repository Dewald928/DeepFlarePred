#!/usr/bin/env bash
cd ../

for seed in 15 124 49
    do
        python main_TCN_Liu.py \
        --seed $seed
    done
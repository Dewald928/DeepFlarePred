#!/usr/bin/env bash

for i in 16 125 654 79 31
    do
        python main_TCN_Liu.py --seed $i
    done
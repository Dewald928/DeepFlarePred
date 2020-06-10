#!/usr/bin/env bash

for i in 15 124 49 273 335
    do
        python main_TCN_Liu.py --seed $i
    done
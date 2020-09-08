#!/usr/bin/env bash
cd ../

for lr_rangetest_iter in 50 100 200 ; do
    python main_TCN_Liu.py \
    --lr_rangetest_iter $lr_rangetest_iter
done


#!/usr/bin/env bash
cd ../

for model_type in MLP ; do
    for optim in SGD Adam ; do
        for layers in 1 2 ; do
            for hidden_units in 40 100 200 ; do
                for weight_decay in $(seq 0 0.001 0.0001 0.00001) ; do
                    for batch_size in 256 512 1024 2048 4096 8192 16384 \
                    32768 65536; do
                        for seed in 15 124 49 ; do
                            python main_TCN_Liu.py \
                            --seed $seed \
                            --optim $optim \
                            --model_type $model_type \
                            --batch_size $batch_size \
                            --weight_decay $weight_decay \
                            --layers $layers \
                            --hidden_units $hidden_units &
                        done
                    done
                done
            done
        done
    done
done

#for model_type in TCN ; do
#    for optim in SGD Adam ; do
#        for levels in 1 2 ; do
#            for ksize in 2 3 ; do
#                for weight_decay in $(seq 0 0.001 0.0001 0.00001 ; do
#                    for batch_size in 256 512 1024 2048 4096 8192 16384 \
#                    32768 65536; do
#                        for seed in 15 124 49 ; do
#                            python main_TCN_Liu.py \
#                            --seed $seed \
#                            --optim $optim \
#                            --model_type $model_type \
#                            --batch_size $batch_size \
#                            --weight_decay $weight_decay \
#                            --levels levels \
#                            --ksize ksize &
#                        done
#                    done
#                done
#            done
#        done
#    done
#done



















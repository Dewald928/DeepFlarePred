#!/usr/bin/env bash
cd ../

#for model_type in MLP ; do
#    for optim in SGD Adam ; do
#        for layers in 1 2 ; do
#            for hidden_units in 40 100 200 ; do
#                for weight_decay in $(seq 0 0.001 0.0001 0.00001) ; do
#                    for batch_size in 65536; do
#                        for seed in 15 124 49 ; do
#                            python main_TCN_Liu.py \
#                            --seed $seed \
#                            --optim $optim \
#                            --model_type $model_type \
#                            --batch_size $batch_size \
#                            --weight_decay $weight_decay \
#                            --layers $layers \
#                            --hidden_units $hidden_units &
#                        done
#                    done
#                done
#            done
#        done
#    done
#done
N=4
for dataset in Liu Liu_train Liu_z Krynauw; do
  for optim in SGD Adam; do
    for dropout in {0.0 0.4}; do
      for weight_decay in {0.0 0.01}; do
        for seed in 15 124; do
          ((i = i % N))
          ((i++ == 0)) && wait
          python main_TCN_Liu.py \
            --seed $seed \
            --optim $optim \
            --dataset $dataset \
            --weight_decay $weight_decay \
            --dropout $dropout &
        done
      done
    done
  done
done

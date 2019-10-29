#!/bin/bash
source activate DeepFlarePred
#gradient experiments run singlenode --name train --projectId prwr96qst --container janakiramm/python:3 --machineType C3 --command 'python train/train.py -i ./data/sal.csv -o /storage/salary' --workspace https://github.com/janakiramm/Salary.git


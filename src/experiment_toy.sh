#!/usr/bin/env bash

from_resample=0
to_resample=5

for (( i = $from_resample; i <= $to_resample; i++ )); do
    # Causal sample
    task="toy resample $i causal-sample"
    ./train.py -t $task causal-model
    ./train.py -t $task
    ./plot.py full -t $task causal-model
    ./plot.py full -t $task
    ./plot.py compare -t $task causal-model -t $task

    # Acausal sample
    task="toy resample $i"
    ./train.py -t $task causal-model
    ./train.py -t $task
    ./plot.py full -t $task causal-model
    ./plot.py full -t $task
    ./plot.py compare -t $task causal-model -t $task
done

#!/usr/bin/env bash
seed_causal=1025
seed_acausal=1030
from_resample=0
to_resample=5

for (( i = $from_resample; i <= $to_resample; i++ )); do
    # Causal sample
    prefix="toy seed $seed_causal resample $i causal"
    ./train.py -t $prefix causal-model
    ./train.py -t $prefix
    ./plot.py full -t $prefix causal-model
    ./plot.py full -t $prefix
    ./plot.py compare -t $prefix causal-model -t $prefix

    # Acausal sample
    prefix="toy seed $seed_acausal resample $i noisy"
    ./train.py -t $prefix causal-model
    ./train.py -t $prefix
    ./plot.py full -t $prefix causal-model
    ./plot.py full -t $prefix
    ./plot.py compare -t $prefix causal-model -t $prefix
done

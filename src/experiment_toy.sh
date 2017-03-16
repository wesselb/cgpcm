#!/usr/bin/env bash

opts=$@
from_resample=0
to_resample=100

for (( i = $from_resample; i <= $to_resample; i++ )); do
    # Causal sample
    ./core.py -t toy resample $i causal-sample causal-model \
              -t toy resample $i causal-sample \
              -p full index 0 \
              -p full index 1 \
              -p compare index1 0 index2 1 \
              $opts

    # Acausal sample
    ./core.py -t toy resample $i causal-model \
              -t toy resample $i \
              -p full index 0 \
              -p full index 1 \
              -p compare index1 0 index2 1 \
              $opts
done

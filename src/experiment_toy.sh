#!/usr/bin/env bash

from_resample=$1
to_resample=$2
opts=${@:3:${#}}

for (( i = $from_resample; i <= $to_resample; i++ )); do
    # Causal sample
    ./core.py -t toy resample $i causal-sample causal-model \
              -t toy resample $i causal-sample \
              -p compare2 index1 0 index2 0 mf1 \
              -p compare2 index1 1 index2 1 mf1 \
              -p compare2 index1 0 index2 1 \
              $opts

    # Acausal sample
    ./core.py -t toy resample $i causal-model \
              -t toy resample $i \
              -p compare2 index1 0 index2 0 mf1 \
              -p compare2 index1 1 index2 1 mf1 \
              -p compare2 index1 0 index2 1 \
              $opts
done

#!/usr/bin/env bash

from_resample=0
to_resample=40

for (( i = $from_resample; i <= $to_resample; i++ )); do
    # Causal sample
    ./core.py -t toy resample $i causal-sample causal-model \
              -t toy resample $i causal-sample \
              --learn

    # Acausal sample
    ./core.py -t toy resample $i causal-model \
              -t toy resample $i \
              --learn
done

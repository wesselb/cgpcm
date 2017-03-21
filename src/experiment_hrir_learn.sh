#!/usr/bin/env bash

from_resample=100
to_resample=120

for (( i = $from_resample; i <= $to_resample; i++ )); do
    ./core.py -t hrir resample $i causal-model \
              -t hrir resample $i \
              --learn
done

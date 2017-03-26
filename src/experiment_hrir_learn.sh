#!/usr/bin/env bash

from_resample=$1
to_resample=$2

for (( i = $from_resample; i <= $to_resample; i++ )); do
    ./core.py -t hrir resample $i causal-model \
              -t hrir resample $i \
              -p compare2 index1 0 index2 0 mp no-psd ms mf1 \
              -p compare2 index1 1 index2 1 mp no-psd ms mf1 \
              -p compare2 index1 0 index2 1 mp no-psd ms \
              --learn
done

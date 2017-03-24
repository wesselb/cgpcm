#!/usr/bin/env bash

from_resample=100
to_resample=120

for (( i = $from_resample; i <= $to_resample; i++ )); do
    ./core.py -t hrir resample $i causal-model \
              -t hrir resample $i \
              -p compare2 index1 0 index2 0 2side-h no-psd ms mf1 \
              -p compare2 index1 1 index2 1 2side-h no-psd ms mf1 \
              -p compare2 index1 0 index2 1 2side-h no-psd ms \
              -r
done

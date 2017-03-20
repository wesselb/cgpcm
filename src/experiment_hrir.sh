#!/usr/bin/env bash

opts=$@
from_resample=100
to_resample=114

for (( i = $from_resample; i <= $to_resample; i++ )); do
    ./core.py -t hrir resample $i causal-model \
              -t hrir resample $i \
              -p compare2 index1 0 index2 1 correct-h ms no-psd \
              $opts
done

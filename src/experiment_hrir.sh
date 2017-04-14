#!/usr/bin/env bash

from_resample=$1
to_resample=$2
opts=${@:3:${#}}

for (( i = $from_resample; i <= $to_resample; i++ )); do
    ./controller.py.py -t hrir resample $i causal-model \
                       -t hrir resample $i \
                       -p compare index0 0 index1 0 no-psd ms mf0 \
                       -p compare index0 1 index1 1 no-psd ms mf0 \
                       -p compare index0 0 index1 1 no-psd ms \
                       $opts
done

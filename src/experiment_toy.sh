#!/usr/bin/env bash

from_resample=$1
to_resample=$2
opts=${@:3:${#}}

for (( i = $from_resample; i <= $to_resample; i++ )); do
    ./core.py -t toy resample $i causal-sample causal-model \
              -t toy resample $i causal-sample \
              -t toy resample $i causal-model \
              -t toy resample $i \
              -p compare index0 0 index1 0 mf0 \
              -p compare index0 1 index1 1 mf0 \
              -p compare index0 0 index1 1 \
              -p compare index0 2 index1 2 mf0 \
              -p compare index0 3 index1 3 mf0 \
              -p compare index0 2 index1 3 \
              $opts
done

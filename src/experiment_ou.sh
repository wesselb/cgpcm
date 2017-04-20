#!/usr/bin/env bash

from_seed=$1
to_seed=$2
opts=${@:3:${#}}

for (( i = $from_seed; i <= $to_seed; i++ )); do
    ./controller.py -t ou seed $i causal-model \
                    -t ou seed $i \
                    -p compare index0 0 index1 0 mf0 \
                    -p compare index0 1 index1 1 mf0 \
                    -p compare index0 0 index1 1 \
                    $opts
done

#!/usr/bin/env bash
set -e

task=$1
from_seed=$2
to_seed=$3
step_seed=$4
opts=${@:5:${#}}

for (( i = $from_seed; i <= $to_seed; i += $step_seed )); do
    ./train.py -t $task seed $i resample 0 small $opts -d kernel
done




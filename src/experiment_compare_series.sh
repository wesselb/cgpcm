#!/usr/bin/env bash

task=$1
seed=$2
from_resample=$3
to_resample=$4
opts=${@:5:${#}}

for (( i = $from_resample; i <= $to_resample; i++ )); do
    ./experiment_compare.sh $task $seed $i $opts
done




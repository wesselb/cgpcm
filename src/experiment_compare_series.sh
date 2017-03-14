#!/usr/bin/env bash
task=$1
seed=$2
to_resample=$3
opts=${@:4:${#}}
for (( i=0; i<=$to_resample; i++ ))
do
    ./experiment_compare.sh $task seed $seed resample $i $opts
done




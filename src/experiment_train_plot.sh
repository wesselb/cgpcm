#!/usr/bin/env bash
set -e

task=$1
seed=$2
resample=$3
opts=${@:4:${#}}

prefix="$task seed $seed resample $resample $opts"
./train.py -t $prefix
./plot.py full -t $prefix -s


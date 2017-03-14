#!/usr/bin/env bash
set -e
./train.py -t $@ causal-model -t $@
./plot.py compare -t $@ causal-model -t $@


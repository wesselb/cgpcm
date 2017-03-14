#!/usr/bin/env bash
set -e
./train.py -t $@
./plot.py full -t $@ -s


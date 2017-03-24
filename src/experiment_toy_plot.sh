#!/usr/bin/env bash

from_resample=0
to_resample=40

for (( i = $from_resample; i <= $to_resample; i++ )); do
    # Causal sample
    ./core.py -t toy resample $i causal-sample causal-model \
              -t toy resample $i causal-sample \
              -p compare2 index1 0 index2 0 mf1 correct-h soft-correct-h \
              -p compare2 index1 1 index2 1 mf1 correct-h soft-correct-h \
              -p compare2 index1 0 index2 1 correct-h soft-correct-h \
              -r


    # Acausal sample
    ./core.py -t toy resample $i causal-model \
              -t toy resample $i \
              -p compare2 index1 0 index2 0 mf1 correct-h soft-correct-h \
              -p compare2 index1 1 index2 1 mf1 correct-h soft-correct-h \
              -p compare2 index1 0 index2 1 correct-h soft-correct-h \
              -r
done

#!/usr/bin/env bash

from_offset=$1
to_offset=$2
offset_increment=$3

from_length=$4
to_length=$5
length_increment=$6

opts=${@:7:${#}}

for (( len = $from_length; len <= $to_length; len += $length_increment )); do
    echo Length: $len
    for (( offset = $from_offset; offset <= $to_offset; offset += $offset_increment )); do
        echo Offset: $offset
        ./controller.py -t crude offset $offset length $len causal-model \
                        -t crude offset $offset length $len \
                        -p compare index0 0 index1 0 mf0 \
                        -p compare index0 1 index1 1 mf0 \
                        -p compare index0 0 index1 1 \
                        $opts
    done
done

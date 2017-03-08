#!/usr/bin/env bash

function title {
    echo ">> $1"
}

title "Generating documentation source"
python generate_doc_source.py
title "Building documentation"
make clean html  # latexpdf

if [ "$1" = "open" ]; then
    open build/html/index.html
fi

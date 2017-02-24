#!/usr/bin/env zsh
for f in *.pdf; do
    pdfcrop --margins 0 $f cropped/$f
done

#!/usr/bin/env bash
PYTHONPATH=`pwd` python learn/$1.py ${@:2}

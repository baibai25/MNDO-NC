#!/bin/sh
path=Predataset/

for file in `ls Predataset/`
do
    argv=$path$file
    echo $file
    python mndo.py $argv
done

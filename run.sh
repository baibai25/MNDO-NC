#!/bin/sh
path=Predataset/

for file in `ls Predataset/`
do
    argv=$path$file
    echo $file
    python train.py $argv
done

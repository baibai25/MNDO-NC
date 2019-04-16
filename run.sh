#!/bin/sh
data=Predataset/
gen=generated/

for file in `ls Predataset/`
do
    argv_data=$data$file
    argv_gen=$gen$file
    echo $file
    #echo $argv_data $argv_gen
    
    python train.py $argv_data $argv_gen
done

#!/bin/sh
dataset=("abalone" "sick_euthyroid" "thyroid_sick" "arrhythmia" "abalone_19")

for file in ${dataset[@]}
do
    echo $file
    python shape.py $file
done

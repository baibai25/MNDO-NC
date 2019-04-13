#!/bin/sh
dataset=("abalone" "sick_euthyroid" "thyroid_sick" "arrhythmia" "abalone_19")

for file in ${dataset[@]}
do
    echo $file
    python src/load_data.py $file
done

#!/bin/bash

workers=(1 2)

echo "workers,wall_time,accuracy" > results.csv

for w in "${workers[@]}"
do

echo "================================="
echo "Running experiment with $w workers"
echo "================================="

output=$(python parameter_server.py --n_workers $w --epochs 25)

result=$(echo "$output" | grep RESULT)

workers=$(echo $result | awk '{print $2}')
wall=$(echo $result | awk '{print $3}')
acc=$(echo $result | awk '{print $4}')

echo "$workers,$wall,$acc" >> results.csv

sleep 10

done

echo ""
echo "Experiments finished"
echo "Results saved in results.csv"
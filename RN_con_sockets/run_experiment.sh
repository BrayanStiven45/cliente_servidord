#!/bin/bash

workers=(4 2 1)

for w in "${workers[@]}"
do

echo "Running experiment with $w workers"

python parameter_server.py --workers $w --epochs 600

sleep 10

done
#!/bin/bash

# Primero crear todos los CSV con encabezado
for p in {1..12}
do
    OUTPUT="results_${p}.csv"
    echo "run,time,accuracy" > $OUTPUT
done

echo "Starting experiments..."

# Ahora el loop externo es el número de run
for run in {1..10}
do
    echo "Run $run..."

    # En cada run se recorren todos los procesos
    for p in {1..12}
    do
        OUTPUT="results_${p}.csv"

        echo "   Testing $p processes..."

        result=$(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                 python RN_Diego_version_2.py --processes $p)

        time=$(echo "$result" | grep -oP 'time=\K[0-9.]+')
        acc=$(echo "$result" | grep -oP 'accuracy=\K[0-9.]+')

        echo "$run,$time,$acc" >> $OUTPUT
    done
done

echo "All experiments completed."

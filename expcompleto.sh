#!/bin/bash

STREAMS=(1 2 4 8 16)

N=32

# echo "Ejecutando primer experimento"
# for ((i=1; i<=N; i++)); do
#     ./exp1
# done 

echo "Ejecutando segundo experimento"
for s in "${STREAMS[@]}"; do
    # for ((i=1; i<=N; i++)); do
        nsys profile -o reporte_solapamiento_"$s"s ./exp2 -s "$s"
    # done
done


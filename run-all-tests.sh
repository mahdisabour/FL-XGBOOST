#!/bin/bash

for n_client in 8
do
    ./run.sh N_CLIENTS=$n_client
    echo "./logs/file_xgboost_${n_client}.log" >> results.txt
    tail -n 1200 "./logs/file_xgboost_${n_client}.log" >> results.txt;
    echo "***********************************************************" >> results.txt
done
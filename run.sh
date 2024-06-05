#!/bin/bash

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

N_CLIENTS=$N_CLIENTS

log_file="./logs/file_xgboost_${N_CLIENTS}.log"

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
python server.py --n-clients $N_CLIENTS &> $log_file &
sleep 40  # Sleep for 5s to give the server enough time to start

for i in $(seq 0 $((N_CLIENTS-1))); do
    echo "Starting client $i"
    python client.py --partition-id=$i --n-clients $N_CLIENTS &> /dev/null &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

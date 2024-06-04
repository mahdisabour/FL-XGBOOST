#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
python server.py &
sleep 30  # Sleep for 5s to give the server enough time to start

for i in `seq 0 7`; do
    echo "Starting client $i"
    python client.py --partition-id=$i &> /dev/null &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

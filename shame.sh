#!/bin/bash

# User to analyze
USER="xwei"

echo "------------------------------------------------"
echo "Queue Analysis for user: $USER"
echo "------------------------------------------------"
echo "COUNT | STATE | NODES PER JOB"
echo "------------------------------------------------"

# 1. Get queue info for user
# -u: filter by user
# -h: no header
# -t: filter states (PD=Pending, R=Running)
# -o: format output (%t=State, %D=NumNodes)
# 2. Sort lines to group them
# 3. Count unique occurrences
squeue -u $USER -t PENDING,RUNNING -h -o "%t %D" | sort | uniq -c | \
while read count state nodes; do
    # Just cosmetic formatting to make it readable
    if [[ "$state" == "R" ]]; then
        STATE_LABEL="RUNNING"
    elif [[ "$state" == "PD" ]]; then
        STATE_LABEL="PENDING"
    else
        STATE_LABEL="$state"
    fi
    
    printf "%5s | %-7s | %s node(s)\n" "$count" "$STATE_LABEL" "$nodes"
done

echo "------------------------------------------------"
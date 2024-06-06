#!/bin/bash
# Shuts down all connected cameras
# To shut down specific cameras, enter their IDs as arguments separated by spaces

if [[ $# -eq 0 ]]
then
    ids="1 2 3 4 5"
else
    ids=$*
fi

for id in $ids
do
    ssh nist@cam$id.local "sudo shutdown now"
done
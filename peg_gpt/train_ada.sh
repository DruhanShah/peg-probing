#!/bin/bash

mkdir -p /scratch/temp-PEGs/{data,models}
if [ $1 = "data" ]; then
    python3 data.py --work-dir /scratch/temp-PEGs
elif [ $1 = "train" ]; then
    python3 train.py --work-dir /scratch/temp-PEGs --lang $2
fi
rsync -t /scratch/temp-PEGs/* druhan@ada:/share1/druhan/pegs/

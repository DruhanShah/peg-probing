#!/bin/bash

mkdir -p /scratch/pegs/{data,models}
if [ $1 == "data" ]; then
    python3 data.py --work_dir /scratch/temp-PEGs
    scp -r /scratch/pegs/data ada:/share1/druhan/pegs/
elif [ $1 == "train" ]; then
    scp -r ada:/share1/druhan/pegs/data/$2.txt /scratch/pegs/data/
    python3 peg_gpt_trainer.py --work_dir /scratch/pegs --lang $2
    scp -r /scratch/pegs/models ada:/share1/druhan/pegs/
fi

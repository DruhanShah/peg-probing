#!/bin/bash

mkdir -p /scratch/pegs/{data/corpus,model/models,data/gens}
if [ $1 == "data" ]; then
    python3 data.py --work_dir /scratch/temp-PEGs
    scp -r /scratch/pegs/data/corpus ada:/share1/druhan/pegs/data
elif [ $1 == "train" ]; then
    scp -r ada:/share1/druhan/pegs/data/corpus/$2.txt /scratch/pegs/data/corpus
    python3 peg_gpt_trainer.py --work_dir /scratch/pegs --lang $2
    scp -r /scratch/pegs/model/models ada:/share1/druhan/pegs/model
elif [ $1 == "dist" ]; then
    scp -r ada:/share1/druhan/pegs/models/models/$2_model_$3.pt /scratch/pegs/model/models/
    python3 test_dist.py --work_dir /scratch/pegs --lang $2 --model $2_model_$3
    scp -r /scratch/pegs/data/gens/ ada:/share1/druhan/pegs/data
fi

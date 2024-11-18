#!/bin/bash

mkdir -p /scratch/pegs/{data/corpus,model/models,data/gens}
if [ $1 == "data" ]; then
    python3 data/generate.py --work_dir /scratch/pegs
    scp -r /scratch/pegs/data/corpus ada:/share1/druhan/pegs/data
elif [ $1 == "train" ]; then
    scp -r ada:/share1/druhan/pegs/data/corpus/$2.txt /scratch/pegs/data/corpus
    python3 train_model.py --work_dir /scratch/pegs --lang $2
    scp /scratch/pegs/model/models/$2_model_* ada:/share1/druhan/pegs/model/models
elif [ $1 == "dist" ]; then
    scp -r ada:/share1/druhan/pegs/model/models/$2_model_$3.pt /scratch/pegs/model/models/
    python3 test_dist.py --work_dir /scratch/pegs --lang $2 --model $2_model_$3
    scp /scratch/pegs/data/gens/$2.txt ada:/share1/druhan/pegs/data/gens
fi

#!/bin/sh

scp -r ada:/share1/druhan/druhan.probing /scratch/
python3 $1
scp -r /scratch/druhan.probing ada:/share1/druhan/

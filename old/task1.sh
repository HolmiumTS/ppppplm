#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=bertweet
#SBATCH --output=out
#SBATCH --get-user-env

for ((i=0;i<31;i++));do
{
    bunzip2 -kc ../bertweet_data/$i.bz2 > ../bertweet_data/$i.txt && python task1.py ../bertweet_data/$i.txt ../bertweet_data/$i.json
} &
done


for ((i=0;i<31;i++));do
{
    cat ../bertweet_data/$i.json >> ../bertweet_data/data.json
}
done

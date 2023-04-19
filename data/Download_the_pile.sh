#!/bin/bash
start=0
end=29
mkdir data/the_pile
for (( i=$start; i<=$end; i++ ))
do
    url="https://the-eye.eu/public/AI/pile/train/$(printf "%02d" $i).jsonl.zst"
    echo "Downloading file: $url"
    curl -C - $url -o data/the_pile/"$(printf "%02d" $i).jsonl.zst"
done

wait

echo "All files downloaded successfully."
mkdir data/pretrain_data
python3 data/preprocess_the_pile.py
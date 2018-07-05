#!/bin/bash
#convert pickle to tsv text
python pickle2tsv.py
# remove broken data
cat dataset.txt | sed '29490d' | sed '40929d' > dataset_clean.txt
# length filter
python filter.py

cat dataset_clean.txt | shuf > dataset_random.txt
cat dataset_random.txt | sed -n '3001,$p' > train.txt
cat dataset_random.txt | head -n 3000 > valid.txt

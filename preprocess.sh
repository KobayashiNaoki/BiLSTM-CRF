#!/bin/bash

trap 'rm $tmp1 $tmp2' ERR EXIT

tmp1=$(mktemp)
tmp2=$(mktemp)

#convert pickle to tsv text
python tools/pickle2tsv.py data/dataset.pkl data/dataset.tsv
# remove broken data
cat data/dataset.tsv | sed '29490d' | sed '40929d' > $tmp1
# length filter
cat $tmp1 | python tools/length_filter.py > $tmp2
# ABIO convert to BIO
cat $tmp2 | python tools/abio2bio.py > $tmp1
# random sort, make train and valid file
cat $tmp1 | shuf > $tmp2
cat $tmp2 | sed -n '3001,$p' > data/train.txt
cat $tmp2 | head -n 3000 > data/valid.txt

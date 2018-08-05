#!/bin/bash

trap 'rm $tmp1 $tmp2' ERR EXIT

tmp1=$(mktemp)
tmp2=$(mktemp)

# convert pickle to tsv text
python tools/pickle2tsv.py data/dataset.pkl data/dataset.tsv
# remove broken data
cat data/dataset.tsv | sed '29490d' | sed '40929d' > $tmp1
# length filter
cat $tmp1 | python tools/length_filter.py > $tmp2
# ABIO convert to BIO
cat $tmp2 | python tools/abio2bio.py > $tmp1
# Add POS, postion features
cat $tmp1 | python tools/add_features.py > $tmp2
# random sort, make train and valid file
cat $tmp2 | shuf > $tmp1
cat $tmp1 | sed -n '3001,$p' > data/train.txt
cat $tmp1 | head -n 3000 > data/valid.txt

# make test
cat data/test.origin | python tools/dummy_tag.py > $tmp1
cat $tmp1 | python tools/add_features.py > data/test.txt

#!/bin/bash

cd ./ner/

for i in 1 2 3 4 5 6 7 8
do
  python3 ./merge_benchmarks.py --input_path ../data/processedsI/processed_$i/ --output_path ../data/processedsO/processed_$i/
done
#!/bin/sh
python training.py --csv_train ./csv/test.csv  --csv_val ./csv/test.csv --datareader tifffile --device cuda:0 --batch_size 4 --module UwU --init_weights --lr 0.001 --iter 500
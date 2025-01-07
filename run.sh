#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./cot_gat_tlr.py  -m "EleutherAI/pythia-1b" -d "deepmind/aqua_rat"


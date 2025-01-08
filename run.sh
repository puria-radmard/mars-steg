#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./custom_trainer.py  -m "EleutherAI/pythia-2.8b" -d "deepmind/aqua_rat"

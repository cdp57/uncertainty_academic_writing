#!/usr/bin/env python3
"""Cheng Demo 2 collection: LLAMA on QASPER (support only).

Example: python llama_qasper.py --model 1B --seed 42 --num-shards 4 --shard-index 0
"""
from experiment_core import run

if __name__ == "__main__":
    run("llama", "qasper")

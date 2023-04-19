#!/bin/bash
mkdir data/instruction_data
curl -C - --retry 3 'https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/sg_90k_part1.json' -o data/sg_90k_part1.json
curl -C - --retry 3 'https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/sg_90k_part2.json' -o data/sg_90k_part2.json
python3 data/preprocess_instruction.py
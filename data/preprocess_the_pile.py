#!/usr/bin/env python
# -- coding: utf-8 --
# @Time :2023/4/19 13:51
# @File : preprocess_the_pile.py

import json
from glob import glob
from tqdm import tqdm
import zstandard as zstd

paths = glob("data/the_pile/*.jsonl.zst")
write_path = "data/pretrain_data/part-pile-{}.jsonl.zst"
total_num = 0
file_num = 0
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for path in tqdm(paths, total=len(paths)):
    with zstd.open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if total_num % 16384 == 0 and total_num > 0:
                file_num += 1
                wfp.close()
                wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
            wfp.write(line.encode("utf-8"))
            total_num += 1
wfp.close()
print("total line: {}\ntotal files: {}".format(total_num, file_num))

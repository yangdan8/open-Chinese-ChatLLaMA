import json
from glob import glob
from tqdm import tqdm
import zstandard as zstd

paths = glob("data/WuDaoCorpus2.0_base_200G/part*")
write_path = "data/pretrain_data/part-wudao-{}.jsonl.zst"
total_num = 0
file_num = 0
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for path in tqdm(paths, total=len(paths)):
    with open(path, "r") as fp:
        data = json.load(fp)
    for line in data:
        if total_num % 16384 == 0 and total_num > 0:
            file_num += 1
            wfp.close()
            wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
        wfp.write(json.dumps(line).encode("utf-8"))
        wfp.write("\n".encode("utf-8"))
        total_num += 1
wfp.close()
print("total line: {}\ntotal files: {}".format(total_num, file_num))
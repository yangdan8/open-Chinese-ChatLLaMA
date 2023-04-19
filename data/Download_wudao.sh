#!/bin/bash
apt install unrar
for i in {1..100}
do
  curl -C - --retry 100 'https://dorc.baai.ac.cn/resources/data/WuDaoCorpora2.0/WuDaoCorpus2.0_base_200G.rar?AccessKeyId=AKLTNasiLRBBTcOgPqzlkPzu1w&Expires=1679127659&Signature=7jh%2FpnJyC2hAeumm9EjaeE5HN9E%3D' -o data/WuDaoCorpus2.0_base_200G.rar
done
unrar x data/WuDaoCorpus2.0_base_200G.rar
mkdir data/pretrain_data
python3 data/preprocess_wudao.py
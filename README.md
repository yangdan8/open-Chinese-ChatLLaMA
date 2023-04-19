# open-Chinese-ChatLLaMA
Open-Llama是一个开源项目，提供了一整套用于构建大型语言模型的训练流程及代码，从数据集准备到分词、预训练、指令调优，以及强化学习技术 RLHF。
## **要求**

- Python 3.7 或更高版本
- PyTorch 1.13
- 特殊版本的[Transformers库](https://github.com/Bayes-Song/transformers)
- [Accelerate库](https://huggingface.co/docs/accelerate/index)
- CUDA 11.6 或更高版本（用于 GPU 加速，基于CUDA11.7进行测试）

## **入门指南**
### 安装

使用下面的命令安装相关依赖

```bash
pip install -r requirements.txt
```

### 数据集准备

目前给出了智源开源的悟道数据集和EleutherAI开源的the pile数据集。数据集下载和处理代码在data目录下。
其中悟道数据集由于需要同意一些协议才能下载因此可能需要修改一下download_wudao中的链接，[悟道](https://data.baai.ac.cn/details/WuDaoCorporaText)。

运行下面的命令进行数据下载并进行分片
```bash
bash data/download_the_pile.sh
bash data/download_wudao.sh
```
数据将按照每个文件最大16384行存储为小文件，便于后续使用多进程训练时进行读取。存储格式为jsonl.zst，使用zstd进行压缩，最终数据大小为519.5G，合计16466个文件。

其中the pile数据集包含210607728行json line，悟道数据集包含59132213行json line。

具体数据格式如下
```
WuDao
{'id': 1, 'dataType': '百科', 'title': 'some title', 'content': 'some content'}

The Pile
{'text': 'some text', 'meta': {'pile_set_name': 'Github'}}
```

### 数据读取
数据读取相关代码可见dataset目录，其中包含根据下载的数据集使用SentencePiece训练分词模型，以及根据分词器构建DataLoader。

训练分词器使用如下命令
```bash
python3 dataset/train_tokenizer.py
```

使用如下命令查看DataLoader输出的结果
```bash
python3 dataset/pretrain_dataset.py
```
### 模型结构
我们基于Transformers库中的[Llama](https://github.com/facebookresearch/llama)

同时我们还参考了[Bloom](https://huggingface.co/bigscience/bloom)，对于Token Embedding引入了Stable Embedding以更好的稳定训练。

最后我们参考[PALM](https://arxiv.org/abs/2204.02311)，使用了Shared Input-Output Embeddings。

### 预训练
我们基于Accelerate库进行多GPU并行训练，启动命令如下
```bash
accelerate launch --config_file configs/default_config.yaml pretrain_llama.py
```
某些情况下可能需要指定下列参数
```
--main_process_ip
--main_process_port
--num_processes
--num_machines
--machine_rank
```
我们使用[Wandb](https://wandb.ai/)进行训练的可视化，需要自行修改环境变量 WANDB_API_KEY 。

其中我们使用了DeepSpeed stage1以减少显存占用。accelerate相关配置可见configs/default_config.yaml。

训练相关超参数可见configs/train_config.py，目前我们使用10W词表的7B Llama模型进行训练，具体配置如下

| max_length | batch_size | learning_rate | weight_decay | params | dimension | n heads | n layer | vocab_size |
|------------|------------------|---------------|--------------|--------|-----------|---------|---------|------------|
| 1024       | 2                | 2e-4          | 1e-1         | 6.88B  | 4096      | 32      | 32      | 100000     |

```
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
LlamaForCausalLM                                        [1, 64, 32, 128]          --
├─LlamaModel: 1-1                                       [1, 64, 32, 128]          --
│    └─Embedding: 2-1                                   [1, 64, 4096]             409,600,000
│    └─LayerNorm: 2-2                                   [1, 64, 4096]             8,192
│    └─ModuleList: 2-3                                  --                        --
│    │    └─LlamaDecoderLayer: x32                      [1, 64, 4096]             202,383,360 x 32
│    └─LlamaRMSNorm: 2-4                                [1, 64, 4096]             4,096
=========================================================================================================
Total params: 6,885,879,808
Trainable params: 6,885,879,808
Non-trainable params: 0
Total mult-adds (G): 6.89
```

具体训练代码和预训练基本一样，代码可见
```
instruction_tuning.py
```

启动命令也基本一致
```bash
accelerate launch --config_file configs/default_config.yaml instruction_tuning.py
```
某些情况下可能需要指定下列参数
```
--main_process_ip
--main_process_port
--num_processes
--num_machines
--machine_rank
```

### RLHF
4月底完成
### Server

单轮对话使用server.py，对于多轮对话使用chat_server.py

基于Gradio开发。 
## 性能对比

### 训练框架
在训练框架方面我们测试了HuggingFace开源的Accelerate库和HPC-AI开源的ColossalAI，我们测试在打满显卡时性能差异较小。因此最终选择了实现相对简单的Accelerate库作为训练框架

测试数据如下，测试过程中使用的模型结构为
| Model | n gpu | n layer | n heads | hidden size | vocab size | seq length |
|-------|-------|---------|---------|-------------|------------|------------|
| GPT2  | 2     | 6       | heads   | 4096        | 250100     | 1024       |

测试结果如下，可以看到当打满时速度和显存相差不大
|                 | HuggingFace                       | HuggingFace                        | ColossalAI                                             | ColossalAI                                             | ColossalAI                         |
|-----------------|-----------------------------------|------------------------------------|--------------------------------------------------------|--------------------------------------------------------|------------------------------------|
| config          | without activation ckpt, bs2      | without activation ckpt, max_bs=12 | with activation ckpt, bs2                              | without activation ckpt, bs2                           | without activation ckpt, max_bs=10 |
| second pre step | 0.336, fw=0.033, bw=0.3, opt=5e-6 | 1.25                               | 0.347                                                  | 0.308, fw=0.067, bw=0.152, opt=0.088                   | 1.055                              |
| gpu memory      | nvidia-smi 45445                  |                                    | fw+bw+opt=21053.63+22064.12+17987.52, nvidia-smi 40961 | fw+bw+opt=24684.74+21087.13+17987.52, nvidia-smi 46821 | oom after 10 steps, 疑似有内存泄漏 |

### 性能优化
在最早版本中我们使用DeepSpeed stage2 + Transformers中的原生Llama实现进行训练但是速度和论文中所说的相差较大，因此后续我们进行了一系列的优化，我们将每一步的性能提升列在下面可供参考。

论文中提到对于6.7B模型使用了1T token进行训练，最终的gpu时为82432，因此可以计算出他的训练速度大致为3370 token/s/gpu。
当使用下面的优化后速度开源基本和论文中速度一致，使用20x8 A100-80G进行测试。预计加入更多融合算子开源取得更好的性能。

|                     | V1           | V2                    |
|---------------------|--------------|-----------------------|
| Model               | Transformers | Transformers+xformers |
| Optimizer           | Pytorch Adam | Fused Adam            |
| DeepSpeed           | stage2       | stage1                |
| Grad Accumulation   | 4            | 12                    |
| Return Padding Mask | yes          | no                    |
| Speed token/s/gpu   | 1378         | 3290                  |

### 和其他开源模型性能对比
下表是一个对目前开源模型性能的一个总结，使用GPU device均为A100，由于模型大小各不相同结构也有一定差异，难以准确的对比性能，作为一个粗略估计可以认为速度和模型参数量基本呈反比关系，这一点看Llama不同大小的模型可以得到印证。基于这个粗略估计可以看到使用本项目的性能明显由于其他项目。

| Model               | Open-Llama | LLAMA    | LLAMA   | LLAMA     | OPT     | Bloom              | GLM   | GPT-NEOX | CPM-ANT | CodeGeeX  |
|---------------------|------------|----------|---------|-----------|---------|--------------------|-------|----------|---------|-----------|
| Model size          | 6.9B       | 6.7B     | 13B     | 65B       | 175B    | 175B               | 130B  | 20B      | 10B     | 13B       |
| Token               |            | 1T       | 1T      | 1.4T      | 180B    | 366B               | 400B  | 402B     | 200B    | 13.9B     |
| GPU Hour            |            | 82,432   | 135,168 | 1,022,362 | 809,472 | 1,082,990          | 43776 | 175680   | 47040   | 3072      |
| speed token/s/gpu   | 3290       | 3370     | 2055    | 380       | 61.8    | 93.9               | 105.7 | 635.6    | 1181    | 1257      |
| 相关依赖            | xformers   | xformers |         |           | measeq  | Megatron-DeepSpeed |       |          | BMtrain | MindSpore |
| speed token/s/gpu/B | 22701      | 22579    | 26715   | 24700     | 10815   | 16432              | 13741 | 12712    | 11810   | 16341     |

## 后续计划

1. 加入更多训练监控，比如训练数据类别的分布等，加入继续训练相关代码
2. 实现Instruction-tuning代码，并开源相关checkpoint
3. 使用Gradio搭建在线Demo
4. 基于模型开发chatpaper等chat工具
5. 加入根据Common Crawl构建预训练数据集相关代码，并开源相关数据集
6. 加入多模态训练代码
7. ......



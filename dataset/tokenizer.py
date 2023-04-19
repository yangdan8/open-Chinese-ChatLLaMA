import torch


class Tokenizer:
    def __init__(self, sp_model):
        self.sp_model = sp_model
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()
        self.vocab_size = self.sp_model.vocab_size()

    def __call__(
        self,
        inputs,
        padding=None,
        max_length=256,
        return_tensors=False,
        truncation=False,
        add_special_tokens=True,
        return_mask=False,
    ):
        if isinstance(inputs, str):
            return self.encode(
                inputs,
                padding=padding,
                max_length=max_length,
                return_tensors=return_tensors,
                truncation=truncation,
                add_special_tokens=add_special_tokens,
                return_mask=return_mask,
            )
        else:
            return self.encode_batch(
                inputs,
                padding=padding,
                max_length=max_length,
                return_tensors=return_tensors,
                truncation=truncation,
                add_special_tokens=add_special_tokens,
                return_mask=return_mask,
            )

    def encode(
        self,
        inputs,
        padding=None,
        max_length=8192,
        return_tensors=False,
        truncation=False,
        add_special_tokens=True,
        return_mask=False,
    ):
        assert isinstance(inputs, str)
        input_ids = self.sp_model.Encode(inputs)
        if return_mask:
            attention_mask = [1] * len(input_ids)
        if truncation:
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L780
            # 参考Transformer中的实现 默认最后一位一定是pad或者eos
            input_ids = input_ids[: max_length - 1]
            if return_mask:
                attention_mask = attention_mask[: max_length - 1]
        if add_special_tokens:
            input_ids = input_ids + [self.eos_id]
            if return_mask:
                attention_mask = attention_mask + [0]
        if padding == "max_length":
            input_ids = input_ids + [self.pad_id] * (max_length - len(input_ids))
            if return_mask:
                attention_mask = attention_mask + [0] * (
                    max_length - len(attention_mask)
                )
        if return_tensors:
            input_ids = torch.tensor([input_ids])
            out = {
                "input_ids": input_ids,
            }
            if return_mask:
                attention_mask = torch.tensor([attention_mask])
                out["attention_mask"] = attention_mask
        else:
            out = {
                "input_ids": input_ids,
            }
            if return_mask:
                out["attention_mask"] = attention_mask
        return out

    def encode_batch(
        self,
        inputs,
        padding=None,
        max_length=8192,
        return_tensors=False,
        truncation=False,
        add_special_tokens=True,
        return_mask=False,
    ):
        input_ids = self.sp_model.Encode(inputs)
        if return_mask:
            attention_mask = [[1] * len(i) for i in input_ids]
        if truncation:
            input_ids = [i[: max_length - 1] for i in input_ids]
            if return_mask:
                attention_mask = [i[: max_length - 1] for i in attention_mask]
        if add_special_tokens:
            input_ids = [i + [self.eos_id] for i in input_ids]
            if return_mask:
                attention_mask = [i + [0] for i in attention_mask]
        if padding == "max_length":
            input_ids_pad = []
            if return_mask:
                attention_mask_pad = []
            for idx, i in enumerate(input_ids):
                input_ids_pad.append(i + [self.pad_id] * (max_length - len(i)))
                if return_mask:
                    j = attention_mask[idx]
                    attention_mask_pad.append(j + [0] * (max_length - len(j)))
            input_ids = input_ids_pad
            if return_mask:
                attention_mask = attention_mask_pad
        if return_tensors:
            input_ids = torch.tensor(input_ids)
            out = {
                "input_ids": input_ids,
            }
            if return_mask:
                attention_mask = torch.tensor(attention_mask)
                out["attention_mask"] = attention_mask
        else:
            out = {
                "input_ids": input_ids,
            }
            if return_mask:
                out["attention_mask"] = attention_mask
        return out

    def decode(self, inputs, max_rounds=None):
        inputs = inputs.tolist()
        out = []
        for i, ids in enumerate(inputs):
            count = 0
            flag = False
            for j, token in enumerate(ids):
                if token == self.eos_id:
                    if max_rounds is None:
                        flag = True
                        break
                    elif isinstance(max_rounds, int):
                        if count < max_rounds:
                            count += 1
                        else:
                            flag = True
                            break
                    elif isinstance(max_rounds, list):
                        if count < max_rounds[i]:
                            count += 1
                        else:
                            flag = True
                            break
            if flag:
                ids = ids[:j]
            else:
                ids = ids
            out.append(ids)
        out = self.sp_model.Decode(out)
        return out


if __name__ == "__main__":
    import sentencepiece as spm
    from unicodedata import normalize

    # Using sentencepiece may not be able to process some reserved keywords like '▁'.
    sp_model = spm.SentencePieceProcessor(
        model_file="configs/10w_vocab_wudao5_pile10.model"
    )
    tokenizer = Tokenizer(sp_model)
    tmp = [
        "hello world",
        "这是开源项目的V1版本，this is the first version of a open-source project!",
        "# this is a python script\nfor i in range(10):\n   print(i)\n   for j in range(10):\n       print(j)",
    ]
    print(tmp)
    out = tokenizer(
        tmp, padding="max_length", return_tensors=True, max_length=64, truncation=True
    )
    for k, v in out.items():
        print(k, v.shape)
    print(out["input_ids"])
    out = tokenizer.decode(out["input_ids"])
    print(out)
    for i, j in zip(tmp, out):
        assert normalize("NFKC", i) == j

    from dataset.data_iter import create_shard_kwargs, DataIter

    patterns = ["data/pretrain_data/part-wudao*.jsonl.zst"]
    paths = create_shard_kwargs(patterns)
    data_iter = DataIter(paths)
    for i, data in enumerate(data_iter):
        assert (
            normalize("NFKC", data["content"])
            == sp_model.Decode(sp_model.Encode(data["content"]))
            or "▁" in data["content"]
        )
        if i == 1000:
            break

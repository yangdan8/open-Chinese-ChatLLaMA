import math


def preprocess_wudao_gen(tokenizer, segment_max_length=1024):
    def preprocess_wudao(line):
        """
        The format of the data is roughly as follows.
        {'id': 1, 'dataType': '百科', 'title': 'some title', 'content': 'some content'}
        Split the data based on the tokenized length according to the maximum length.
        """
        total = line["title"] + "\n" + line["content"]
        out = tokenizer(total)
        input_ids = out["input_ids"]
        return [
            input_ids[i * segment_max_length : (i + 1) * segment_max_length]
            for i in range(math.ceil(len(input_ids) / segment_max_length))
        ]

    return preprocess_wudao


def preprocess_the_pile_gen(tokenizer, segment_max_length=1024):
    def preprocess_the_pile(line):
        """
        The format of the data is roughly as follows.
        {'text': 'some text', 'meta': {'pile_set_name': 'Github'}}
        Split the data based on the tokenized length according to the maximum length.
        """
        total = line["text"]
        out = tokenizer(total)
        input_ids = out["input_ids"]
        return [
            input_ids[i * segment_max_length : (i + 1) * segment_max_length]
            for i in range(math.ceil(len(input_ids) / segment_max_length))
        ]

    return preprocess_the_pile


if __name__ == "__main__":
    import sentencepiece as spm

    from dataset.tokenizer import Tokenizer
    from dataset.data_iter import create_shard_kwargs, DataIter

    sp_model = spm.SentencePieceProcessor(
        model_file="configs/10w_vocab_wudao5_pile10.model"
    )
    tokenizer = Tokenizer(sp_model)
    patterns = ["data/pretrain_data/part-*.jsonl.zst"]
    paths = create_shard_kwargs(patterns)
    transform_dict = {
        "wudao": preprocess_wudao_gen(tokenizer),
        "pile": preprocess_the_pile_gen(tokenizer),
    }
    data_set = DataIter(
        paths, transform_dict=transform_dict, concat_docs=True, max_length=1024
    )
    for sample in data_set:
        print(sample)
        break

# # The language modeling task is to assign a probability for the likelihood of a given word 
# # (or a sequence of words) to follow a sequence of words.

# from torchtext.datasets import WikiText2
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

# train_iter = WikiText2(split='train')
# tokenizer = get_tokenizer('basic_english')
# vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
# vocab.set_default_index(vocab['<unk>'])

# def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
#     """Converts raw text into a flat Tensor."""
#     data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
#     return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# # ``train_iter`` was "consumed" by the process of building the vocab,
# # so we have to create it again
# train_iter, val_iter, test_iter = WikiText2()
# train_data = data_process(train_iter)
# val_data = data_process(val_iter)
# test_data = data_process(test_iter)

# print(train_data)


from data.dataset import TextIterableDataset
from readers.parquet_reader import ShardInfo
from torch.utils.data import DataLoader

dataset = TextIterableDataset([ShardInfo("/Users/raghavakannikanti/opensource_2024/neural_networks_apps/example.parquet", [0], ["one"])])
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size)

# Iterate over the DataLoader
for batch in dataloader:
    print("Batch:", batch)
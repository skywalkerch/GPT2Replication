import tiktoken as tk
from .GPTDataset import GPTDatasetV1
from torch.utils.data import DataLoader
import os


os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(os.getcwd(), ".tiktoken")


def create_dataloader(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tk.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader

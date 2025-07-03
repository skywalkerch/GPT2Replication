from model.config.config import GPT_CONFIG_774M
from model.config.config import hyper_param
from model.core.GPTModel import GPT2Model
import torch
from libs.GenerateText import GenerateTextSimple
import tiktoken as tk
import os
from libs.GPTDownload import download_and_load_gpt2
from libs.load_weights import load_weights_into_gpt

settings, params = download_and_load_gpt2(model_size="774M", models_dir="gpt2")

os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(os.getcwd(), ".tiktoken")
tokenizer = tk.get_encoding("gpt2")
hp = hyper_param

cfg = GPT_CONFIG_774M
model = GPT2Model(cfg)
load_weights_into_gpt(model, params)
model.cuda()
model.eval()

while(True):
    input_str = input(">:")
    input_ids = torch.tensor([tokenizer.encode(input_str)], device="cuda")
    idx = GenerateTextSimple(model, input_ids, 20, cfg["context_length"])
    for i in idx:
        print(tokenizer.decode(i.tolist()))

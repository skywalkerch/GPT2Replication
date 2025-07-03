GPT_CONFIG_124M = {
    "embedding_dim": 768,
    "context_length": 1024,
    "num_heads": 12,
    "num_layers": 12,
    "qkv_bias": True,
    "dropout": 0.1,
    "vocab_size": 50257,
}

GPT_CONFIG_1558M = {
    "embedding_dim": 1600,
    "context_length": 1024,
    "num_heads": 25,
    "num_layers": 48,
    "qkv_bias": True,
    "dropout": 0.1,
    "vocab_size": 50257,
}
GPT_CONFIG_774M = {
    "embedding_dim": 1280,
    "context_length": 1024,
    "num_heads": 20,
    "num_layers": 36,
    "qkv_bias": True,
    "dropout": 0.1,
    "vocab_size": 50257,
}


hyper_param = {
    "batch_size": 2,
    "stride": 1,
    "num_workers": 0,
}

from typing import Dict
from typing import List
import re


class SimpleTokenizerV1:
    def __init__(self, vocab: Dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {integer: token for token, integer in vocab.items()}

    def encode(self, text: str) -> List[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return list(map(lambda x: self.str_to_int[x], preprocessed))

    def decode(self, text: List[int]) -> str:
        text = " ".join(map(lambda x: self.int_to_str[x], text))
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)


class SimpleTokenizerV2:
    def __init__(self, vocab: Dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {integer: token for token, integer in vocab.items()}

    def encode(self, text: str) -> List[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        return list(map(lambda x: self.str_to_int[x], preprocessed))

    def decode(self, text: List[int]) -> str:
        text = " ".join(map(lambda x: self.int_to_str[x], text))
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)

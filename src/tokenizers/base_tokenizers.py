import re
from abc import ABC, abstractmethod

from pydantic import BaseModel

from src.tokenizers.vocabulary import Vocabulary


class Tokenizer(ABC, BaseModel):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[str]) -> str:
        pass

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        pass


class RegularExpressionTokenizer(Tokenizer, BaseModel):
    pattern: str
    vocab: Vocabulary | None = None

    def fit(self, texts: list[str]) -> None:
        vocabulary = Vocabulary()
        for text in texts:
            matches = re.finditer(self.pattern, text)
            vocabulary.add([match.group() for match in matches])

        self.vocab = vocabulary

    def encode(self, text: str) -> list[int]:
        return [self.vocab[match.group()] for match in re.finditer(self.pattern, text)]

    def decode(self, token_ids: list[int] | int) -> str:
        if isinstance(token_ids, list):
            return "".join(self.vocab.idx_to_vocab[token_id] for token_id in token_ids)
        else:
            return self.vocab.idx_to_vocab[token_ids]


if __name__ == "__main__":
    tokenizer = RegularExpressionTokenizer(pattern=r"\w+")
    tokenizer.fit(["Hello, world!", "Hello, world!"])
    print(tokenizer.vocab)
    print(tokenizer.encode("Hello, world!"))
    print(tokenizer.decode([1, 2, 3]))

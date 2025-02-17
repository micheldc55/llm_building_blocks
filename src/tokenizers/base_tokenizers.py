import re
from abc import ABC, abstractmethod
from collections import Counter

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


class SimplestTokenizer(Tokenizer, BaseModel):
    num_steps: int
    vocab: dict | None
    merges: list[tuple]

    def fit(self, text: str) -> list[tuple]:
        list_text = list(text)
        merges = []
        for i in range(self.num_steps):
            counts_dict = self._create_pair_token_counts(list_text)

            if len(counts_dict) <= 1:
                break

            most_freq_pair = max(counts_dict.items(), key=lambda p: p[1])[0]

            merges.append(most_freq_pair)

            list_text = self.join_pair_in_word_list(list_text, most_freq_pair)

        self.merges = merges

    @staticmethod
    def _create_pair_token_counts(text_list: list[str]) -> dict[tuple, int]:
        counter_dict = Counter()

        for i in range(len(text_list) - 1):
            pair = (text_list[i], text_list[i + 1])

            counter_dict[pair] += 1

        return counter_dict

    @staticmethod
    def join_pair_in_word_list(text_list: list[str], pair: tuple) -> list[str]:
        new_list = []
        idx = 0

        while True:
            candidate_pair = (text_list[idx], text_list[idx + 1])

            if candidate_pair == pair:
                new_list.append("".join(pair))
                idx += 2
            else:
                new_list.append(text_list[idx])
                idx += 1

            if idx >= len(text_list) - 2:
                break

        return new_list


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

    tokenizer_simple = SimplestTokenizer()

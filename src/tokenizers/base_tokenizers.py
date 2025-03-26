import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable

from pydantic import BaseModel, Field

from src.tokenizers.utils.vocabulary import Vocabulary, VocabularyFactory


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

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


class WPETokenizer(Tokenizer, BaseModel):
    num_steps: int
    vocab_type: str = Field(default="slim")
    vocab: Vocabulary = Field(init=False, default_factory=Vocabulary)
    merges: list[tuple] = Field(init=False, default_factory=list)

    def fit(self, text: str) -> list[tuple]:
        list_text = list(text)
        merges = []

        vocab = self._create_base_vocab(text)

        for _ in range(self.num_steps):
            counts_dict = self._create_pair_token_counts(list_text)

            if len(counts_dict) <= 1:
                break

            most_freq_pair = max(counts_dict.items(), key=lambda p: p[1])[0]

            merges.append(most_freq_pair)
            list_text = self._join_pair_in_word_list(list_text, most_freq_pair)

        self.vocab = self._extend_vocab(vocab, merges=merges)
        self.merges = merges

        return merges

    def encode(self, text: str) -> list[int]:
        text_list = list(text)

        for merge in self.merges:
            text_list = self._join_pair_in_word_list(text_list, merge)

        return [self.vocab[token] for token in text_list]

    def decode(self, token_ids: Iterable[int] | int) -> str:
        if isinstance(token_ids, int):
            return self.vocab.get_token_from_id(token_ids)
        return "".join(self.vocab.get_token_from_id(token_id) for token_id in token_ids)

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    @staticmethod
    def _create_pair_token_counts(text_list: list[str]) -> dict[tuple, int]:
        counter_dict = Counter()

        for i in range(len(text_list) - 1):
            pair = (text_list[i], text_list[i + 1])
            counter_dict[pair] += 1

        return counter_dict

    @staticmethod
    def _join_pair_in_word_list(text_list: list[str], pair: tuple) -> list[str]:
        new_list = []
        idx = 0

        while idx < len(text_list) - 1:
            candidate_pair = (text_list[idx], text_list[idx + 1])

            if candidate_pair == pair:
                new_list.append("".join(pair))
                idx += 2
            else:
                new_list.append(text_list[idx])
                idx += 1

        if idx < len(text_list):
            new_list.append(text_list[idx])

        return new_list

    @staticmethod
    def _create_base_vocab(text: str) -> dict:
        vocab = VocabularyFactory.create(vocab_type="slim")
        sorted_text = sorted(set(text))
        vocab.add_tokens(sorted_text)
        return vocab

    @staticmethod
    def _extend_vocab(vocab: dict[str, int], merges: list[tuple]) -> dict[str, int]:
        vocab.add_tokens([subtoken1 + subtoken2 for subtoken1, subtoken2 in merges])

        return vocab


# TODO Implement BPE Tokenizer
# class BPETokenizer(Tokenizer, BaseModel):
#     vocab: dict | None = None
#     idx_to_vocab: dict | None = None
#     merges: list[tuple] | None = None


class RegularExpressionTokenizer(Tokenizer, BaseModel):
    pattern: str
    vocab_type: str = Field(default="slim")
    vocab: Vocabulary = Field(init=False, default_factory=Vocabulary)

    def fit(self, texts: list[str]) -> None:
        vocabulary = VocabularyFactory.create(vocab_type=self.vocab_type)

        for text in texts:
            matches = re.finditer(self.pattern, text)
            vocabulary.add_tokens([match.group() for match in matches])

        self.vocab = vocabulary

    def encode(self, text: str) -> list[int]:
        return [self.vocab[match.group()] for match in re.finditer(self.pattern, text)]

    def decode(self, token_ids: list[int] | int, join_token: str | None = None) -> str:
        if join_token is None:
            join_token = " "

        if isinstance(token_ids, list):
            return join_token.join(
                self.vocab.get_token_from_id(token_id) for token_id in token_ids
            )
        else:
            return self.vocab.get_token_from_id(token_ids)

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


if __name__ == "__main__":
    tokenizer = RegularExpressionTokenizer(pattern=r"\w+")
    tokenizer.fit(["Hello, world! Nice to meet you, Hello"])
    print(tokenizer.vocab)
    print(tokenizer.encode("Hello, world!"))
    print(tokenizer.decode([0, 1, 2]))

    text = """_The characteristics of Miss Austen's humour are so subtle and delicate
that they are, perhaps, at all times easier to apprehend than to
express, and at any particular time likely to be differently
apprehended by different persons. To me this humour seems to possess a
greater affinity, on the whole, to that of Addison than to any other of
the numerous species of this great British genus. It's best work it's not worse."""

    tokenizer = WPETokenizer(num_steps=100)
    tokenizer.fit(text)
    print()
    print(tokenizer.vocab)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))

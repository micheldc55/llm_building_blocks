import json
from abc import ABC, abstractmethod
from typing import Iterable, Iterator

from pydantic import BaseModel, Field


class Vocabulary(BaseModel):
    pass


class AbstractVocabulary(ABC, Vocabulary):
    @abstractmethod
    def add_token(self, token: str) -> None:
        pass

    @abstractmethod
    def add_tokens(self, tokens: Iterable[str]) -> None:
        pass

    @abstractmethod
    def remove_token(self, token: str) -> None:
        pass

    @abstractmethod
    def remove_tokens(self, tokens: Iterable[str]) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def from_json(self, json_str: str) -> None:
        pass

    @abstractmethod
    def to_json(self) -> str:
        pass

    @abstractmethod
    def get_token_id(self, token: str) -> int:
        pass

    @abstractmethod
    def get_token_from_id(self, id: int) -> str:
        pass

    @staticmethod
    @abstractmethod
    def _revert_mapping(mapping: dict[int, str]) -> dict[int, str]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, int]:
        pass


class SlimVocabulary(AbstractVocabulary):
    """Bare bones implementation of vocabulary with no bells or whistles. This base vocabulary
    doesn't contain much of the added complexity of other more specialized vocabularies, like
    <eos>, <pad>, <unk> tokens.

    If you want to add them you must add them externally through the `add_token` method.
    """

    vocab: dict[str, int] = Field(default_factory=dict, init=False)
    idx_to_vocab: dict[int, str] = Field(default_factory=dict, init=False)

    def add_token(self, token: str) -> None:
        if token in self.vocab:
            return None

        idx = len(self.vocab)
        self.vocab[token] = idx
        self.idx_to_vocab[idx] = token

    def add_tokens(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.add_token(token)

    def remove_token(self, token: str, reindex: bool = True) -> None:
        if token in self.vocab:
            token_id = self.vocab[token]
            del self.vocab[token]
            del self.idx_to_vocab[token_id]

            if reindex:
                self._reindex_vocab()

        else:
            raise ValueError(f"Token `{token}` not found in vocabulary")

    def remove_tokens(self, tokens: Iterable[str], reindex: bool = True) -> None:
        for token in tokens:
            self.remove_token(token, reindex=False)

        if reindex:
            self._reindex_vocab()

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            self.vocab = json.load(f)
            self.idx_to_vocab = self._revert_mapping(self.vocab)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    def from_json(self, json_str: str) -> None:
        self.vocab = json.loads(json_str)
        self.idx_to_vocab = self._revert_mapping(self.vocab)

    def to_json(self) -> str:
        return json.dumps(self.vocab)

    def get_token_id(self, token: str) -> int:
        return self.vocab[token]

    def get_token_from_id(self, id: int) -> str:
        return self.idx_to_vocab[id]

    @staticmethod
    def _revert_mapping(mapping: dict[int, str]) -> dict[int, str]:
        return {v: k for k, v in mapping.items()}

    def _reindex_vocab(self) -> None:
        self.vocab = {v: k for k, v in enumerate(self.vocab)}
        self.idx_to_vocab = self._revert_mapping(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def __getitem__(self, key: str) -> int:
        return self.vocab[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.vocab)

    def to_dict(self) -> dict[str, int]:
        return self.model_dump()


class VocabularyFactory:
    @staticmethod
    def create(vocab_type: str) -> Vocabulary:
        if vocab_type == "slim":
            return SlimVocabulary()
        else:
            raise ValueError(
                f"Invalid vocabulary type: `{vocab_type}` is not supported"
            )


if __name__ == "__main__":
    vocab = VocabularyFactory.create(vocab_type="slim")
    vocab.add_tokens(["hello", "world"])
    print(vocab.vocab)
    print(vocab.idx_to_vocab)

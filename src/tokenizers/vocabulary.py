from typing import Any, Iterator

from pydantic import BaseModel, Field


class Vocabulary(BaseModel):
    vocab: dict[str, int] = {}
    idx_to_vocab: dict[int, str] = {}
    unk_token: str = Field(default="<UNK>")
    pad_token: str = Field(default="<PAD>")
    bos_token: str = Field(default="<BOS>")
    eos_token: str = Field(default="<EOS>")

    def model_post_init(self, __context: Any) -> None:
        """Ensure special tokens exist in the vocabulary after Pydantic initialization."""
        vocab = self.vocab.copy() if self.vocab else {}

        vocab.setdefault(self.pad_token, len(vocab))
        vocab.setdefault(self.unk_token, len(vocab))
        vocab.setdefault(self.bos_token, len(vocab))
        vocab.setdefault(self.eos_token, len(vocab))

        object.__setattr__(self, "vocab", vocab)
        object.__setattr__(self, "idx_to_vocab", {v: k for k, v in vocab.items()})

    def __len__(self) -> int:
        return len(self.vocab)

    def __getitem__(self, key: str) -> int:
        return self.vocab.get(key, self.vocab[self.unk_token])

    def to_dict(self) -> dict[str, int]:
        """Return the internal vocabulary dictionary."""
        return self.vocab

    def __str__(self) -> str:
        return str(self.vocab)

    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self) -> Iterator[str]:
        return iter(self.vocab)

    def add(self, token: list[str] | str) -> None:
        if isinstance(token, list):
            for t in token:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
                    self.idx_to_vocab[len(self.vocab) - 1] = t
        else:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.idx_to_vocab[len(self.vocab) - 1] = token

    def _revert_mapping(self, id: int) -> str:
        return self.idx_to_vocab[id]


if __name__ == "__main__":
    vocab = Vocabulary()
    print(vocab)
    print(vocab.to_dict())
    vocab.add("hello")
    print(vocab)
    print(vocab.to_dict())
    vocab.add("world")
    print(vocab.idx_to_vocab[5])
    print(vocab._revert_mapping(5))

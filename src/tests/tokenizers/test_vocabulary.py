import pytest

from src.tokenizers.vocabulary import Vocabulary


@pytest.fixture
def vocabulary():
    return Vocabulary()


def test_vocabulary_init(vocabulary):
    assert vocabulary.vocab == {"<BOS>": 2, "<EOS>": 3, "<PAD>": 0, "<UNK>": 1}
    assert vocabulary.idx_to_vocab == {2: "<BOS>", 3: "<EOS>", 0: "<PAD>", 1: "<UNK>"}
    assert vocabulary.unk_token == "<UNK>"
    assert vocabulary.pad_token == "<PAD>"
    assert vocabulary.bos_token == "<BOS>"
    assert vocabulary.eos_token == "<EOS>"


def test_vocabulary_add(vocabulary):
    vocabulary.add("hello")
    assert vocabulary.vocab == {
        "<BOS>": 2,
        "<EOS>": 3,
        "<PAD>": 0,
        "<UNK>": 1,
        "hello": 4,
    }
    assert vocabulary.idx_to_vocab == {
        4: "hello",
        2: "<BOS>",
        3: "<EOS>",
        0: "<PAD>",
        1: "<UNK>",
    }

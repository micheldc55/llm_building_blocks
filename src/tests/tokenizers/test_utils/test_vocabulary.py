import pytest

from src.tokenizers.utils.vocabulary import SlimVocabulary


@pytest.fixture
def slim_vocabulary():
    return SlimVocabulary()


def test_slim_vocabulary_add(slim_vocabulary):
    slim_vocabulary.add_token("hello")
    assert slim_vocabulary.vocab == {
        "hello": 0,
    }


def test_slim_vocabulary_add_tokens(slim_vocabulary):
    slim_vocabulary.add_tokens(["hello", "world", "<eos>", "<pad>", "<unk>"])
    assert slim_vocabulary.vocab == {
        "hello": 0,
        "world": 1,
        "<eos>": 2,
        "<pad>": 3,
        "<unk>": 4,
    }


def test_slim_vocabulary_remove_token(slim_vocabulary):
    slim_vocabulary.add_tokens(["hello", "world", "<eos>", "<pad>", "<unk>"])
    slim_vocabulary.remove_token("hello", reindex=True)
    assert slim_vocabulary.vocab == {
        "world": 0,
        "<eos>": 1,
        "<pad>": 2,
        "<unk>": 3,
    }
    assert slim_vocabulary.idx_to_vocab == {
        0: "world",
        1: "<eos>",
        2: "<pad>",
        3: "<unk>",
    }


def test_slim_vocabulary_remove_tokens(slim_vocabulary):
    slim_vocabulary.add_tokens(["hello", "world", "<eos>", "<pad>", "<unk>"])
    slim_vocabulary.remove_tokens(["hello", "world"], reindex=True)
    assert slim_vocabulary.vocab == {
        "<eos>": 0,
        "<pad>": 1,
        "<unk>": 2,
    }


def test_slim_vocabulary_remove_token(slim_vocabulary):
    slim_vocabulary.add_tokens(["hello", "world", "<eos>", "<pad>", "<unk>"])
    slim_vocabulary.remove_token("hello", reindex=True)
    assert slim_vocabulary.vocab == {
        "world": 0,
        "<eos>": 1,
        "<pad>": 2,
        "<unk>": 3,
    }
    assert slim_vocabulary.idx_to_vocab == {
        0: "world",
        1: "<eos>",
        2: "<pad>",
        3: "<unk>",
    }


def test_slim_vocabulary_remove_tokens(slim_vocabulary):
    slim_vocabulary.add_tokens(["hello", "world", "<eos>", "<pad>", "<unk>"])
    slim_vocabulary.remove_tokens(["hello", "world"], reindex=True)
    assert slim_vocabulary.vocab == {
        "<eos>": 0,
        "<pad>": 1,
        "<unk>": 2,
    }

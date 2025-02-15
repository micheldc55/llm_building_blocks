import pytest

from src.tokenizers.base_tokenizers import RegularExpressionTokenizer


@pytest.fixture
def tokenizer():
    tokenizer = RegularExpressionTokenizer(
        pattern=r'\s+|[,.?!"\'\-_()]+|[^\s,.?!"\'\-_()]+'
    )
    tokenizer.fit(["Hello, world!", "Hello, world!", "Oh, my god!"])
    return tokenizer


def test_tokenizer_init(tokenizer):
    assert tokenizer.pattern == r'\s+|[,.?!"\'\-_()]+|[^\s,.?!"\'\-_()]+'
    assert tokenizer.vocab is not None


def test_tokenizer_encode(tokenizer):
    assert tokenizer.encode("Hello, world!") == [4, 5, 6, 7, 8]
    assert tokenizer.encode("NewWordUnseen") == [1]


def test_tokenizer_decode(tokenizer):
    assert tokenizer.decode([4, 5, 6, 7, 8]) == "Hello, world!"
    assert tokenizer.decode([1]) == "<UNK>"


def test_tokenizer_fit(tokenizer):
    tokenizer.fit(["Hello, world!", "Hello, world!", "Oh, my!"])
    assert tokenizer.vocab is not None
    assert tokenizer.vocab.vocab == {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "Hello": 4,
        ",": 5,
        " ": 6,
        "world": 7,
        "!": 8,
        "Oh": 9,
        "my": 10,
    }

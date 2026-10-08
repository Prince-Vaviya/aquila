from src.tokenizer import tokenize


def test_tokenize():
    text = "The Sun is HOT!"
    tokens = tokenize(text)

    assert tokens == [
        "the",
        "sun",
        "is",
        "hot"
    ]

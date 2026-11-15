from graphrag.llm.token_counter import TokenCounter


def test_count_uses_the_models_tokenizer():
    assert TokenCounter("gpt-4o").count("hello world") == 2
    assert TokenCounter("gpt-4").count("hello world") == 2
    assert TokenCounter().count("") == 0


def test_count_treats_special_tokens_as_text():
    assert TokenCounter().count("<|endoftext|>") > 1

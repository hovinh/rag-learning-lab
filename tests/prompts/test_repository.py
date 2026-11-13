from graphrag.prompts.repository import PromptRepository


def test_load_reads_file_and_strips_surrounding_whitespace(tmp_path):
    (tmp_path / "greeting.txt").write_text("\n  Hello, world!  \n", encoding="utf-8")

    result = PromptRepository(tmp_path).load("greeting.txt")

    assert result == "Hello, world!"


def test_load_defaults_to_own_package_directory():
    result = PromptRepository().load("answer_system.txt")

    assert result

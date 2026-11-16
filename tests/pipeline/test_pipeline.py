from dataclasses import dataclass

from graphrag.pipeline.pipeline import Pipeline, step_title


@dataclass
class FakeContext:
    log: list[str]
    total: int = 0


def append_a(context: FakeContext) -> None:
    context.log.append("a")


def append_b(context: FakeContext) -> None:
    context.log.append("b")


def add_one(context: FakeContext) -> None:
    context.total += 1


def test_run_executes_steps_in_order():
    context = FakeContext(log=[])

    Pipeline([append_a, append_b]).run(context)

    assert context.log == ["a", "b"]


def test_run_lets_steps_mutate_shared_context():
    context = FakeContext(log=[])

    Pipeline([add_one, add_one, add_one]).run(context)

    assert context.total == 3


def test_run_returns_the_context():
    context = FakeContext(log=[])

    result = Pipeline([append_a]).run(context)

    assert result is context


class ExtractGraphStep:
    def __call__(self, context: FakeContext) -> None: ...


class TitledStep:
    title = "My own title"

    def __call__(self, context: FakeContext) -> None: ...


def test_step_title_comes_from_title_class_name_or_function_name():
    assert step_title(ExtractGraphStep()) == "Extract graph"
    assert step_title(TitledStep()) == "My own title"
    assert step_title(append_a) == "Append a"


def test_run_with_no_steps_is_a_noop():
    context = FakeContext(log=[])

    Pipeline([]).run(context)

    assert context.log == []

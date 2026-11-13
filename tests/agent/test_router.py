import pytest

from graphrag.agent.errors import ToolRoutingError
from graphrag.agent.router import ToolRouter
from graphrag.agent.tools import AnswerGivenTool, Tool, string_parameters
from graphrag.llm.client import ToolCall


class EchoTool(Tool):
    name = "echo"
    description = "Echo the text"
    parameters = string_parameters("text", "Text to echo")

    def run(self, arguments):
        return [{"echo": arguments["text"]}]


def test_route_offers_every_tool_and_runs_the_chosen_ones(llm):
    llm.tool_calls = [[ToolCall("echo", {"text": "hi"}), ToolCall("respond", {"answer": "ok"})]]
    router = ToolRouter(llm, [EchoTool(), AnswerGivenTool()], model="test-model")

    results = router.route("Say hi", [{"role": "assistant", "content": "earlier answer"}])

    assert results == [[{"echo": "hi"}], "ok"]
    [call] = llm.choose_tools_calls
    assert call["model"] == "test-model"
    assert [t["function"]["name"] for t in call["tools"]] == ["echo", "respond"]
    messages = call["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "assistant", "content": "earlier answer"}
    assert messages[2] == {
        "role": "user",
        "content": "The user question to find a tool to answer: 'Say hi'",
    }


def test_route_returns_nothing_when_no_tool_is_chosen(llm):
    llm.tool_calls = [[]]

    assert ToolRouter(llm, [EchoTool()]).route("?") == []


def test_route_rejects_unknown_tool(llm):
    llm.tool_calls = [[ToolCall("answer_given", {"answer": "x"})]]

    with pytest.raises(ToolRoutingError, match="answer_given"):
        ToolRouter(llm, [AnswerGivenTool()]).route("?")


def test_route_rejects_missing_required_argument(llm):
    llm.tool_calls = [[ToolCall("echo", {})]]

    with pytest.raises(ToolRoutingError, match="text"):
        ToolRouter(llm, [EchoTool()]).route("?")

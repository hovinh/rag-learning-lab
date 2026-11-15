from graphrag.agent.agent import Agent
from graphrag.agent.critic import AnswerCritic
from graphrag.agent.query_decomposer import QueryDecomposer
from graphrag.agent.query_updater import QueryUpdater
from graphrag.agent.router import ToolRouter
from graphrag.agent.tools import AnswerGivenTool, Tool, string_parameters
from graphrag.llm.client import ToolCall


class MovieLookupTool(Tool):
    name = "movie_info_by_title"
    description = "Get information about a movie by providing the title"
    parameters = string_parameters("title", "The movie title")

    def run(self, arguments):
        return [{"title": arguments["title"], "cast": ["Keanu Reeves"]}]


def make_agent(llm, query_planner=None):
    router = ToolRouter(llm, [MovieLookupTool(), AnswerGivenTool()])
    query_planner = query_planner or QueryUpdater(llm)
    return Agent(llm, query_planner, router, AnswerCritic(llm), model="test-model")


def test_run_answers_directly_when_critic_is_satisfied(llm):
    llm.completions = [
        '{"question": "Who acted in The Matrix?"}',  # query update
        '{"questions": []}',  # critique
        "Keanu Reeves.",  # final answer
    ]
    llm.tool_calls = [[ToolCall("movie_info_by_title", {"title": "The Matrix"})]]

    answer = make_agent(llm).run("Who's in Matrix?")

    assert answer == "Keanu Reeves."
    final = llm.complete_calls[-1]
    assert final["model"] == "test-model"
    assert final["json_mode"] is False
    assert final["messages"][0]["role"] == "system"
    assert final["messages"][1] == {
        "role": "assistant",
        "content": (
            "For the question: 'Who acted in The Matrix?', we have the answer: "
            '\'[[{"title": "The Matrix", "cast": ["Keanu Reeves"]}]]\''
        ),
    }
    assert final["messages"][2] == {
        "role": "user",
        "content": "The user question to answer: Who's in Matrix?",
    }


def test_run_gathers_critic_follow_ups_once(llm):
    llm.completions = [
        '{"question": "Who is the main actor in The Matrix?"}',
        '{"questions": ["What movies is Keanu Reeves in?", "When was he born?"]}',
        '{"question": "What movies is Keanu Reeves in, and when was he born?"}',
        "Keanu Reeves, in The Matrix.",
    ]
    llm.tool_calls = [
        [ToolCall("movie_info_by_title", {"title": "The Matrix"})],
        [ToolCall("respond", {"answer": "The Matrix; born 1964"})],
    ]

    answer = make_agent(llm).run("Who's the main actor in Matrix and what else are they in?")

    assert answer == "Keanu Reeves, in The Matrix."
    # The follow-ups are joined into one question for the second update.
    second_update = llm.complete_calls[2]["messages"]
    assert second_update[-1]["content"] == (
        "The user question to rewrite: 'What movies is Keanu Reeves in? When was he born?'"
    )
    # The second round sees the first round's answer.
    assert len(second_update) == 3
    assert len(llm.choose_tools_calls[1]["messages"]) == 3
    final_history = llm.complete_calls[-1]["messages"][1:-1]
    assert len(final_history) == 2
    assert "born 1964" in final_history[1]["content"]


def test_respond_returns_the_gathered_answers_as_contexts(llm):
    llm.completions = ['{"question": "Who acted in The Matrix?"}', '{"questions": []}', "Keanu."]
    llm.tool_calls = [[ToolCall("movie_info_by_title", {"title": "The Matrix"})]]

    response = make_agent(llm).respond("Who's in Matrix?")

    assert response.answer == "Keanu."
    assert response.contexts == [
        (
            "For the question: 'Who acted in The Matrix?', we have the answer: "
            '\'[[{"title": "The Matrix", "cast": ["Keanu Reeves"]}]]\''
        )
    ]


def test_final_answer_prompt_points_at_the_tool_answers(llm):
    llm.completions = ['{"question": "q"}', '{"questions": []}', "answer"]
    llm.tool_calls = [[ToolCall("respond", {"answer": "a"})]]

    make_agent(llm).run("q")

    system = llm.complete_calls[-1]["messages"][0]["content"]
    # Quotes agent_tool_answer.txt's wording, which the history uses.
    assert "For the question: '...', we have the answer: '...'" in system
    assert "the database's complete result" in system
    assert "Don't call information missing because you expected more of it." in system
    assert "Only say information is missing when none of these results contains it" in system


def test_run_starts_each_question_with_empty_history(llm):
    agent = make_agent(llm)
    for _ in range(2):
        llm.completions += ['{"question": "q"}', '{"questions": []}', "answer"]
        llm.tool_calls.append([ToolCall("respond", {"answer": "a"})])

    agent.run("first")
    agent.run("second")

    second_run_final = llm.complete_calls[-1]["messages"]
    assert len(second_run_final) == 3


def test_decomposed_questions_each_get_a_tool_round(llm):
    llm.completions = [
        '{"questions": ["Who acted in The Matrix?", "When was Keanu Reeves born?"]}',
        '{"questions": []}',
        "Keanu Reeves, born 1964.",
    ]
    llm.tool_calls = [
        [ToolCall("movie_info_by_title", {"title": "The Matrix"})],
        [ToolCall("respond", {"answer": "1964"})],
    ]

    response = make_agent(llm, QueryDecomposer(llm)).respond(
        "Who acted in The Matrix and when was Keanu Reeves born?"
    )

    assert response.answer == "Keanu Reeves, born 1964."
    first_route, second_route = llm.choose_tools_calls
    assert first_route["messages"][-1]["content"] == (
        "The user question to find a tool to answer: 'Who acted in The Matrix?'"
    )
    # The second question is routed with the first one's answer.
    assert second_route["messages"][1]["content"] == response.contexts[0]
    assert second_route["messages"][-1]["content"] == (
        "The user question to find a tool to answer: 'When was Keanu Reeves born?'"
    )
    assert response.contexts == [
        (
            "For the question: 'Who acted in The Matrix?', we have the answer: "
            '\'[[{"title": "The Matrix", "cast": ["Keanu Reeves"]}]]\''
        ),
        "For the question: 'When was Keanu Reeves born?', we have the answer: '[\"1964\"]'",
    ]

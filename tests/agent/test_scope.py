from graphrag.agent.critic import AnswerCritic
from graphrag.agent.query_decomposer import QueryDecomposer
from graphrag.agent.query_updater import QueryUpdater
from graphrag.agent.scope import AgentScope
from graphrag.agent.tools import AnswerGivenTool

SCOPE = AgentScope(source="a library catalogue", topic="books", capabilities="book questions")


def test_fill_replaces_placeholders_but_keeps_json_braces():
    template = 'Answer from {source} about {topic}: {"questions": []}'

    assert SCOPE.fill(template) == 'Answer from a library catalogue about books: {"questions": []}'


def system_prompt(llm, planner_or_critic, question="Q?"):
    llm.completions = ['{"question": "Q?", "questions": ["Q?"]}']
    if isinstance(planner_or_critic, AnswerCritic):
        planner_or_critic.critique(question, [])
    else:
        planner_or_critic.plan(question)
    return llm.complete_calls[-1]["messages"][0]["content"]


def test_planners_and_critic_describe_the_scope(llm):
    for component in (
        QueryUpdater(llm, scope=SCOPE),
        QueryDecomposer(llm, scope=SCOPE),
    ):
        prompt = system_prompt(llm, component)
        assert "answered from a library catalogue, not from the internet" in prompt
        assert "{source}" not in prompt

    critic_prompt = system_prompt(llm, AnswerCritic(llm, scope=SCOPE))
    assert "isn't related to books and or is smalltalk" in critic_prompt


def test_respond_tool_names_the_scope_capabilities():
    assert AnswerGivenTool(SCOPE).description.endswith("you can only answer book questions.")

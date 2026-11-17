from graphrag.agent.critic import AnswerCritic
from graphrag.agent.query_updater import QueryUpdater
from graphrag.agent.tools import AnswerGivenTool
from graphrag.domains.movies import MOVIES_AGENT_SCOPE, MovieGraphSeeder, build_movie_agent
from graphrag.graph.kuzu_cypher_store import KuzuCypherGraphStore


def test_movies_scope_reproduces_the_books_prompt_wording(llm):
    """The book's prompts hard-code movies; with MOVIES_AGENT_SCOPE the
    filled prompts must read exactly as they did.
    """
    llm.completions = ['{"question": "Q?"}', '{"questions": []}']
    QueryUpdater(llm, scope=MOVIES_AGENT_SCOPE).plan("Q?")
    AnswerCritic(llm, scope=MOVIES_AGENT_SCOPE).critique("Q?", [])
    updater_prompt, critic_prompt = (call["messages"][0]["content"] for call in llm.complete_calls)

    assert (
        "The questions are answered from a database of movies and the people who worked on "
        "them, not from the internet." in updater_prompt
    )
    assert (
        "If the question isn't related to movies and or is smalltalk, there is no need for "
        "follow up questions." in critic_prompt
    )
    assert AnswerGivenTool(MOVIES_AGENT_SCOPE).description.endswith(
        "remind them that you can only answer questions about movies and their cast."
    )


def test_build_movie_agent_routes_to_the_movie_tools(llm, tmp_path):
    with KuzuCypherGraphStore(tmp_path / "movies_db") as store:
        MovieGraphSeeder(store).seed()

        agent = build_movie_agent(llm, store, model="test-model")

        tool_names = [tool.name for tool in agent._router._tools.values()]
        assert tool_names == [
            "movie_info_by_title",
            "movies_info_by_actor",
            "text2cypher",
            "respond",
        ]

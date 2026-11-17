from pathlib import Path

from graphrag.services import OpenAIServices
from graphrag.settings import Settings


def test_db_path_for_names_a_sibling_database():
    settings = Settings(openai_api_key="key", kuzu_db_path=Path("data/kuzu_db"))

    assert settings.db_path_for("movies") == Path("data/kuzu_db_movies")


def test_openai_services_share_one_usage_meter():
    from graphrag.observability import UsageMeter

    usage = UsageMeter()
    services = OpenAIServices.create(
        Settings(openai_api_key="key", kuzu_db_path=Path("db")), usage=usage
    )

    assert services.llm._usage is usage
    assert services.embedder._usage is usage

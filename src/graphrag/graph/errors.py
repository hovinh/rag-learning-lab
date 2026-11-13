class GraphQueryError(Exception):
    """A Cypher query was rejected by the graph database (e.g. invalid
    LLM-generated Cypher from text2cypher).
    """

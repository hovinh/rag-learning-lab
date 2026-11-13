class LLMResponseError(Exception):
    """The LLM's response couldn't be used, e.g. JSON that doesn't parse
    or lacks an expected field. The book's notebooks print and carry on
    in this case; raising instead keeps a bad response from silently
    turning into an empty answer.
    """

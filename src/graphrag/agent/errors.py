class ToolRoutingError(Exception):
    """The LLM chose a tool that doesn't exist, or left out a required
    argument.
    """

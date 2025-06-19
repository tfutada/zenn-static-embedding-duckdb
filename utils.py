import textwrap


def fold_text(body: str, width: int = 10, max_chars: int = 50) -> str:
    """
    Truncates the input text to a maximum number of characters and wraps it into lines of a specified width.
    Inserts HTML line breaks for better display in web-based visualizations.

    Args:
        body (str): The input text to be processed.
        width (int, optional): The maximum number of characters per line. Defaults to 10.
        max_chars (int, optional): The maximum number of characters to consider from the input text. Defaults to 50.

    Returns:
        str: The processed text with HTML line breaks.
    """
    truncated = body[:max_chars]
    wrapped_lines = textwrap.wrap(truncated, width=width)
    return "<br>".join(wrapped_lines)

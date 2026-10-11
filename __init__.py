"""Stream text from the signed-in Grok Build CLI.

The CLI session created by ``grok login`` is the credential. This package
does not read or send an API key.
"""

from grokbuildstreaming.stream import (
    DEFAULT_MODEL,
    GrokBuildError,
    collect_text,
    find_grok_bin,
    format_prompt,
    stream_text,
    subscription_env,
)

__version__ = "0.1.3"

__all__ = [
    "DEFAULT_MODEL",
    "GrokBuildError",
    "collect_text",
    "find_grok_bin",
    "format_prompt",
    "stream_text",
    "subscription_env",
]

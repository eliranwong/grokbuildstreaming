# grokbuildstreaming

Stream text from the [Grok Build](https://x.ai) CLI. This library is a text-completion alternative to [agentmake](https://github.com/eliranwong/agentmake): you pass chat messages, and it yields response text as the CLI produces it.

Authentication is the subscription session from `grok login` (`~/.grok/auth.json`). The child process does not receive `XAI_API_KEY`. No API key is read or sent.

## Install

```bash
pip install -e /path/to/grokbuildstreaming
```

Requires the `grok` executable on `PATH` or at `~/.grok/bin/grok`, and a completed `grok login`.

## Stream

```python
import asyncio
from grokbuildstreaming import stream_text

async def main():
    messages = [
        {"role": "system", "content": "You are a concise Bible study assistant."},
        {"role": "user", "content": "What is John 3:16 about?"},
        {"role": "assistant", "content": "It states God's love for the world."},
    ]
    async for chunk in stream_text(messages, follow_up="Quote it briefly.", model="grok-4.7"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

`messages` use `role` and `content`, the same shape agentmake accepts. `follow_up` is the newest user turn. System text is sent with `--system-prompt-override`. Each call is a new headless process: the library does not resume a previous Grok session.

The process is text-only. It runs with `--max-turns 1`, plan mode off, web search off, subagents off, and built-in tools removed, in a temporary working directory. Cancel an in-flight stream by setting the `asyncio.Event` passed as `cancel_event`.

`collect_text` returns the whole reply:

```python
from grokbuildstreaming import collect_text

text = await collect_text(follow_up="Reply with exactly: pong")
```

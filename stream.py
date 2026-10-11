"""Headless Grok Build text streaming.

Each call starts one ``grok`` process and yields ``text`` events from
``--output-format streaming-json``. The process uses the signed-in
subscription session. ``XAI_API_KEY`` is removed from the child environment
so an exported API key is not the credential.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_MODEL = "grok-4.7"
STREAM_LIMIT = 8 * 1024 * 1024
# A per-model api_key in ~/.grok/config.toml still outranks the login session.
# These variables are the environment fallback and are removed on purpose.
_API_KEY_ENV = ("XAI_API_KEY", "XAI_API_KEY_FILE")
_TEXT_ONLY_RULE = (
    "Reply with the answer text only. "
    "Do not call tools, read or edit files, browse the web, or run commands."
)
# Internal tool ids. Both shell names appear across CLI versions.
_DISALLOWED_TOOLS = ",".join((
    "run_terminal_command",
    "run_terminal_cmd",
    "read_file",
    "search_replace",
    "write",
    "list_dir",
    "grep",
    "web_search",
    "web_fetch",
    "todo_write",
    "spawn_subagent",
    "kill_command_or_subagent",
    "get_command_or_subagent_output",
    "scheduler_create",
    "scheduler_delete",
    "scheduler_list",
    "monitor",
    "search_tool",
    "use_tool",
    "workflow",
    "enter_plan_mode",
    "exit_plan_mode",
    "ask_user_question",
    "image_gen",
    "image_edit",
    "image_to_video",
    "reference_to_video",
    "send_feedback",
    "Agent",
))


class GrokBuildError(RuntimeError):
    """The Grok Build CLI is missing, or a turn failed before any text."""


def find_grok_bin() -> str:
    """Return the ``grok`` executable from ``PATH`` or ``~/.grok/bin/grok``."""
    found = shutil.which("grok")
    if found and os.path.isfile(found) and os.access(found, os.X_OK):
        return found
    fallback = os.path.expanduser("~/.grok/bin/grok")
    if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
        return fallback
    raise GrokBuildError(
        "The grok CLI was not found. Install Grok Build, then sign in with: grok login"
    )


def subscription_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a child process that keeps the ``grok login`` session."""
    env = dict(os.environ if base is None else base)
    for name in _API_KEY_ENV:
        env.pop(name, None)
    env["GROK_DISABLE_AUTOUPDATER"] = "1"
    return env


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if not isinstance(text, str):
                    text = item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def format_prompt(
    messages: Sequence[Mapping[str, Any]] | None = None,
    follow_up: str | None = None,
    system: str | None = None,
) -> tuple[str, str]:
    """Split chat messages into a user prompt and a system prompt.

    A lone user turn is left unlabeled so the model sees that text exactly.
    Earlier turns are labeled when history is present. System messages are
    not copied into the user prompt.
    """
    system_parts: list[str] = []
    if isinstance(system, str) and system.strip():
        system_parts.append(system.strip())
    turns: list[tuple[str, str]] = []
    for message in messages or ():
        role = str(message.get("role") or "user").strip().lower()
        content = _content_to_text(message.get("content"))
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            turns.append(("Assistant", content))
        else:
            turns.append(("User", content))
    if isinstance(follow_up, str) and follow_up.strip():
        turns.append(("User", follow_up.strip()))
    system_parts.append(_TEXT_ONLY_RULE)
    if not turns:
        raise GrokBuildError("No prompt to send.")
    if len(turns) == 1 and turns[0][0] == "User":
        prompt = turns[0][1]
    else:
        prompt = "\n\n".join(f"{role}:\n{content}" for role, content in turns)
    return prompt, "\n\n".join(system_parts)


def build_command(
    grok_bin: str,
    prompt_path: str,
    work_dir: str,
    system_prompt: str,
    model: str,
    max_turns: int = 1,
) -> list[str]:
    """Argv for one text-only headless turn. No API key and no tool approval."""
    return [
        grok_bin,
        "--prompt-file", prompt_path,
        "--output-format", "streaming-json",
        "--cwd", work_dir,
        "-m", model,
        "--max-turns", str(max_turns),
        "--no-plan",
        "--no-subagents",
        "--no-auto-update",
        "--verbatim",
        "--disable-web-search",
        "--permission-mode", "dontAsk",
        "--disallowed-tools", _DISALLOWED_TOOLS,
        "--system-prompt-override", system_prompt,
    ]


def parse_stream_line(line: str) -> dict[str, Any] | None:
    """Parse one stdout line from ``--output-format streaming-json``."""
    text = line.strip()
    if not text:
        return None
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(event, dict):
        return event
    return None


def text_from_event(event: Mapping[str, Any]) -> str | None:
    """Return a response text chunk. Other event types contribute no answer text."""
    if event.get("type") != "text":
        return None
    data = event.get("data")
    if isinstance(data, str) and data:
        return data
    return None


def _completion_workspace() -> str:
    """Stable directory outside the caller's project, with no git root of its own."""
    path = os.path.join(tempfile.gettempdir(), "grokbuildstreaming-workspace")
    os.makedirs(path, exist_ok=True)
    return path


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        return


async def _next_line(stream: asyncio.StreamReader, cancel_event: asyncio.Event | None) -> bytes | None:
    """Read one stdout line. ``None`` means the caller cancelled."""
    read_task = asyncio.create_task(stream.readline())
    if cancel_event is None:
        return await read_task
    if cancel_event.is_set():
        await _cancel_task(read_task)
        return None
    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {read_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        await _cancel_task(read_task)
        await _cancel_task(cancel_task)
        raise
    if read_task in done and not read_task.cancelled():
        await _cancel_task(cancel_task)
        return read_task.result()
    await _cancel_task(read_task)
    await _cancel_task(cancel_task)
    return None


async def _read_stderr(stream: asyncio.StreamReader | None) -> str:
    if stream is None:
        return ""
    chunks: list[str] = []
    size = 0
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            chunks.append(text)
            size += len(text)
            if size > 16000:
                chunks = chunks[-20:]
                size = sum(len(item) for item in chunks)
    except asyncio.CancelledError:
        return "".join(chunks)[-8000:]
    return "".join(chunks)[-8000:]


async def _stop_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except asyncio.TimeoutError:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            except Exception:
                process.kill()
        try:
            await process.wait()
        except Exception:
            return


async def stream_text(
    messages: Sequence[Mapping[str, Any]] | None = None,
    follow_up: str | None = None,
    system: str | None = None,
    model: str | None = None,
    grok_bin: str | None = None,
    cancel_event: asyncio.Event | None = None,
    max_turns: int = 1,
):
    """Yield text chunks from one signed-in Grok Build turn.

    Closing the generator, or setting ``cancel_event``, stops the CLI process.
    ``model`` defaults to :data:`DEFAULT_MODEL`. The working directory is a
    temporary folder, and this call does not resume an earlier session.
    """
    prompt, system_prompt = format_prompt(messages, follow_up=follow_up, system=system)
    executable = grok_bin or find_grok_bin()
    chosen_model = model.strip() if isinstance(model, str) and model.strip() else DEFAULT_MODEL
    work_dir = _completion_workspace()
    prompt_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="grokbuildstreaming-",
        suffix=".txt",
        delete=False,
    )
    prompt_path = prompt_handle.name
    process: asyncio.subprocess.Process | None = None
    stderr_task: asyncio.Task | None = None
    produced_text = False
    try:
        prompt_handle.write(prompt)
        prompt_handle.close()
        command = build_command(
            executable,
            prompt_path,
            work_dir,
            system_prompt,
            chosen_model,
            max_turns=max_turns,
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=subscription_env(),
            start_new_session=True,
            limit=STREAM_LIMIT,
        )
        if process.stdout is None:
            raise GrokBuildError("grok did not provide an output stream")
        stderr_task = asyncio.create_task(_read_stderr(process.stderr))
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return
            line = await _next_line(process.stdout, cancel_event)
            if line is None:
                return
            if line == b"":
                break
            event = parse_stream_line(line.decode("utf-8", errors="replace"))
            if not event:
                continue
            if event.get("type") == "error":
                if cancel_event is not None and cancel_event.is_set():
                    return
                message = event.get("message") or "Grok Build reported an error."
                raise GrokBuildError(str(message))
            chunk = text_from_event(event)
            if chunk:
                produced_text = True
                yield chunk
        returncode = await process.wait()
        stderr_text = ""
        if stderr_task is not None:
            stderr_text = (await stderr_task).strip()
            stderr_task = None
        if returncode not in (0, None) and not produced_text:
            detail = stderr_text or f"grok exited with code {returncode}"
            raise GrokBuildError(detail)
    finally:
        await _stop_process(process)
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            os.remove(prompt_path)
        except OSError:
            pass


async def collect_text(
    messages: Sequence[Mapping[str, Any]] | None = None,
    follow_up: str | None = None,
    system: str | None = None,
    model: str | None = None,
    grok_bin: str | None = None,
    cancel_event: asyncio.Event | None = None,
    max_turns: int = 1,
) -> str:
    """Run :func:`stream_text` and return the joined text."""
    chunks: list[str] = []
    async for chunk in stream_text(
        messages,
        follow_up=follow_up,
        system=system,
        model=model,
        grok_bin=grok_bin,
        cancel_event=cancel_event,
        max_turns=max_turns,
    ):
        chunks.append(chunk)
    return "".join(chunks)

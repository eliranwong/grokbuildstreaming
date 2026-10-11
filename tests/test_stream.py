import unittest
from unittest import mock

from grokbuildstreaming.stream import (
    DEFAULT_MODEL,
    GrokBuildError,
    build_command,
    find_grok_bin,
    format_prompt,
    parse_stream_line,
    subscription_env,
    text_from_event,
)


class FormatPromptTests(unittest.TestCase):
    def test_single_user_turn_is_unlabeled(self):
        prompt, system = format_prompt(
            [{"role": "system", "content": "You are BibleMate AI."}],
            follow_up="List two tools.",
        )
        self.assertEqual(prompt, "List two tools.")
        self.assertIn("You are BibleMate AI.", system)
        self.assertNotIn("List two tools.", system)
        self.assertNotIn("User:", prompt)

    def test_history_keeps_the_latest_request_intact(self):
        prompt, system = format_prompt(
            [
                {"role": "system", "content": "Role."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
            ],
            follow_up="Second question",
            system="Extra role",
        )
        self.assertIn("User:\nFirst question", prompt)
        self.assertIn("Assistant:\nFirst answer", prompt)
        self.assertTrue(prompt.endswith("User:\nSecond question"))
        self.assertIn("Extra role", system)
        self.assertIn("Role.", system)
        self.assertNotIn("Second question", system)

    def test_content_blocks_join_text(self):
        prompt, _system = format_prompt(
            [{"role": "user", "content": [{"type": "text", "text": "Hello"}, {"text": "there"}]}],
        )
        self.assertEqual(prompt, "Hello\nthere")

    def test_empty_prompt_raises(self):
        with self.assertRaises(GrokBuildError):
            format_prompt([])


class EventTests(unittest.TestCase):
    def test_text_chunk(self):
        event = parse_stream_line('{"type":"text","data":"pong"}')
        self.assertEqual(text_from_event(event), "pong")

    def test_tool_event_adds_no_answer_text(self):
        event = parse_stream_line('{"type":"tool_call","toolName":"read_file","rawInput":{}}')
        self.assertIsNone(text_from_event(event))

    def test_non_json_is_ignored(self):
        self.assertIsNone(parse_stream_line("plain log line"))


class CommandTests(unittest.TestCase):
    def test_command_is_text_only_and_has_no_api_key(self):
        command = build_command(
            "/usr/bin/grok",
            "/tmp/prompt.txt",
            "/tmp/work",
            "Be brief.",
            DEFAULT_MODEL,
        )
        joined = " ".join(command)
        self.assertNotIn("XAI_API_KEY", joined)
        self.assertNotIn("--always-approve", command)
        self.assertIn("--verbatim", command)
        self.assertIn("--permission-mode", command)
        self.assertIn("dontAsk", command)
        self.assertIn("--output-format", command)
        self.assertIn("streaming-json", command)
        self.assertIn(DEFAULT_MODEL, command)
        self.assertIn("--max-turns", command)
        self.assertIn("1", command)

    def test_subscription_env_drops_api_key(self):
        env = subscription_env({"XAI_API_KEY": "xai-secret", "PATH": "/usr/bin", "HOME": "/Users/me"})
        self.assertNotIn("XAI_API_KEY", env)
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertEqual(env["GROK_DISABLE_AUTOUPDATER"], "1")

    def test_find_grok_bin_uses_path(self):
        with mock.patch("grokbuildstreaming.stream.shutil.which", return_value="/opt/grok"):
            with mock.patch("grokbuildstreaming.stream.os.path.isfile", return_value=True):
                with mock.patch("grokbuildstreaming.stream.os.access", return_value=True):
                    self.assertEqual(find_grok_bin(), "/opt/grok")

    def test_find_grok_bin_errors_when_missing(self):
        with mock.patch("grokbuildstreaming.stream.shutil.which", return_value=None):
            with mock.patch("grokbuildstreaming.stream.os.path.isfile", return_value=False):
                with self.assertRaises(GrokBuildError):
                    find_grok_bin()


if __name__ == "__main__":
    unittest.main()

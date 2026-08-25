#!/usr/bin/env python3
"""Unit tests for the Path ACL (session-scoped allow/ask/deny).

Covers:
- PathAcl defaults: CWD allowed, $HOME reads allowed (explicit only), home
  dotfiles / /proc /sys /etc are `ask`, everything else outside is denied.
- Session overrides via add()/remove(), longest-prefix precedence.
- _path_acl_decision: ask -> non-TTY deny, hard-block of sensitive /proc files,
  and the "explicit home request only" rule (../ escapes stay denied).
- Shell command operand scan (run_command path awareness): cat /etc, cat /proc
  now require approval; CWD paths and rule-less paths are unaffected.
- The /agentic acl command dispatch (list / allow / deny / remove / reset).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as m

CWD = os.path.realpath(os.getcwd())
HOME = os.path.realpath(os.path.expanduser("~"))


class FakeCtx:
    auto_confirm = False

    def __init__(self):
        self.path_acl = m.PathAcl()


class TestPathAclDefaults(unittest.TestCase):
    def setUp(self):
        self.acl = m.PathAcl()

    def _ev(self, op, p):
        return self.acl.evaluate(op, os.path.realpath(os.path.expanduser(p)))

    def test_cwd_allowed(self):
        self.assertEqual(self._ev("read", CWD)[0], "allow")
        self.assertEqual(self._ev("read", CWD + "/sub/file.py")[0], "allow")
        self.assertEqual(self._ev("write", CWD + "/out.txt")[0], "allow")

    def test_home_explicit_read_allowed(self):
        self.assertEqual(self._ev("read", HOME + "/notes.txt")[0], "allow")

    def test_home_dotfile_ask(self):
        decision, rule = self._ev("read", HOME + "/.ssh/id_rsa")
        self.assertEqual(decision, "ask")
        self.assertEqual(rule["kind"], "home_dotfile")
        decision, rule = self._ev("write", HOME + "/.vimrc")
        self.assertEqual(decision, "ask")
        self.assertEqual(rule["kind"], "home_dotfile")

    def test_system_paths_ask_for_read(self):
        for p in ("/proc/self/cgroup", "/sys/fs/cgroup", "/etc/passwd"):
            self.assertEqual(self._ev("read", p)[0], "ask", p)

    def test_system_paths_deny_for_write(self):
        for p in ("/proc/x", "/sys/x", "/etc/passwd"):
            self.assertEqual(self._ev("write", p)[0], "deny", p)

    def test_other_outside_paths_denied(self):
        self.assertEqual(self._ev("read", "/tmp/foo")[0], "deny")
        self.assertEqual(self._ev("read", "/var/log/syslog")[0], "deny")


class TestPathAclOverrides(unittest.TestCase):
    def setUp(self):
        self.acl = m.PathAcl()

    def test_session_allow_overrides_default_ask(self):
        self.acl.add("read", "allow", "/etc", source="session")
        self.assertEqual(self.acl.evaluate("read", "/etc/passwd")[0], "allow")
        self.assertEqual(self.acl.evaluate("read", "/etc/nginx/conf.d/x")[0], "allow")

    def test_session_deny_overrides_default_ask(self):
        self.acl.add("read", "deny", "/proc", source="session")
        self.assertEqual(self.acl.evaluate("read", "/proc/cpuinfo")[0], "deny")
        # /sys unaffected
        self.assertEqual(self.acl.evaluate("read", "/sys/kernel")[0], "ask")

    def test_longest_prefix_wins(self):
        self.acl.add("read", "ask", "/etc", source="default")
        self.acl.add("read", "allow", "/etc/nginx", source="session")
        self.assertEqual(self.acl.evaluate("read", "/etc/nginx/conf.d/x")[0], "allow")
        self.assertEqual(self.acl.evaluate("read", "/etc/passwd")[0], "ask")

    def test_remove_and_reset(self):
        self.acl.add("read", "allow", "/etc", source="session")
        self.acl.remove("/etc")
        self.assertEqual(self.acl.evaluate("read", "/etc/passwd")[0], "ask")
        self.acl.add("read", "deny", "/proc", source="session")
        self.acl.reset()
        self.assertEqual(self.acl.evaluate("read", "/proc/cpuinfo")[0], "ask")


class TestRawHomeExplicit(unittest.TestCase):
    def test_explicit_home_requests(self):
        self.assertTrue(m._raw_is_home_explicit("~/notes.txt"))
        self.assertTrue(m._raw_is_home_explicit(HOME + "/notes.txt"))

    def test_relative_escapes_not_explicit(self):
        self.assertFalse(m._raw_is_home_explicit("../etc/passwd"))
        self.assertFalse(m._raw_is_home_explicit("notes.txt"))
        self.assertFalse(m._raw_is_home_explicit(None))


class TestPathAclDecision(unittest.TestCase):
    def setUp(self):
        self.ctx = FakeCtx()

    def test_system_read_ask_denied_nontty(self):
        decision, msg = m._path_acl_decision(self.ctx, "read", "/proc/version",
                                             tool="read_file", raw_path="/proc/version")
        self.assertEqual(decision, "deny")
        self.assertIn("denied by path ACL rule", msg)

    def test_system_read_allowed_after_session(self):
        self.ctx.path_acl.add("read", "allow", "/proc", source="session")
        decision, msg = m._path_acl_decision(self.ctx, "read", "/proc/version",
                                             tool="read_file", raw_path="/proc/version")
        self.assertEqual(decision, "allow")

    def test_relative_escape_into_home_denied(self):
        # ../etc/passwd resolves under $HOME but is not an explicit home request.
        rp = os.path.realpath(os.path.join(os.path.dirname(CWD), "etc", "passwd"))
        decision, msg = m._path_acl_decision(self.ctx, "read", rp,
                                             tool="read_file", raw_path="../etc/passwd")
        self.assertEqual(decision, "deny")

    def test_home_read_allowed_when_explicit(self):
        decision, _ = m._path_acl_decision(self.ctx, "read", HOME + "/notes.txt",
                                           tool="read_file", raw_path=HOME + "/notes.txt")
        self.assertEqual(decision, "allow")

    def test_sensitive_virtual_file_hard_blocked(self):
        decision, msg = m._path_acl_decision(self.ctx, "read", "/proc/1/mem",
                                             tool="read_file", raw_path="/proc/1/mem")
        self.assertEqual(decision, "deny")
        self.assertIn("System file blocked", msg)

    def test_write_to_system_path_denied(self):
        decision, msg = m._path_acl_decision(self.ctx, "write", "/etc/passwd",
                                             tool="write_file", raw_path="/etc/passwd")
        self.assertEqual(decision, "deny")


class TestShellOperandRealpath(unittest.TestCase):
    def test_flags_and_bare_words(self):
        self.assertIsNone(m._shell_operand_realpath("-r"))
        self.assertIsNone(m._shell_operand_realpath("hello"))
        self.assertIsNone(m._shell_operand_realpath("FOO=bar"))

    def test_relative_and_absolute(self):
        self.assertEqual(m._shell_operand_realpath("file.txt"),
                         os.path.realpath(os.path.join(os.getcwd(), "file.txt")))
        self.assertEqual(m._shell_operand_realpath("/etc/passwd"), "/etc/passwd")

    def test_redirect_attached(self):
        self.assertEqual(m._shell_operand_realpath(">>/etc/foo"), "/etc/foo")

    def test_glob_uses_base(self):
        self.assertEqual(m._shell_operand_realpath("/etc/*.conf"), "/etc")


class TestShellAclScan(unittest.TestCase):
    def setUp(self):
        self.ctx = FakeCtx()

    def test_cat_system_path_blocked(self):
        res = m.check_shell_approval("cat /etc/passwd", ctx=self.ctx)
        self.assertFalse(res["approved"])
        self.assertIn("path ACL", res["message"])

    def test_cat_proc_blocked(self):
        res = m.check_shell_approval("cat /proc/self/cgroup", ctx=self.ctx)
        self.assertFalse(res["approved"])

    def test_sensitive_proc_hard_blocked(self):
        res = m.check_shell_approval("cat /proc/1/mem", ctx=self.ctx)
        self.assertFalse(res["approved"])
        self.assertIn("System file blocked", res["message"])

    def test_cwd_file_allowed(self):
        probe = os.path.join(CWD, "acl_probe.txt")
        with open(probe, "w") as f:
            f.write("x")
        try:
            res = m.check_shell_approval(f"cat acl_probe.txt", ctx=self.ctx)
            self.assertTrue(res["approved"])
        finally:
            os.remove(probe)

    def test_ruleless_path_deferred(self):
        res = m.check_shell_approval("ls /var/log", ctx=self.ctx)
        self.assertTrue(res["approved"])

    def test_plain_command_unaffected(self):
        res = m.check_shell_approval("echo hi", ctx=self.ctx)
        self.assertTrue(res["approved"])

    def test_allow_after_session(self):
        self.ctx.path_acl.add("read", "allow", "/etc", source="session")
        res = m.check_shell_approval("cat /etc/hosts", ctx=self.ctx)
        self.assertTrue(res["approved"])


def _loop():
    loop = object.__new__(m.ChatLoop)
    loop.ctx = FakeCtx()
    return loop


class TestAgenticAclCommand(unittest.TestCase):
    def test_allow_adds_rule(self):
        loop = _loop()
        loop._handle_agentic_acl(['/agentic', 'acl', 'allow', '/etc', 'read'])
        decision, _ = loop.ctx.path_acl.evaluate("read", "/etc/passwd")
        self.assertEqual(decision, "allow")

    def test_deny_adds_rule(self):
        loop = _loop()
        loop._handle_agentic_acl(['/agentic', 'acl', 'deny', '/proc', 'read'])
        decision, _ = loop.ctx.path_acl.evaluate("read", "/proc/cpuinfo")
        self.assertEqual(decision, "deny")

    def test_remove_and_reset(self):
        loop = _loop()
        loop._handle_agentic_acl(['/agentic', 'acl', 'allow', '/etc', 'read'])
        loop._handle_agentic_acl(['/agentic', 'acl', 'remove', '/etc'])
        decision, _ = loop.ctx.path_acl.evaluate("read", "/etc/passwd")
        self.assertEqual(decision, "ask")
        loop._handle_agentic_acl(['/agentic', 'acl', 'allow', '/etc', 'read'])
        loop._handle_agentic_acl(['/agentic', 'acl', 'reset'])
        decision, _ = loop.ctx.path_acl.evaluate("read", "/etc/passwd")
        self.assertEqual(decision, "ask")

    def test_list_and_log_no_crash(self):
        loop = _loop()
        self.assertFalse(loop._handle_agentic_acl(['/agentic', 'acl']))
        self.assertFalse(loop._handle_agentic_acl(['/agentic', 'acl', 'log']))


class TestAclFileTools(unittest.TestCase):
    def setUp(self):
        self.ctx = FakeCtx()
        self.reg = m.ToolRegistry(ctx=self.ctx)

    def test_read_file_proc_denied_by_default(self):
        result = self.reg.execute("read_file", {"file_path": "/proc/version"})
        self.assertFalse(result["success"])
        self.assertIn("denied by path ACL rule", result.get("error", ""))

    def test_read_file_proc_allowed_after_session(self):
        self.ctx.path_acl.add("read", "allow", "/proc", source="session")
        result = self.reg.execute("read_file", {"file_path": "/proc/version"})
        self.assertTrue(result["success"])
        self.assertIn("Linux", result.get("output", ""))

    def test_write_file_etc_denied(self):
        self.ctx.auto_confirm = True  # skip the destructive-tool prompt, hit the ACL
        result = self.reg.execute("write_file",
                                  {"file_path": "/etc/test-acl-write", "content": "x"})
        self.assertFalse(result["success"])
        self.assertIn("denied by path ACL rule", result.get("error", ""))

    def test_list_directory_dot_allowed_when_cwd_in_home(self):
        # Regression: CWD lives under $HOME, so `list_directory('.')` must not
        # trip the home-explicitness re-check.
        result = self.reg.execute("list_directory", {"path": "."})
        self.assertTrue(result["success"])
        self.assertIn("ollamaquery2.py", result.get("output", ""))

    def test_read_file_readme_allowed_when_cwd_in_home(self):
        result = self.reg.execute("read_file", {"file_path": "README.md"})
        self.assertTrue(result["success"])

    def test_read_file_traversal_denied(self):
        result = self.reg.execute("read_file", {"file_path": "../etc/passwd"})
        self.assertFalse(result["success"])
        self.assertIn("denied", result.get("error", ""))


if __name__ == '__main__':
    unittest.main(verbosity=2)
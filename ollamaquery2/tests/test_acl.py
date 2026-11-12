#!/usr/bin/env python3
"""Unit tests for the Path ACL (session-scoped allow/ask/deny).

Covers:
- PathAcl defaults: CWD allowed, $HOME reads allowed (explicit only), home
  dotfiles / /proc /sys /etc are `ask`, a read anywhere else is `ask`
  (confirmed once), a write anywhere else is denied, and the shared temp area
  is denied except this session's scratch dir (`/tmp/ollamaquery/pid-N`).
- Session overrides via add()/remove(), longest-prefix precedence.
- _path_acl_decision: ask -> non-TTY deny, hard-block of sensitive /proc files,
  and the "explicit home request only" rule (../ escapes stay denied).
- Shell command operand scan (run_command path awareness): cat /etc, cat /proc
  now require approval; CWD paths and rule-less paths are unaffected.
- The session scratch dir: creation (0700), cleanup leaf-only, dead-PID sweep,
  shell leniency, plan-mode confinement, and the denial redirect.
- The /agentic acl command dispatch (list / allow / deny / remove / reset).
- fetch_url scheme guard: non-http(s) URLs (file://, ftp://, data:) are
  rejected so fetch cannot read local files and bypass the ACL.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

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

    def test_other_outside_reads_ask(self):
        # Outside CWD/$HOME a read is confirmed once (never silently denied):
        # a deny there would only push the model to `cat`, which the same ACL
        # cannot gate. Writes stay denied. The shared temp area is the one
        # exception: it is denied outright (see TestScratchDir).
        self.assertEqual(self._ev("read", "/var/log/syslog")[0], "ask")
        self.assertIsNone(self._ev("read", "/var/log/syslog")[1])

    def test_other_outside_writes_denied(self):
        self.assertEqual(self._ev("write", "/var/log/syslog")[0], "deny")
        self.assertEqual(self._ev("write", "/opt/aq-out.txt")[0], "deny")


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


class TestScratchDir(unittest.TestCase):
    """The session scratch dir is the one writable place outside CWD/$HOME."""

    def setUp(self):
        self.acl = m.PathAcl()
        self.scratch = m._scratch_root()

    def _ev(self, op, p):
        return self.acl.evaluate(op, os.path.realpath(os.path.expanduser(p)))

    def test_scratch_is_created_private(self):
        import stat
        self.assertTrue(os.path.isdir(self.scratch))
        self.assertEqual(stat.S_IMODE(os.stat(self.scratch).st_mode), 0o700)
        self.assertEqual(os.path.dirname(self.scratch), m.SCRATCH_PARENT)

    def test_scratch_read_and_write_allowed_without_a_session_grant(self):
        decision, rule = self._ev("write", self.scratch + "/build/out.o")
        self.assertEqual(decision, "allow")
        self.assertEqual(rule["source"], "default")
        self.assertEqual(self._ev("read", self.scratch + "/build/out.o")[0], "allow")

    def test_rest_of_the_temp_area_is_denied(self):
        for p in ("/tmp/x", m.SCRATCH_PARENT, m.SCRATCH_PARENT + "/pid-1/x"):
            self.assertEqual(self._ev("read", p)[0], "deny", p)
            self.assertEqual(self._ev("write", p)[0], "deny", p)
        parent_rule = self._ev("read", "/tmp/x")[1]
        self.assertEqual(parent_rule["kind"], "temp_root")

    def test_scratch_allow_beats_the_temp_deny_regardless_of_order(self):
        # The defaults install the deny first; longest-prefix must still win.
        self.assertLess(len(self._ev("read", "/tmp/x")[1]["path"]),
                        len(self._ev("write", self.scratch + "/a")[1]["path"]))
        # A re-added deny first must not change the answer either.
        self.acl.rules.reverse()
        self.assertEqual(self._ev("write", self.scratch + "/a")[0], "allow")

    def test_cwd_under_the_temp_area_stays_usable(self):
        """`cd /tmp/proj && python3 ollamaquery2.py` must not lock the session out."""
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            cwd = os.getcwd()
            os.chdir(td)
            try:
                acl = m.PathAcl()
                self.assertEqual(acl.evaluate("write", os.path.join(td, "a.txt"))[0], "allow")
                self.assertEqual(acl.evaluate("read", os.path.join(td, "a.txt"))[0], "allow")
                # Everything else in /tmp is still denied.
                self.assertEqual(acl.evaluate("read", "/tmp/elsewhere")[0], "deny")
            finally:
                os.chdir(cwd)

    def test_reset_restores_the_scratch_dir(self):
        self.acl.add("write", "deny", self.scratch, source="session")
        self.assertEqual(self._ev("write", self.scratch + "/a")[0], "deny")
        self.acl.reset()
        self.assertEqual(self._ev("write", self.scratch + "/a")[0], "allow")

    def test_session_deny_on_the_scratch_dir_wins(self):
        self.acl.add("write", "deny", self.scratch, source="session")
        self.assertEqual(self._ev("write", self.scratch + "/a")[0], "deny")

    def test_session_deny_on_a_scratch_subdirectory_is_scoped(self):
        self.acl.add("write", "deny", self.scratch + "/secret", source="session")
        self.assertEqual(self._ev("write", self.scratch + "/secret/a")[0], "deny")
        self.assertEqual(self._ev("write", self.scratch + "/other/a")[0], "allow")

    def test_plan_strict_confines_the_scratch_dir(self):
        ctx = FakeCtx()
        ctx.plan_mode = True
        ctx.plan_run = False
        decision, msg = m._path_acl_decision(ctx, "read", self.scratch + "/a",
                                             tool="read_file")
        self.assertEqual(decision, "deny")
        self.assertIn("strict plan mode is confined to", msg)
        # An explicit session read grant still works (spill paging).
        with tempfile.TemporaryDirectory() as outside:
            ctx.path_acl.add("read", "allow", outside, source="session")
            decision, msg = m._path_acl_decision(ctx, "read", outside + "/p.txt",
                                                 tool="read_file")
        self.assertEqual(decision, "allow")

    def test_prompt_and_map_name_the_scratch_dir(self):
        allowed = " ".join(self.acl.allowed_lines())
        self.assertIn(self.scratch, allowed)
        self.assertIn("wiped at exit", allowed)
        full = " ".join(self.acl.describe())
        self.assertIn(self.scratch, full)
        self.assertIn("shared temp area", full)

    def test_tmp_denial_redirects_to_the_scratch_dir(self):
        ctx = FakeCtx()
        decision, msg = m._path_acl_decision(ctx, "write", "/tmp/x", tool="write_file")
        self.assertEqual(decision, "deny")
        self.assertIn("deny /tmp", msg)
        self.assertIn(f"for temporary files use {self.scratch}", msg)
        self.assertIn("path_policy", msg)

    def test_shell_leniency_covers_the_scratch_dir_only(self):
        ctx = FakeCtx()
        with patch.object(m.PathAcl, "prompt", side_effect=AssertionError("scratch must not prompt")):
            res = m.check_shell_approval(f"rm -rf {self.scratch}/build", ctx=ctx)
        self.assertTrue(res["approved"], res.get("message"))
        with patch("sys.stdin.isatty", return_value=False):
            res = m.check_shell_approval("rm -rf /tmp/build", ctx=ctx)
        self.assertFalse(res["approved"])

    def test_relative_operand_does_not_borrow_the_scratch_leniency(self):
        nodes = m._shell_breakdown("rm -rf ../outside")
        self.assertFalse(m._shell_all_operands_inside_cwd(
            nodes, extra_roots=[self.scratch]))

    def test_write_file_in_the_scratch_dir_needs_no_confirmation(self):
        reg = m.ToolRegistry(ctx=FakeCtx())
        target = os.path.join(self.scratch, "work", "a.txt")
        with patch.object(m.ToolRegistry, "_prompt_confirm",
                          side_effect=AssertionError("scratch must not prompt")):
            result = reg.execute("write_file", {"file_path": target, "content": "x"})
        self.assertTrue(result["success"], result.get("error"))
        self.assertTrue(os.path.exists(target))

    def test_write_file_outside_the_scratch_dir_still_confirms(self):
        reg = m.ToolRegistry(ctx=FakeCtx())
        with patch.object(m.ToolRegistry, "_prompt_confirm", return_value=False) as mocked:
            result = reg.execute("write_file", {"file_path": "outside.txt", "content": "x"})
        self.assertFalse(result["success"])
        self.assertIn("Cancelled by user", result["error"])
        mocked.assert_called_once()

    def test_a_symlink_inside_the_scratch_dir_is_not_exempt(self):
        link = os.path.join(self.scratch, "escape-link")
        try:
            if os.path.lexists(link):
                os.unlink(link)
            os.symlink(os.path.join(CWD, "escape.txt"), link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        reg = m.ToolRegistry(ctx=FakeCtx())
        try:
            with patch.object(m.ToolRegistry, "_prompt_confirm", return_value=False) as mocked:
                reg.execute("write_file", {"file_path": link, "content": "x"})
            mocked.assert_called_once()
        finally:
            os.unlink(link)

    def test_cleanup_removes_the_leaf_and_keeps_the_parent(self):
        leaf = tempfile.mkdtemp(prefix="pid-", dir=m.SCRATCH_PARENT)
        with open(os.path.join(leaf, "artifact.bin"), "w") as f:
            f.write("x")
        saved = list(m._scratch_root_cache)
        m._scratch_root_cache[:] = [leaf]
        try:
            m._cleanup_scratch_dir()
        finally:
            m._scratch_root_cache[:] = saved
        self.assertFalse(os.path.exists(leaf))
        self.assertTrue(os.path.isdir(m.SCRATCH_PARENT))

    def test_sweep_keeps_live_sessions_and_removes_dead_ones(self):
        dead = os.path.join(m.SCRATCH_PARENT, "pid-999999")
        alive = os.path.join(m.SCRATCH_PARENT, f"pid-{os.getpid()}")
        os.makedirs(dead, exist_ok=True)
        os.makedirs(alive, exist_ok=True)
        try:
            with patch.object(m, "_pid_alive", side_effect=lambda pid: pid != 999999):
                m._sweep_dead_scratch_dirs()
            self.assertFalse(os.path.exists(dead))
            self.assertTrue(os.path.isdir(alive))
        finally:
            import shutil
            shutil.rmtree(dead, ignore_errors=True)
            shutil.rmtree(alive, ignore_errors=True)


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

    def test_outside_read_not_granted_names_the_remedy(self):
        # Non-TTY: the implicit ask cannot be answered, so it denies — but the
        # message must name the remedy and forbid the tool-switching workaround,
        # otherwise the model tries `cat` then `python -c`.
        decision, msg = m._path_acl_decision(self.ctx, "read", "/var/lib/aq-probe.log",
                                             tool="read_file",
                                             raw_path="/var/lib/aq-probe.log")
        self.assertEqual(decision, "deny")
        self.assertIn("path_policy", msg)
        self.assertIn("run_command", msg)

    def test_outside_read_granted_adds_session_allow(self):
        # Real prompt path (TTY + "s" = allow for the session): the grant must
        # add a session allow rule for that path so the next read is silent.
        with patch("sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="s"):
            decision, msg = m._path_acl_decision(self.ctx, "read", "/var/lib/granted.log",
                                                 tool="read_file",
                                                 raw_path="/var/lib/granted.log")
        self.assertEqual(decision, "allow")
        self.assertIsNone(msg)
        self.assertEqual(self.ctx.path_acl.evaluate("read", "/var/lib/granted.log")[0], "allow")

    def test_outside_write_denied_without_prompt(self):
        with patch.object(m.PathAcl, "prompt", side_effect=AssertionError("write must not prompt")):
            decision, msg = m._path_acl_decision(self.ctx, "write", "/opt/out.txt",
                                                 tool="write_file", raw_path="/opt/out.txt")
        self.assertEqual(decision, "deny")
        self.assertIn("write outside", msg)
        # The denial redirects to the scratch dir instead of leaving a dead end.
        self.assertIn(m._scratch_root(), msg)


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


class TestShellCodeBlobScan(unittest.TestCase):
    """Absolute paths hidden inside code blobs (`python3 -c`, `sh -c`) are seen.

    Before this pass the whole `-c` argument was one opaque token that only
    resolved to a meaningless CWD-relative path, so `python3 -c
    "open('/etc/passwd')"` was auto-approved while `cat /etc/passwd` was
    blocked — the model's documented escape route (read_file → cat → python).
    """

    def setUp(self):
        self.ctx = FakeCtx()

    def _scan(self, command):
        return m._shell_acl_scan(command, self.ctx)

    def test_python_c_read_of_etc_denied(self):
        res = self._scan("python3 -c \"print(open('/etc/passwd').read())\"")
        self.assertIsNotNone(res)
        self.assertFalse(res["approved"])
        self.assertIn("path ACL", res["reason"])

    def test_sh_c_read_of_etc_denied(self):
        res = self._scan("sh -c 'cat /etc/passwd'")
        self.assertIsNotNone(res)
        self.assertFalse(res["approved"])

    def test_dd_assignment_operand_denied(self):
        res = self._scan("dd if=/etc/shadow of=/tmp/x")
        self.assertIsNotNone(res)
        self.assertFalse(res["approved"])

    def test_tilde_path_inside_code_denied(self):
        res = self._scan("python3 -c \"print(open('~/.ssh/id_rsa').read())\"")
        self.assertIsNotNone(res)
        self.assertFalse(res["approved"])

    def test_sensitive_virtual_file_in_code_hard_blocked(self):
        res = self._scan("python3 -c \"print(open('/proc/1/mem').read())\"")
        self.assertIsNotNone(res)
        self.assertIn("System file blocked", res["reason"])

    def test_end_to_end_gate_blocks_python_code(self):
        res = m.check_shell_approval(
            "python3 -c \"print(open('/etc/passwd').read())\"", ctx=self.ctx)
        self.assertFalse(res["approved"])

    def test_commit_message_mentioning_etc_not_scanned(self):
        # Prose that names a path without touching it must not prompt.
        self.assertIsNone(self._scan("git commit -m 'fix /etc handling'"))

    def test_code_path_without_rule_not_flagged(self):
        # A path with no explicit rule is left to the shell gate / file tools:
        # the code-blob scan must not invent a rule for it.
        self.assertIsNone(self._scan("python3 -c \"print(open('/var/log/aq/x').read())\""))

    def test_tmp_deny_is_visible_to_the_code_blob_scan(self):
        """`python3 -c "open('/tmp/x','w')"` must be denied by the temp-area rule."""
        res = self._scan("python3 -c \"print(open('/tmp/x').read())\"")
        self.assertIsNotNone(res)
        self.assertIn("deny /tmp", res["reason"])
        self.assertIn(m._scratch_root(), res["reason"])

    def test_url_not_treated_as_path(self):
        self.assertIsNone(self._scan("curl -o out.html https://example.com/a/b"))


class TestAclDenialPointsAtTool(unittest.TestCase):
    """A denial is one line: the reason + a pointer to the `path_policy` tool.

    It deliberately does NOT re-send the policy map. The allowed roots are in the
    system prompt, and the full map is one `path_policy` call away, so a repeated
    refusal costs ~200 chars instead of ~700 — and the model still has a defined
    next step (`path_policy`) instead of improvising with `cat`/`python -c`.
    """

    def setUp(self):
        self.ctx = FakeCtx()

    def _deny(self, op="read", path="/var/lib/aq-probe.log", op_name="read_file"):
        decision, msg = m._path_acl_decision(self.ctx, op, path, tool=op_name, raw_path=path)
        self.assertEqual(decision, "deny")
        return msg

    def test_denial_is_labelled_short_and_points_at_the_tool(self):
        msg = self._deny()
        self.assertTrue(msg.startswith("[path ACL denied]"), msg[:80])
        self.assertIn("outside the project and $HOME, not granted", msg)
        self.assertIn("path_policy", msg)
        self.assertIn("do not bypass", msg)
        self.assertNotIn("path policy for this session:", msg,
                         "the map belongs in path_policy / the prompt, not in every denial")
        self.assertLess(len(msg), 400)

    def test_hard_block_also_points_at_the_tool(self):
        decision, msg = m._path_acl_decision(self.ctx, "read", "/proc/1/mem",
                                             tool="read_file", raw_path="/proc/1/mem")
        self.assertEqual(decision, "deny")
        self.assertIn("System file blocked", msg)
        self.assertIn("path_policy", msg)

    def test_shell_denial_matches(self):
        res = m._shell_acl_scan("python3 -c \"print(open('/etc/passwd').read())\"", self.ctx)
        self.assertIn("[path ACL denied]", res["reason"])
        self.assertIn("path_policy", res["reason"])
        self.assertNotIn("path policy for this session:", res["reason"])
        self.assertLess(len(res["reason"]), 400)

    def test_plan_strict_denial_says_cwd_only(self):
        self.ctx.plan_mode = True
        self.ctx.plan_run = False
        msg = self._deny(path="/etc/passwd")
        self.assertIn("strict plan mode is confined to", msg)
        self.assertIn("path_policy", msg)


class TestPathPolicyTool(unittest.TestCase):
    """`path_policy` is the on-demand map: no filesystem access, no prompt."""

    def setUp(self):
        self.ctx = FakeCtx()
        self.ctx.agentic_mode = True
        self.reg = m.ToolRegistry(ctx=self.ctx)

    def _tool(self, **args):
        return self.reg.execute("path_policy", args)

    def test_map_lists_allowed_ask_denied_and_grants(self):
        self.ctx.path_acl.add("read", "allow", "/tmp", source="session")
        out = self._tool()["output"]
        self.assertIn("path policy for this session:", out)
        self.assertIn(CWD, out)
        self.assertIn("ask", out)
        self.assertIn("/proc/*/mem", out)
        self.assertIn("allow read: /tmp", out)
        self.assertIn("/agentic acl allow", out)

    def test_deny_rules_are_listed_before_grants(self):
        self.ctx.path_acl.add("read", "allow", "/tmp", source="session")
        self.ctx.path_acl.add("read", "deny", "~/.ssh", source="session")
        out = self._tool()["output"]
        self.assertLess(out.index("deny read"), out.index("allow read"))

    def test_map_is_bounded(self):
        for i in range(20):
            self.ctx.path_acl.add("read", "allow", f"/opt/grant{i}", source="session")
        out = self._tool()["output"]
        self.assertIn("more)", out)
        self.assertLess(len(out), 1200)

    def test_cwd_path_reports_allow(self):
        self.assertIn("ALLOW", self._tool(path="./some_file.txt")["output"])

    def test_rule_path_reports_ask_without_prompting(self):
        with patch.object(m.PathAcl, "prompt", side_effect=AssertionError("must not prompt")):
            out = self._tool(path="/etc/passwd")["output"]
        self.assertIn("ASK", out)
        self.assertIn("/etc", out)

    def test_sensitive_virtual_file_reports_hard_deny(self):
        out = self._tool(path="/proc/1/mem")["output"]
        self.assertIn("DENIED", out)
        self.assertIn("hard-blocked", out)

    def test_bad_op_is_an_error(self):
        res = self._tool(path="/tmp/x", op="delete")
        self.assertFalse(res["success"])
        self.assertIn("read", res["error"])

    def test_available_in_strict_plan_mode(self):
        self.ctx.plan_mode = True
        self.ctx.plan_run = False
        self.assertIn("path_policy", m.plan_active_tools(self.ctx))
        self.assertIn("path_policy", self.reg._plan_allowed_tools())

    def test_plan_strict_cwd_allowed_home_read_denied(self):
        self.ctx.plan_mode = True
        self.ctx.plan_run = False
        self.assertIn("ALLOW", self._tool(path="./x.txt")["output"])
        self.assertIn("DENY", self._tool(path="~/notes.txt")["output"])


class TestAgenticPromptPathPolicy(unittest.TestCase):
    """The prompt states the POSITIVE half only; the map lives behind the tool."""

    def setUp(self):
        self.ctx = m.CommandContext()
        self.ctx.backend = 'llamacpp'
        self.ctx.base_url = 'http://127.0.0.1:1'
        self.ctx.model = 'qwen3.5:9b'
        self.ctx.agentic_mode = True
        self.ctx.plan_mode = False
        self.ctx.plan_run = False
        self.loop = m.ChatLoop(self.ctx)
        # CommandContext is a process-wide singleton: restore the ACL rules and
        # the plan flags after each test, or a session grant added here (e.g.
        # `read allow /tmp`) silently changes every later module's expectations.
        rules = list(self.ctx.path_acl.rules)

        def _restore():
            self.ctx.path_acl.rules = list(rules)
            self.ctx.plan_mode = False
            self.ctx.plan_run = False
            self.ctx.agentic_mode = False
        self.addCleanup(_restore)

    def test_agentic_prompt_lists_allowed_paths_only(self):
        self.ctx.path_acl.add("read", "allow", "/tmp", source="session")
        system, _tools, _native = self.loop._agentic_prompt_and_tools()
        self.assertIn("## Path access", system)
        self.assertIn("allowed here:", system)
        self.assertIn(CWD, system)                       # what the model may use
        self.assertIn("granted this session: /tmp", system)
        self.assertIn("path_policy", system)             # where the rest lives
        self.assertNotIn("path policy for this session:", system,
                         "the full ask/denied map belongs to the path_policy tool")

    def test_plan_prompt_restricts_to_cwd(self):
        self.ctx.plan_mode = True
        self.ctx.plan_run = False
        system, _tools, _native = self.loop._agentic_prompt_and_tools()
        self.assertIn("read only — strict plan mode", system)


class TestRulesBlockPathPolicyRule(unittest.TestCase):
    """The shared rules block makes `path_policy` the reflex after a denial.

    The denial line already names the tool; this rule makes it a standing
    instruction for every model and every prompt style (native/inline, normal
    and plan), so the model does not improvise with `cat`/`python -c`.
    """

    def test_rule_present_for_every_prompt_style(self):
        for model in ('qwen3.5:9b', 'nemotron-cascade', 'some-unknown:latest'):
            for plan in (False, True):
                prompt = m.get_agentic_prompt(model, plan_mode=plan)
                self.assertIn("Call path_policy once", prompt, model)
                self.assertIn("do NOT retry it with run_command/run_python", prompt, model)

    def test_dedicated_tools_are_preferred_over_shell_equivalents(self):
        """Tool choice is steered, not forbidden: shell = run, tools = file I/O."""
        for model in ('qwen3.5:9b', 'nemotron-cascade', 'some-unknown:latest'):
            for plan in (False, True):
                prompt = m.get_agentic_prompt(model, plan_mode=plan)
                self.assertIn("instead of shell equivalents", prompt, model)
                self.assertIn("Use run_command (when a shell is available) to RUN things",
                              prompt, model)

    def test_read_file_description_does_not_invite_the_bypass(self):
        """A denial must not point at run_command — that is the ACL bypass."""
        desc = m.AGENTIC_TOOL_DEFS["read_file"]["description"]
        self.assertNotIn("use run_command instead", desc)
        self.assertIn("call path_policy once", desc)
        self.assertIn("do not route around the refusal", desc)

    def test_write_file_description_names_the_scratch_dir_for_temporary_files(self):
        desc = m.AGENTIC_TOOL_DEFS["write_file"]["description"]
        self.assertIn("session scratch dir", desc)
        self.assertIn("wiped at exit", desc)


class TestShellApprovalPrompt(unittest.TestCase):
    """The `a` answer is session-scoped (nothing persisted) and broader than `s`."""

    def setUp(self) -> None:
        m._SHELL_SESSION_APPROVED.clear()

    def tearDown(self) -> None:
        m._SHELL_SESSION_APPROVED.clear()

    def _verdict(self) -> dict:
        verdict = m.shell_check_command("ssh host", m.DEFAULT_SHELL_PERMISSION)
        self.assertEqual(verdict["effect"], "ask")
        self.assertEqual(verdict["patterns"], ["ssh host"])
        self.assertEqual(verdict["always"], ["ssh *"])
        return verdict

    def test_prompt_label_does_not_claim_always(self):
        with patch("builtins.input", return_value="a") as fake_input:
            m._shell_prompt_approval("ssh host", self._verdict())
        prompt = fake_input.call_args[0][0]
        self.assertIn("a=session", prompt)
        self.assertNotIn("a=always", prompt)

    def test_always_answer_adds_exact_and_broader_patterns(self):
        with patch("builtins.input", return_value="a"):
            res = m._shell_prompt_approval("ssh host", self._verdict())
        self.assertTrue(res["approved"])
        added = [rule["pattern"] for rule in m._SHELL_SESSION_APPROVED]
        self.assertIn("ssh host", added)
        self.assertIn("ssh *", added)

    def test_session_answer_adds_only_exact_pattern(self):
        with patch("builtins.input", return_value="s"):
            m._shell_prompt_approval("ssh host", self._verdict())
        added = [rule["pattern"] for rule in m._SHELL_SESSION_APPROVED]
        self.assertEqual(added, ["ssh host"])


class TestFetchUrlSchemeGuard(unittest.TestCase):
    """fetch_url must not become a local-read primitive (Path ACL bypass)."""

    def test_file_scheme_does_not_read_local_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("SENTINEL-LOCAL-READ")
            path = f.name
        try:
            text, tool = m.fetch_and_convert_url("file://" + path)
        finally:
            os.unlink(path)
        self.assertEqual(tool, "None")
        self.assertTrue(text.startswith("[Failed to fetch URL:"), text)
        self.assertNotIn("SENTINEL-LOCAL-READ", text)

    def test_rejected_schemes_and_malformed_urls(self):
        for url in ("file:///etc/hostname", "ftp://example.com/x",
                    "data:text/plain,hi", "gopher://example.com/",
                    "/etc/passwd", "http://", "http:/example.com"):
            with self.subTest(url=url):
                text, tool = m.fetch_and_convert_url(url)
                self.assertEqual(tool, "None")
                self.assertTrue(text.startswith("[Failed to fetch URL:"), text)

    def test_curl_fallback_rejects_file_scheme(self):
        self.assertEqual(m._curl_fetch_text("file:///etc/hostname"), ("", False))

    def test_fetch_tool_reports_scheme_rejection(self):
        out = m._tool_handle_fetch_url(None, {"url": "file:///etc/hostname"})
        self.assertFalse(out["success"])
        self.assertIn("only http", out["error"])

    def test_http_and_https_pass_the_guard(self):
        for url in ("http://example.com/x", "HTTPS://example.com/y"):
            with self.subTest(url=url):
                with patch("ollamaquery2._request_with_retry",
                           side_effect=Exception("transport-reached")), \
                     patch("ollamaquery2._curl_fetch_text", return_value=("", False)):
                    text, tool = m.fetch_and_convert_url(url)
                self.assertIn("transport-reached", text)


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

    def test_list_shows_how_to_change_the_rules(self):
        """Bare `/agentic acl` is also the help screen: the commands must be visible."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _loop()._handle_agentic_acl(['/agentic', 'acl'])
        out = err.getvalue()
        self.assertIn("/agentic acl allow <path> [op]", out)
        self.assertIn("e.g. /agentic acl allow /tmp write", out)
        self.assertIn("/agentic acl remove <path>", out)
        self.assertIn("/agentic acl reset", out)
        self.assertIn("/agentic scratch", out)

    def test_allowing_a_path_that_contains_the_scratch_dir_warns(self):
        """`allow /tmp write` defeats the scratch redirect — say so, out loud."""
        loop = _loop()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop._handle_agentic_acl(['/agentic', 'acl', 'allow', '/tmp', 'write'])
        self.assertIn("widens the model's writable surface", err.getvalue())
        self.assertEqual(loop.ctx.path_acl.evaluate("write", "/tmp/x")[0], "allow")

    def test_allowing_the_scratch_dir_itself_does_not_warn(self):
        loop = _loop()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop._handle_agentic_acl(['/agentic', 'acl', 'allow',
                                      m._scratch_root(), 'write'])
        self.assertNotIn("widens the model's writable surface", err.getvalue())


class TestAgenticScratchCommand(unittest.TestCase):
    """`/agentic scratch` makes the one writable temp path visible to the user."""

    def test_bare_prints_the_path_and_the_cleanup_rule(self):
        loop = _loop()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertFalse(loop.run_handle_agentic('/agentic scratch'))
        self.assertIn(m._scratch_root(), err.getvalue())
        self.assertIn("removed at session exit", err.getvalue())

    def test_list_then_clear(self):
        loop = _loop()
        artifact = os.path.join(m._scratch_root(), "artifact.txt")
        with open(artifact, "w") as f:
            f.write("x")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop._handle_agentic_scratch(['/agentic', 'scratch', 'list'])
        self.assertIn("artifact.txt", err.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            loop._handle_agentic_scratch(['/agentic', 'scratch', 'clear'])
        self.assertFalse(os.path.exists(artifact))

    def test_unknown_action_prints_usage(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _loop()._handle_agentic_scratch(['/agentic', 'scratch', 'nonsense'])
        self.assertIn("Usage: /agentic scratch", err.getvalue())

    def test_completion_offers_the_actions(self):
        self.assertEqual(m.AGENTIC_SECOND_LEVEL['scratch'], ['path', 'list', 'clear'])
        self.assertIn('scratch', m.AGENTIC_SUBCOMMANDS)


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
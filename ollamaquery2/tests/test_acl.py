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
- Shell gate contracts: the home/root hard guard reaches wrappers, assignments,
  `find`/`xargs`, `env -S` and a sensitive `cd`; CWD leniency is not vacuous,
  follows `cd`, and refuses wrappers/`xargs`/shell `-c`; redirect targets use
  the full write policy.
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

    def test_prompt_y_allows_once_without_installing_a_rule(self):
        """`y` must not behave like `s`: no session rule, only this access."""
        rule_count = len(self.ctx.path_acl.rules)
        with patch("sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="y"):
            decision, msg = m._path_acl_decision(
                self.ctx, "read", "/var/lib/aq-once.log",
                tool="read_file", raw_path="/var/lib/aq-once.log")
        self.assertEqual(decision, "allow")
        self.assertIsNone(msg)
        self.assertEqual(len(self.ctx.path_acl.rules), rule_count)
        self.assertEqual(
            self.ctx.path_acl.evaluate("read", "/var/lib/aq-once.log")[0], "ask")

    def test_prompt_y_on_a_prefix_rule_does_not_open_the_whole_rule(self):
        """A `y` on the /etc ask rule must not install `allow read /etc`."""
        with patch("sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="y"):
            decision, _ = m._path_acl_decision(
                self.ctx, "read", "/etc/passwd", tool="read_file",
                raw_path="/etc/passwd")
        self.assertEqual(decision, "allow")
        self.assertEqual(self.ctx.path_acl.evaluate("read", "/etc/shadow")[0], "ask")

    def test_project_files_named_like_proc_entries_are_not_hard_blocked(self):
        """The hard block is /proc and /sys only, not an ordinary basename."""
        for path in ("/project/docs/maps", "/project/src/mem",
                     "/project/tests/environ", "/project/data/smaps"):
            self.assertFalse(m._is_blocked_system_file(path), path)
        for path in ("/proc/self/maps", "/proc/1/mem", "/proc/self/environ",
                     "/proc/kcore", "/sys/kernel/maps"):
            self.assertTrue(m._is_blocked_system_file(path), path)

    def test_read_file_reads_a_project_file_named_maps(self):
        with tempfile.TemporaryDirectory() as td:
            old_cwd = os.getcwd()
            os.chdir(td)
            try:
                os.makedirs("docs")
                with open(os.path.join("docs", "maps"), "w", encoding="utf-8") as f:
                    f.write("project map\n")
                reg = m.ToolRegistry(ctx=self.ctx)
                result = reg.execute("read_file", {"file": "docs/maps"})
                self.assertTrue(result["success"], msg=result.get("error"))
                self.assertIn("project map", result["output"])
            finally:
                os.chdir(old_cwd)

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


class TestShellNewlineSeparator(unittest.TestCase):
    """A newline splits commands exactly like `;` (C1).

    `_shell_tokenize` treated `\\n` as ordinary whitespace while `shell=True`
    treats it as a command separator, so `echo hi\\nrm -rf ~` was one node whose
    head was `echo`: the `echo *` allow rule matched the whole line and the
    home guard never saw the `rm`.
    """

    def setUp(self):
        self.ctx = FakeCtx()

    def _patterns(self, command):
        verdict = m.shell_check_command(command, m.DEFAULT_SHELL_PERMISSION)
        return [node["pattern"] for node in verdict["nodes"]]

    def test_second_line_is_a_node_of_its_own(self):
        self.assertEqual(self._patterns("echo hi\nrm -rf /tmp/x"),
                         ["echo hi", "rm -rf /tmp/x"])

    def test_destructive_second_line_hits_the_home_guard(self):
        res = m.check_shell_approval("echo hi\nrm -rf ~", ctx=self.ctx)
        self.assertFalse(res["approved"])
        self.assertIn("BLOCKED (rejected)", res["message"])

    def test_destructive_second_line_hits_the_home_guard_behind_an_allow_head(self):
        # The exact reproduced case: `echo *` is allow, so nothing but the
        # newline split can put `rm` in front of the home guard.
        res = m.check_shell_approval("echo hi\nrm -rf $HOME", ctx=self.ctx)
        self.assertFalse(res["approved"])

    def test_crlf_is_a_single_separator(self):
        self.assertEqual(self._patterns("echo hi\r\nrm -rf /tmp/x"),
                         ["echo hi", "rm -rf /tmp/x"])

    def test_multiple_lines_are_all_gated(self):
        self.assertEqual(self._patterns("ls\npwd\necho done"),
                         ["ls", "pwd", "echo done"])

    def test_backslash_continuation_is_not_a_separator(self):
        # `\` + newline is line continuation for bash; it must not split.
        patterns = self._patterns("echo hi \\\nrm -rf /tmp/x")
        self.assertEqual(len(patterns), 1)
        self.assertIn("rm -rf /tmp/x", patterns[0])

    def test_heredoc_body_is_not_parsed_as_commands(self):
        cmd = ("cat > out.c <<'EOF'\n#include <stdio.h>\n"
               "int main(void){return 0;}\nEOF")
        self.assertEqual(len(self._patterns(cmd)), 1)

    def test_heredoc_body_cannot_swallow_the_next_command(self):
        # A `<<` inside the body must not queue a bogus heredoc that eats the
        # rest of the line: that was a real bypass during development.
        cmd = "cat > out.txt <<EOF\ncout << x\nEOF\nrm -rf ~"
        self.assertEqual(len(self._patterns(cmd)), 2)
        res = m.check_shell_approval(cmd, ctx=self.ctx)
        self.assertFalse(res["approved"])

    def test_unquoted_heredoc_substitution_is_still_gated(self):
        # bash expands (and therefore executes) `$(...)` in an unquoted body.
        res = m.check_shell_approval("cat > out.txt <<EOF\n$(rm -rf ~)\nEOF",
                                     ctx=self.ctx)
        self.assertFalse(res["approved"])

    def test_quoted_heredoc_body_is_data(self):
        res = m.check_shell_approval("cat > out.txt <<'EOF'\n$(rm -rf ~)\nEOF",
                                     ctx=self.ctx)
        self.assertTrue(res["approved"], res.get("message"))

    def test_unresolvable_delimiter_fails_closed(self):
        # `$'EOF'` cannot be reproduced by this parser, so the heredoc is
        # ignored and the body is gated as commands (over-prompting beats
        # hiding the trailing `rm`).
        res = m.check_shell_approval("cat <<$'EOF'\nbody\nrm -rf ~", ctx=self.ctx)
        self.assertFalse(res["approved"])

    def test_two_heredocs_on_one_line(self):
        cmd = "cat <<-A <<B\n\tbody a\nA\nbody b\nB\nrm -rf ~"
        self.assertEqual(len(self._patterns(cmd)), 2)
        self.assertFalse(m.check_shell_approval(cmd, ctx=self.ctx)["approved"])


class TestShellRedirectTargets(unittest.TestCase):
    """Write-redirection targets are operands (C2).

    `_shell_arity_tokens` drops the operator *and its target*, so no downstream
    check ever saw them: `echo hacked > ~/.bashrc` and `echo x >> /etc/hosts`
    were `allow` (the whole line matched `echo *`) while `touch ~/.bashrc` was
    `ask`.
    """

    def setUp(self):
        self.ctx = FakeCtx()

    def _scan(self, command):
        return m._shell_acl_scan(command, self.ctx)

    def test_target_is_not_an_argument(self):
        node = m._shell_breakdown("echo hi > out.txt")[0]
        self.assertEqual(node.tokens, ["echo", "hi"])
        self.assertEqual(node.redirect_targets, ["out.txt"])

    def test_redirect_to_dotfile_asks(self):
        res = self._scan("echo hacked > ~/.bashrc")
        self.assertIsNotNone(res)
        self.assertFalse(res["approved"])
        self.assertIn(".bashrc", res["reason"])
        self.assertIn("ask", res["reason"])

    def test_redirect_to_etc_is_not_allowed(self):
        res = self._scan("echo x >> /etc/hosts")
        self.assertIsNotNone(res)
        self.assertFalse(res["approved"])
        self.assertIn("/etc/hosts", res["reason"])

    def test_redirect_inside_cwd_stays_allowed(self):
        self.assertIsNone(self._scan("echo x > out.txt"))

    def test_leniency_does_not_cover_an_outside_target(self):
        # The argument is in the project, the destination is not: the old
        # CWD-leniency check only looked at arguments and auto-allowed this.
        self.assertFalse(m._shell_all_operands_inside_cwd(
            m._shell_breakdown("python3 gen.py > /etc/config")))

    def test_leniency_still_covers_inside_and_discard_targets(self):
        self.assertTrue(m._shell_all_operands_inside_cwd(
            m._shell_breakdown("rm -rf build > out.txt")))
        self.assertTrue(m._shell_all_operands_inside_cwd(
            m._shell_breakdown("rm -rf build 2>/dev/null")))

    def test_dev_null_and_terminal_devices_are_ignored(self):
        for cmd in ("make -j4 2>/dev/null", "make -j4 >/dev/null 2>&1",
                    "echo hi > /dev/tty", "echo hi >&2"):
            with self.subTest(cmd=cmd):
                self.assertIsNone(self._scan(cmd))

    def test_gate_prompts_for_a_redirect_to_a_dotfile(self):
        with patch("sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="n") as asked:
            res = m.check_shell_approval("echo hacked > ~/.bashrc", ctx=self.ctx)
        self.assertFalse(res["approved"])
        self.assertTrue(asked.called)

    def test_gate_denies_redirect_to_etc_without_a_tty(self):
        with patch("sys.stdin.isatty", return_value=False):
            res = m.check_shell_approval("echo x >> /etc/hosts", ctx=self.ctx)
        self.assertFalse(res["approved"])
        self.assertIn("path ACL", res["message"])

    def test_read_redirect_target_is_not_a_write_target(self):
        node = m._shell_breakdown("wc -l < input.txt")[0]
        self.assertEqual(node.redirect_targets, [])


class TestShellHomeRefExpansion(unittest.TestCase):
    """`$HOME`/`${HOME}` must grade exactly like a literal `~` (G1).

    `_shell_operand_realpath` bailed out on any `$` (env refs are not simple
    paths) and the embedded-path regex could not match the `/` after `E` — so
    `echo hacked > $HOME/.bashrc` and `cat $HOME/.bashrc` reached no rule at all
    while the `~` spelling asked. Verified against bash: both spellings write and
    read the same file (`HOME=<tmp> bash -c 'echo x > $HOME/.bashrc'`).
    """

    def setUp(self):
        self.ctx = FakeCtx()

    def _scan(self, command):
        return m._shell_acl_scan(command, self.ctx)

    def test_redirect_target_is_expanded(self):
        for cmd in ("echo hacked > $HOME/.bashrc",
                    "echo hacked > ${HOME}/.bashrc"):
            with self.subTest(cmd=cmd):
                res = self._scan(cmd)
                self.assertIsNotNone(res, cmd)
                self.assertFalse(res["approved"])
                self.assertIn(".bashrc", res["reason"])

    def test_operand_is_expanded(self):
        for cmd in ("cat $HOME/.bashrc", "head ${HOME}/.ssh/id_rsa"):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(self._scan(cmd), cmd)

    def test_braced_form_resolves_the_right_path(self):
        # `${HOME}` used to slip through the regex and resolve `/.bashrc`.
        self.assertEqual(m._shell_operand_realpath("${HOME}/.bashrc"),
                         os.path.realpath(os.path.expanduser("~/.bashrc")))

    def test_code_blob_is_expanded(self):
        self.assertIsNotNone(self._scan("sh -c 'cat $HOME/.bashrc'"))

    def test_home_non_dotfile_is_unchanged(self):
        # A write under $HOME is allowed today for `~`; expanding must not change
        # the grade of the same path spelled the other way.
        self.assertIsNone(self._scan("echo x > $HOME/notes.txt"))
        self.assertIsNone(self._scan("echo x > ~/notes.txt"))

    def test_longer_variable_name_is_not_rewritten(self):
        self.assertEqual(m._shell_expand_home_refs("$HOMEDIR/x"), "$HOMEDIR/x")
        self.assertIsNone(self._scan("echo x > $HOMEDIR/notes.txt"))

    def test_quoted_home_still_asks_conservatively(self):
        # bash expands neither `~` nor `$HOME` inside quotes, so these create a
        # literal name under the CWD. The scanner strips quotes and over-prompts;
        # that is the safe direction and matches the pre-existing `~` behaviour,
        # so it is pinned rather than "fixed" silently.
        for cmd in ('echo x > "~/.bashrc"', "echo x > '$HOME/.bashrc'"):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(self._scan(cmd))


class TestShellLineContinuationInWords(unittest.TestCase):
    """`\\`+newline is deleted before tokenizing, even inside a word (G2).

    `_shell_scan_word` kept the two characters, so the target became
    `\\⏎~/.bashrc`: no leading `~`, `expanduser` a no-op, realpath under the CWD,
    CWD leniency, allow — while bash writes `~/.bashrc`.
    """

    def setUp(self):
        self.ctx = FakeCtx()

    def _scan(self, command):
        return m._shell_acl_scan(command, self.ctx)

    def test_leading_continuation_in_the_target(self):
        res = self._scan("echo hacked > \\\n~/.bashrc")
        self.assertIsNotNone(res)
        self.assertIn(".bashrc", res["reason"])

    def test_continuation_inside_the_word(self):
        self.assertIsNotNone(self._scan("echo hacked > ~/\\\n.bashrc"))

    def test_continuation_inside_double_quotes(self):
        self.assertIsNotNone(self._scan('echo hacked > "~/\\\n.bashrc"'))

    def test_continuation_does_not_split_the_command(self):
        # `echo hi \<NL>rm -rf ~` is one command for bash; the guard must still
        # see the `rm` (fail-closed).
        nodes = m._shell_breakdown("echo hi \\\nrm -rf ~")
        self.assertEqual(len(nodes), 1)
        self.assertIn("rm -rf ~", nodes[0].pattern)

    def test_continuation_joins_a_word_like_bash(self):
        self.assertEqual(m._shell_breakdown("echo a\\\nb")[0].tokens,
                         ["echo", "ab"])

    def test_no_phantom_empty_operand(self):
        self.assertEqual(m._shell_breakdown("echo a \\\n b")[0].tokens,
                         ["echo", "a", "b"])


class TestShellFdRedirectTargets(unittest.TestCase):
    """`>&file` is a file write; `2>&1`/`>&-` are fd operations (G3).

    `>&` was excluded from the write-redirection set on the assumption that its
    target is always an fd number. Bash accepts `>&file` (verified: it writes the
    file), so `echo hacked >&~/.bashrc` was an `allow` whose destination no rule
    inspected.
    """

    def setUp(self):
        self.ctx = FakeCtx()

    def _scan(self, command):
        return m._shell_acl_scan(command, self.ctx)

    def test_file_target_is_collected(self):
        node = m._shell_breakdown("echo hacked >&~/.bashrc")[0]
        self.assertEqual(node.redirect_targets, ["~/.bashrc"])

    def test_file_target_is_gated(self):
        self.assertIsNotNone(self._scan("echo hacked >&~/.bashrc"))
        self.assertIsNotNone(self._scan("echo hacked >&/etc/hosts"))

    def test_fd_forms_stay_silent(self):
        for cmd in ("echo x >&1", "echo x >&-", "echo x 2>&1"):
            with self.subTest(cmd=cmd):
                self.assertIsNone(self._scan(cmd))
                self.assertEqual(m._shell_breakdown(cmd)[0].redirect_targets, [])
        # A real target *after* an fd duplication is still collected (and still
        # silent, because /dev/null is a discard device).
        self.assertIsNone(self._scan("echo x 2>&1 >/dev/null"))
        self.assertEqual(
            m._shell_breakdown("echo x 2>&1 >/dev/null")[0].redirect_targets,
            ["/dev/null"])

    def test_digit_target_is_not_a_file(self):
        self.assertEqual(m._shell_breakdown("echo x >&1")[0].redirect_targets, [])


class TestShellRedirectWriteSemantics(unittest.TestCase):
    """A redirect target is a write, so the write half of the policy grades it (G4).

    `>> /etc/hosts` used to be scanned with `evaluate("read", …)`: the `/etc` rule
    asks, and `auto_confirm` answers yes. The C2 fix's own promise is "writes
    outside the project stay deny", so the target is graded as a write; an
    operand stays read (the command may read or write it).
    """

    def setUp(self):
        self.ctx = FakeCtx()

    def _scan(self, command):
        return m._shell_acl_scan(command, self.ctx)

    def test_etc_target_is_a_hard_deny(self):
        for cmd in ("echo x >> /etc/hosts", "echo x > /etc/config",
                    "echo x >&/etc/hosts"):
            with self.subTest(cmd=cmd):
                res = self._scan(cmd)
                self.assertIsNotNone(res, cmd)
                self.assertFalse(res["approved"])
                self.assertIn("denied by path ACL rule", res["reason"], cmd)

    def test_proc_and_sys_targets_are_hard_denies(self):
        for cmd in ("echo x > /proc/self/comm", "echo x > /sys/kernel/x"):
            with self.subTest(cmd=cmd):
                res = self._scan(cmd)
                self.assertIn("denied by path ACL rule", res["reason"])

    def test_auto_confirm_cannot_approve_it(self):
        self.ctx.auto_confirm = True
        res = m.check_shell_approval("echo x >> /etc/hosts", ctx=self.ctx)
        self.assertFalse(res["approved"])

    def test_home_dotfile_still_asks(self):
        res = self._scan("echo x > ~/.bashrc")
        self.assertFalse(res["approved"])
        self.assertIn("no TTY to ask", res["reason"])

    def test_cwd_target_is_allowed(self):
        self.assertIsNone(self._scan("echo x > out.txt"))

    def test_outside_relative_target_is_denied(self):
        # Redirect targets use the full write policy, not explicit-only:
        # the default outside-CWD write deny applies.
        res = self._scan("echo x > ../outside.txt")
        self.assertIsNotNone(res)
        self.assertFalse(res["approved"])
        self.assertIn("denied by path ACL rule", res["reason"])
        gate = m.check_shell_approval("echo x > ../outside.txt", ctx=self.ctx)
        self.assertFalse(gate["approved"])

    def test_cd_redirect_outside_is_denied(self):
        gate = m.check_shell_approval("cd . > ../outside.txt", ctx=self.ctx)
        self.assertFalse(gate["approved"])
        self.assertIn("outside the project", gate.get("message", ""))

    def test_home_escape_redirect_is_denied(self):
        gate = m.check_shell_approval("echo x > ~/../outside.txt", ctx=self.ctx)
        self.assertFalse(gate["approved"])

    def test_tmp_target_keeps_the_scratch_hint(self):
        res = self._scan("echo x > /tmp/out.txt")
        self.assertIn("temporary files", res["reason"])


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

    def test_implicit_home_read_is_denied_like_the_real_tool(self):
        """P2: a relative escape into $HOME must not be announced as ALLOW.

        `path_policy` used the bare `PathAcl.evaluate` implicit home allowance;
        `_path_acl_decision` additionally requires an explicit home request. A
        model sent to `../etc/passwd` (which resolves under $HOME) was told
        ALLOW and then refused by read_file.
        """
        if CWD == HOME:
            self.skipTest("CWD is $HOME: every home path is also inside the project")
        target = os.path.realpath(os.path.join(HOME, "aq-path-policy-probe.txt"))
        raw = os.path.relpath(target, CWD)
        self.assertFalse(m._raw_is_home_explicit(raw))
        out = self._tool(path=raw, op="read")["output"]
        self.assertIn("DENY", out)
        self.assertIn("not requested explicitly", out)
        decision, msg = m._path_acl_decision(self.ctx, "read", target,
                                             tool="read_file", raw_path=raw)
        self.assertEqual(decision, "deny")
        self.assertIn("not requested explicitly", msg)

    def test_explicit_home_read_is_allowed_like_the_real_tool(self):
        raw = "~/aq-path-policy-probe.txt"
        self.assertIn("ALLOW", self._tool(path=raw, op="read")["output"])
        decision, _ = m._path_acl_decision(
            self.ctx, "read", os.path.realpath(os.path.expanduser(raw)),
            tool="read_file", raw_path=raw)
        self.assertEqual(decision, "allow")

    def test_plan_strict_scratch_dir_is_denied_like_the_real_tool(self):
        """P2: strict plan denies the default-allow scratch dir outside CWD.

        The tool denies every allow outside CWD that is not an explicit session
        read grant; `path_policy` accepted any non-None rule and announced
        ALLOW for the scratch dir.
        """
        self.ctx.plan_mode = True
        self.ctx.plan_run = False
        target = os.path.realpath(os.path.join(m._scratch_root(), "probe.txt"))
        if m._path_is_within(target, CWD):
            self.skipTest("the session scratch dir is inside the project")
        out = self._tool(path=target, op="read")["output"]
        self.assertIn("DENY", out)
        decision, _msg = m._path_acl_decision(self.ctx, "read", target,
                                              tool="read_file", raw_path=target)
        self.assertEqual(decision, "deny")


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

    def test_session_approvals_are_deduplicated_and_bounded(self):
        m._remember_shell_approval("git status")
        m._remember_shell_approval("git status")
        self.assertEqual(len(m._SHELL_SESSION_APPROVED), 1)
        for i in range(m.SHELL_SESSION_APPROVED_MAX + 20):
            m._remember_shell_approval(f"cmd-{i} *")
        self.assertLessEqual(len(m._SHELL_SESSION_APPROVED),
                             m.SHELL_SESSION_APPROVED_MAX)


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

    def test_reset_also_clears_shell_session_approvals(self):
        """A bad `a` answer must have a way out that is not a process restart."""
        m._SHELL_SESSION_APPROVED.clear()
        self.addCleanup(m._SHELL_SESSION_APPROVED.clear)
        m._remember_shell_approval("ssh *")
        _loop()._handle_agentic_acl(['/agentic', 'acl', 'reset'])
        self.assertEqual(m._SHELL_SESSION_APPROVED, [])

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


class TestShellDashCDecomposition(unittest.TestCase):
    """The home guard reaches inside `-c` payloads (M2).

    Pre-fix, `bash -c "rm -rf ~"` was a plain `ask` that `auto_confirm`
    approved while a direct `rm -rf ~` is a hard deny: the guard only read the
    first-level tokens, and the quoted payload was one opaque word.
    """

    def setUp(self):
        self.ctx = FakeCtx()

    def _res(self, command, ctx=None):
        return m.check_shell_approval(command, ctx=ctx or self.ctx)

    def _assert_nested_deny(self, command):
        res = self._res(command)
        self.assertFalse(res["approved"], command)
        self.assertEqual(res["effect"], "deny", command)
        self.assertIn("BLOCKED (rejected)", res.get("message", ""), command)
        self.assertIn("nested `-c`", res.get("message", ""), command)

    def test_direct_rm_still_denied(self):
        res = self._res("rm -rf ~")
        self.assertEqual(res["effect"], "deny")
        self.assertNotIn("nested", res.get("message", ""))

    def test_bash_double_quoted(self):
        self._assert_nested_deny('bash -c "rm -rf ~"')

    def test_sh_single_quoted(self):
        self._assert_nested_deny("sh -c 'rm -rf ~'")

    def test_bundled_flags(self):
        self._assert_nested_deny("bash -lc 'rm -rf ~'")

    def test_home_variable_spelling(self):
        self._assert_nested_deny("bash -xc 'rm -rf $HOME'")

    def test_root_target(self):
        self._assert_nested_deny('dash -c "rm -rf /"')

    def test_wrapper_sudo_with_option_value(self):
        self._assert_nested_deny("sudo -u root bash -c 'rm -rf ~'")

    def test_wrapper_env_assignment(self):
        self._assert_nested_deny("env FOO=1 bash -c 'rm -rf ~'")

    def test_wrapper_busybox(self):
        self._assert_nested_deny("busybox sh -c 'rm -rf ~'")

    def test_python_os_system(self):
        self._assert_nested_deny("python3 -c \"import os; os.system('rm -rf ~')\"")

    def test_python_subprocess_shell(self):
        self._assert_nested_deny(
            "python3 -c \"subprocess.run('rm -rf ~', shell=True)\"")

    def test_python_list_form_is_not_a_shell(self):
        # No shell, no `~` expansion: this deletes a file literally named "~".
        res = self._res("python3 -c \"subprocess.run(['rm', '-rf', '~'])\"")
        self.assertNotEqual(res["effect"], "deny")

    def test_resolved_home_path(self):
        """The payload is normalized like the top-level one (`/home/<user>` → `~`)."""
        home = os.path.expanduser("~")
        self._assert_nested_deny(f"bash -c 'rm -rf {home}'")

    def test_harmless_payloads_are_not_blocked(self):
        for command in ("bash -c 'echo hi'",
                        "python3 -c \"print('hello')\"",
                        "bash -c 'rm -rf /tmp/x'",
                        "bash -c 'rm -rf /home/somebody'",
                        "echo 'bash -c \"rm -rf ~\"'"):
            res = self._res(command)
            self.assertNotIn("nested `-c`", res.get("message") or "", command)

    def test_nested_three_levels(self):
        self._assert_nested_deny("bash -c \"bash -c \\\"bash -c 'rm -rf ~'\\\"\"")

    def test_nested_four_levels_uses_the_text_fallback(self):
        command = ("bash -c \"bash -c \\\"bash -c \\\\\\\"bash -c "
                   "'rm -rf ~'\\\\\\\"\\\"\"")
        self._assert_nested_deny(command)

    def test_auto_confirm_cannot_approve_it(self):
        ctx = FakeCtx()
        ctx.auto_confirm = True
        res = self._res('bash -c "rm -rf ~"', ctx=ctx)
        self.assertEqual(res["effect"], "deny")
        self.assertIn("nested `-c`", res.get("message", ""))

    def test_recursion_is_bounded(self):
        command = "bash -c " * 12 + "'rm -rf ~'"
        res = self._res(command)
        self.assertEqual(res["effect"], "deny")


class TestShellHardGuardPrefixes(unittest.TestCase):
    """The home/root hard guard reaches wrappers, assignments and find/xargs.

    Sonnet 5.5 measured that `sudo rm -rf ~` was only an ask. The effective
    command normalization makes it a deny, like a bare `rm -rf ~`.
    """

    def setUp(self):
        self.ctx = FakeCtx()

    def _res(self, command, auto=False):
        ctx = FakeCtx()
        ctx.auto_confirm = auto
        with patch.object(m.PathAcl, "prompt", return_value=False):
            return m.check_shell_approval(command, ctx=ctx)

    def _assert_deny(self, command, auto=False):
        res = self._res(command, auto=auto)
        self.assertFalse(res["approved"], command)
        self.assertEqual(res["effect"], "deny", command)
        self.assertIn("BLOCKED (rejected)", res.get("message", ""), command)

    def test_wrappers_do_not_hide_rm_home(self):
        for prefix in ("sudo", "env", "nohup", "command", "exec", "time",
                       "nice", "builtin", "busybox"):
            with self.subTest(prefix=prefix):
                self._assert_deny(f"{prefix} rm -rf ~")
                self._assert_deny(f"{prefix} rm -rf /")

    def test_sudo_options_do_not_hide_rm(self):
        for command in (
            "sudo -u root rm -rf ~",
            "sudo -u root -g root rm -rf ~",
            "sudo -- rm -rf ~",
            "sudo -n rm -rf ~",
            "sudo -i rm -rf ~",
            "sudo -A rm -rf ~",
            "sudo --askpass rm -rf ~",
            "sudo --user root rm -rf ~",
            "sudo -uroot rm -rf ~",
        ):
            with self.subTest(command=command):
                self._assert_deny(command)

    def test_assignments_and_absolute_paths_do_not_hide_rm(self):
        for command in (
            "FOO=bar rm -rf ~",
            "A=1 B=2 rm -rf /",
            "FOO+=bar rm -rf ~",
            "A[0]=1 rm -rf /",
            "/bin/rm -rf ~",
            "/usr/bin/rm -rf /",
            '"rm" -rf ~',
            "r\\m -rf ~",
        ):
            with self.subTest(command=command):
                self._assert_deny(command)

    def test_timeout_does_not_hide_rm(self):
        for command in (
            "timeout 10 rm -rf ~",
            "timeout -s KILL 10 rm -rf ~",
            "timeout --signal=KILL 10s rm -rf /",
        ):
            with self.subTest(command=command):
                self._assert_deny(command)

    def test_auto_confirm_cannot_approve_wrapped_home_delete(self):
        self._assert_deny("sudo rm -rf ~", auto=True)
        self._assert_deny("env rm -rf /", auto=True)
        self._assert_deny("find ~ -delete", auto=True)

    def test_wrappers_do_not_hide_nested_shell_c(self):
        for command in (
            "timeout 10 bash -c 'rm -rf ~'",
            "sudo -n bash -c 'rm -rf ~'",
            "env -i bash -c 'rm -rf ~'",
            "busybox env bash -c 'rm -rf ~'",
        ):
            with self.subTest(command=command):
                res = self._res(command)
                self.assertEqual(res["effect"], "deny", command)
                self.assertIn("nested `-c`", res.get("message", ""), command)

    def test_cd_outside_then_delete_is_hard_denied(self):
        for command in (
            "cd / && rm -rf *",
            "cd .. && rm -rf *",
            "cd ~ && rm -rf *",
            "cd $HOME && rm -rf notes.txt",
            "cd ${HOME} && rm -rf *",
            "cd && rm -rf *",
            "cd / && find . -delete",
            "cd ~ && find . -delete",
            "cd / && xargs rm",
            "cd / && bash -c 'rm -rf *'",
            "cd ~ && python3 -c 'print(1)'",
            "cd / && python3 -m http.server",
            "cd / && bash script.sh",
            "cd / && awk 'BEGIN{system(\"id\")}'",
            "cd / && env -S 'rm -rf *'",
            "bash -c 'cd / && rm -rf *'",
            "env -S 'cd / && rm -rf *'",
        ):
            for auto in (False, True):
                with self.subTest(command=command, auto=auto):
                    res = self._res(command, auto=auto)
                    self.assertEqual(res["effect"], "deny", (command, auto))
                    self.assertIn("BLOCKED (rejected)",
                                  res.get("message", ""), (command, auto))

    def test_cd_outside_version_probe_is_not_hard_denied(self):
        for command in ("cd / && python3 --version",
                        "cd / && bash --version",
                        "cd / && perl --version"):
            with self.subTest(command=command):
                verdict = m.shell_check_command(
                    command, m.DEFAULT_SHELL_PERMISSION,
                    m._SHELL_SESSION_APPROVED)
                self.assertFalse(
                    m._shell_is_home_destructive(verdict["nodes"]), command)

    def test_env_split_string_does_not_hide_rm(self):
        for command in (
            "env -S 'rm -rf ~'",
            "env --split-string 'rm -rf ~'",
            "env --split-string='rm -rf ~'",
            "env -S 'rm' -rf ~",
            "env -S'rm' -rf ~",
            "env -S 'bash' -c 'rm -rf ~'",
            "env -S 'bash' '-c' 'rm -rf ~'",
            "env FOO=1 env -S 'rm -rf ~'",
            "sudo env FOO=1 env -S 'rm -rf /'",
            "timeout 10 env -S 'find ~ -delete'",
        ):
            with self.subTest(command=command):
                res = self._res(command)
                self.assertEqual(res["effect"], "deny", command)
                self.assertIn("BLOCKED (rejected)", res.get("message", ""), command)

    def test_wrapped_rm_outside_home_is_not_hard_denied(self):
        for command in ("sudo rm -rf /tmp/x", "env rm -rf ./build",
                        "nice rm -rf build"):
            with self.subTest(command=command):
                res = self._res(command)
                self.assertNotIn("BLOCKED (rejected)", res.get("message") or "",
                                 command)

    def test_effective_command_unit(self):
        cases = {
            "sudo rm -rf ~": ("rm", ["-rf", "~"]),
            "sudo -n rm -rf ~": ("rm", ["-rf", "~"]),
            "sudo -i rm -rf ~": ("rm", ["-rf", "~"]),
            "sudo --user root rm -rf ~": ("rm", ["-rf", "~"]),
            "env FOO=1 bash -c id": ("bash", ["-c", "id"]),
            "timeout 10 rm -rf ~": ("rm", ["-rf", "~"]),
            "timeout -s KILL 10 rm -rf ~": ("rm", ["-rf", "~"]),
            "nice -n 10 rm -rf ~": ("rm", ["-rf", "~"]),
            "stdbuf -o L rm -rf ~": ("rm", ["-rf", "~"]),
            "exec -a foo rm -rf ~": ("rm", ["-rf", "~"]),
            "builtin rm -rf ~": ("rm", ["-rf", "~"]),
            "busybox rm -rf ~": ("rm", ["-rf", "~"]),
            "FOO=bar /bin/rm -rf ~": ("rm", ["-rf", "~"]),
        }
        for command, (head, args) in cases.items():
            with self.subTest(command=command):
                tokens = m._shell_breakdown(command)[0].tokens
                self.assertEqual(m._shell_effective_command(tokens), (head, args))


class TestShellFindDelete(unittest.TestCase):
    """`find -delete` / destructive `find -exec` / `xargs rm` are hard-guarded."""

    def setUp(self):
        self.ctx = FakeCtx()

    def _destructive(self, command):
        verdict = m.shell_check_command(
            command, m.DEFAULT_SHELL_PERMISSION, m._SHELL_SESSION_APPROVED)
        return m._shell_is_home_destructive(verdict["nodes"])

    def test_find_home_and_root_delete_are_destructive(self):
        for command in (
            "find ~ -delete",
            "find ~/ -delete",
            "find / -delete",
            "find /* -delete",
            "find /./ -delete",
            "find /tmp/.. -delete",
            'find "$HOME" -delete',
            "find ${HOME} -delete",
            "find -L ~ -delete",
            "find -D search ~ -delete",
            "find -- ~ -delete",
        ):
            with self.subTest(command=command):
                self.assertTrue(self._destructive(command), command)

    def test_find_exec_destructive_commands(self):
        for command in (
            "find ~ -exec rm -rf {} +",
            "find ~ -execdir rm -rf {} +",
            "find ~ -ok rm -rf {} \\;",
            "find ~ -exec /bin/rm -rf {} +",
            "find ~ -exec sudo rm -rf {} +",
            "find ~ -exec env rm -rf {} +",
            "find ~ -exec timeout 10 rm -rf {} +",
            "find ~ -exec sh -c 'rm -rf ~' \\;",
            "find ~ -exec busybox sh -c 'rm -rf ~' \\;",
            "find ~ -exec python3 -c 'print(1)' \\;",
            "find ~ -exec perl -e 'unlink' \\;",
            "find ~ -exec awk 'BEGIN{system(\"rm -rf ~\")}' \\;",
            "find ~ -exec sed -n '1p' {} \\;",
            "find ~ -exec env -S 'rm -rf ~' \\;",
            "find ~ -exec xargs env -S 'rm -rf ~' \\;",
            "find ~ -exec mv {} /tmp/ \\;",
        ):
            with self.subTest(command=command):
                self.assertTrue(self._destructive(command), command)

    def test_find_read_only_exec_is_not_destructive(self):
        for command in (
            "find ~ -exec grep -l x {} +",
            "find ~ -exec cat {} \\;",
            "find ~ -exec ls -l {} \\;",
        ):
            with self.subTest(command=command):
                self.assertFalse(self._destructive(command), command)

    def test_find_cwd_delete_is_not_hard_denied(self):
        for command in ("find . -delete", "find . -exec rm -rf {} +",
                        "find -name '*~*' -delete"):
            with self.subTest(command=command):
                self.assertFalse(self._destructive(command), command)

    def test_xargs_after_home_root_find_is_destructive(self):
        for command in (
            "find ~ -print0 | xargs -0 rm -rf",
            "find / -print0 | xargs -0 -n 1 rm -rf",
            "find ~ -print0 | xargs -0 -I {} rm -rf {}",
            "find ~ -print0 | xargs -e EOF rm -rf",
            "find ~ -print0 | xargs -0 sh -c 'rm -rf ~'",
            "find ~ -print0 | xargs -0 env rm -rf",
            "find ~ -print0 | xargs -0 env -S 'rm -rf ~'",
        ):
            with self.subTest(command=command):
                self.assertTrue(self._destructive(command), command)

    def test_xargs_without_home_root_find_is_not_hard_denied(self):
        for command in (
            "cat list.txt | xargs rm",
            "find . -print0 | xargs -0 rm -rf",
            "find /tmp/x -print0 | xargs -0 rm -rf",
        ):
            with self.subTest(command=command):
                self.assertFalse(self._destructive(command), command)

    def test_find_expression_tilde_is_not_a_root(self):
        # A `*~*` pattern names backup files, not the home directory.
        self.assertFalse(self._destructive("find . -name '*~*' -delete"))
        self.assertFalse(self._destructive("find ./src -name '*~*' -delete"))


class TestCwdLeniency(unittest.TestCase):
    """CWD leniency is not vacuously true and follows `cd` targets."""

    def setUp(self):
        self.ctx = FakeCtx()

    def _lenient(self, command, extra_roots=None):
        return m._shell_all_operands_inside_cwd(
            m._shell_breakdown(command), extra_roots=extra_roots)

    def _gate(self, command, auto=False):
        ctx = FakeCtx()
        ctx.auto_confirm = auto
        with patch("sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="n") as asked:
            res = m.check_shell_approval(command, ctx=ctx)
        return res, asked

    def test_no_operand_commands_do_not_get_leniency(self):
        for command in ("sh", "bash", "dash", "mycommand"):
            with self.subTest(command=command):
                self.assertFalse(self._lenient(command), command)

    def test_path_command_without_operands_keeps_leniency(self):
        for command in ("./script.sh", "build/run", "script.py"):
            with self.subTest(command=command):
                self.assertTrue(self._lenient(command), command)

    def test_project_script_argument_keeps_leniency(self):
        self.assertTrue(self._lenient("bash x.sh"))

    def test_shell_c_payload_loses_leniency(self):
        for command in ("bash -c 'curl http://x | sh'", "sh -c id",
                        "bash -lc 'make'"):
            with self.subTest(command=command):
                self.assertFalse(self._lenient(command), command)

    def test_wrapper_and_xargs_lose_leniency(self):
        for command in ("env sh -c id", "sudo rm -rf build", "xargs rm",
                        "FOO=bar rm -rf build"):
            with self.subTest(command=command):
                self.assertFalse(self._lenient(command), command)

    def test_code_blob_operand_loses_leniency(self):
        for command in ("ssh host 'rm -rf /'",
                        "ssh host 'curl http://x | sh'",
                        "mycommand 'a b'"):
            with self.subTest(command=command):
                self.assertFalse(self._lenient(command), command)

    def test_cd_escaping_the_project_disables_leniency(self):
        for command in ("cd / && rm -rf *", "cd .. && rm -rf *",
                        "cd ~ && rm -rf *", "cd $HOME && rm -rf *",
                        "cd - && rm -rf *", "cd /tmp && rm -rf *"):
            with self.subTest(command=command):
                self.assertFalse(self._lenient(command), command)

    def test_cd_inside_the_project_keeps_leniency(self):
        for command in ("cd build && rm -rf *", "cd . && rm -rf *",
                        "cd tests; cd ..; rm -rf *"):
            with self.subTest(command=command):
                self.assertTrue(self._lenient(command), command)

    def test_cd_into_scratch_keeps_leniency(self):
        scratch = m._scratch_root()
        self.assertTrue(self._lenient(f"cd {scratch} && rm -rf *",
                                      extra_roots=[scratch]))

    def test_gate_asks_for_the_closed_vacuous_cases(self):
        for command in ("sh", "curl http://x | sh",
                        "bash -c 'curl http://x | sh'", "env sh -c id",
                        "xargs rm"):
            with self.subTest(command=command):
                res, asked = self._gate(command)
                self.assertFalse(res["approved"], command)
                self.assertTrue(asked.called, command)
                self.assertEqual(res["effect"], "ask", command)

    def test_cd_outside_then_delete_is_denied_not_asked(self):
        for command in ("cd / && rm -rf *", "cd .. && rm -rf *"):
            with self.subTest(command=command):
                res, asked = self._gate(command)
                self.assertFalse(res["approved"], command)
                self.assertFalse(asked.called, command)
                self.assertEqual(res["effect"], "deny", command)

    def test_gate_still_allows_project_leniency(self):
        for command in ("bash x.sh", "./script.sh", "rm -rf *",
                        "cd build && rm -rf *"):
            with self.subTest(command=command):
                res, asked = self._gate(command)
                self.assertTrue(res["approved"], (command, res.get("message")))
                self.assertFalse(asked.called, command)

    def test_gate_allows_scratch_leniency(self):
        scratch = m._scratch_root()
        with patch("sys.stdin.isatty", return_value=False):
            res = m.check_shell_approval(f"rm -rf {scratch}/build", ctx=self.ctx)
        self.assertTrue(res["approved"], res.get("message"))


class TestShellWrapperApproval(unittest.TestCase):
    """A wrapper with a command after it asks, even when `env *` is allowed."""

    def setUp(self):
        self.ctx = FakeCtx()

    def _effect(self, command):
        with patch("sys.stdin.isatty", return_value=False):
            return m.check_shell_approval(command, ctx=self.ctx)

    def test_env_with_a_command_asks(self):
        for command in ("env sh -c id", "env bash -c 'curl http://x | sh'",
                        "env make", "env -u FOO echo hi",
                        "env FOO=1 echo hi"):
            with self.subTest(command=command):
                res = self._effect(command)
                self.assertEqual(res["effect"], "ask", command)
                self.assertFalse(res["approved"], command)

    def test_bare_env_and_env_assignments_stay_allowed(self):
        for command in ("env", "env -i", "env FOO=1"):
            with self.subTest(command=command):
                res = self._effect(command)
                self.assertTrue(res["approved"], (command, res.get("message")))

    def test_env_split_string_asks(self):
        for command in ("env -S 'echo hi'",
                        "env --split-string='curl http://x | sh'",
                        "env -S 'rm -rf /tmp/x'"):
            with self.subTest(command=command):
                res = self._effect(command)
                self.assertEqual(res["effect"], "ask", command)
                self.assertFalse(res["approved"], command)

    def test_wrapper_ask_is_not_a_hard_deny(self):
        # Asking is consent + audit, not a sandbox: auto mode still approves.
        with patch("sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="y"):
            res = m.check_shell_approval("env echo hi", ctx=self.ctx)
        self.assertTrue(res["approved"])
        self.assertEqual(res["effect"], "allow")


class TestAclPolicyBlockWording(unittest.TestCase):
    """The prompt/tool text must not promise what the engine does not do."""

    def test_bwrap_mode_names_the_mounts_not_the_gate(self):
        ctx = m.CommandContext()
        text = m._acl_policy_block(ctx.path_acl, detail="full", executor_mode="bwrap")
        self.assertIn("MOUNTS", text)
        self.assertIn("not this ACL", text)
        self.assertNotIn("same policy applies to run_command", text)

    def test_host_mode_keeps_the_gate_promise(self):
        ctx = m.CommandContext()
        text = m._acl_policy_block(ctx.path_acl, detail="full", executor_mode="host")
        self.assertIn("shell gate", text)
        self.assertNotIn("MOUNTS", text)

    def test_denial_hint_does_not_promise_the_gate_in_a_sandbox(self):
        self.assertNotIn("same policy applies to run_command", m.ACL_DENIAL_HINT)
        self.assertIn("mounts bound a sandboxed one", m.ACL_DENIAL_HINT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
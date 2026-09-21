#!/usr/bin/python3
# vim: set ts=4 sw=4 sts=4 et:
#
# Design choices:
#   - Single file: maximally portable (cp/scp and run), zero install, no packaging
#   - CLI-only: no GUI dependencies, works over SSH, headless servers
#   - Tests: tests/test_features.py (138+), tests/test_modifications.py (34+),
#     tests/test_agentic.py (55+) — over 200 unit + end-to-end tests total
#     Run: python3 -m unittest discover tests -v
#   - 3rd-party deps: none required (yaml, html2text, pandoc, lynx optional)
#   - Print/callbacks in business logic: intentional for CLI; would decouple for GUI port
#
# ============================================================================
# ============= IMPORTS & CONFIGURATION ======================================
# ============================================================================
import os
import socket
import sys
import json
import re
import base64
import argparse
import subprocess
import shlex
import threading
import time
import traceback
import glob
import difflib
import unicodedata
from datetime import datetime

from html.parser import HTMLParser
from typing import Optional, Dict, List
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

try:
    import pty
    PTY_AVAILABLE = True
except ImportError:
    PTY_AVAILABLE = False

import atexit


__version__ = "0.2.10"


# ============================================================================
# ============= DEFAULT CONFIGURATION =======================================
# ============================================================================

BUILTIN_PROMPTS = {
    "default": (
        "You are an accurate chatbot replying to a user in their terminal."
        " The user may switch between languages mid-conversation. Mirror their language"
        " — if they write in French, reply in French. Only translate if explicitly asked"
        " (e.g. 'translate this to French')."
        " Format responses with markdown (code blocks, lists, headings) for readability"
        " in a terminal. Only use plain text if the user explicitly requests it."
        " Answer the question accurately. If unsure, say so rather than guessing."
    ),
    "coder": (
        "You are a coding specialist focused on Python and C++, with broad system and"
        " software engineering knowledge."
        " The user may switch between languages mid-conversation. Mirror their language"
        " — if they write in French, reply in French. Only translate if explicitly asked."
        " Format responses with markdown (code blocks, lists, headings) for readability"
        " in a terminal."
        " Use emoji to emphasize section titles — this helps visually organize technical"
        " explanations. If the request is ambiguous, ask clarifying questions rather than"
        " guessing the intent."
    ),
    "sysadmin": (
        "You are a Linux system administrator helping a user manage their system."
        " The user may switch between languages mid-conversation. Mirror their language"
        " — if they write in French, reply in French. Only translate if explicitly asked."
        " Format responses with markdown (code blocks, lists, headings) for readability"
        " in a terminal."
        " Keep answers short and to the point — the user is in a terminal and needs quick"
        " information. If given an image, summarize its contents briefly."
    ),
    "concise": (
        "You are a highly efficient assistant providing factual answers."
        " The user may switch between languages mid-conversation. Mirror their language"
        " — if they write in French, reply in French. Only translate if explicitly asked."
        " Format responses with markdown (code blocks, lists, headings) for readability"
        " in a terminal."
        " Skip pleasantries, filler, and unnecessary explanations. The user wants direct,"
        " factual answers with no extra verbosity."
    ),
    "doctor": (
        "You are a helpful medical information assistant. You provide general health"
        " information but must always clarify that you are NOT a licensed medical professional"
        " and cannot diagnose conditions or prescribe treatments. The user should consult a"
        " real doctor for medical advice."
        " The user may switch between languages mid-conversation. Mirror their language"
        " — if they write in French, reply in French. Only translate if explicitly asked."
        " Format responses with markdown (code blocks, lists, headings) for readability"
        " in a terminal."
        " When discussing symptoms or conditions, explain the reasoning clearly and note"
        " when something requires urgent professional attention."
    ),
    "teacher": (
        "You are a patient and knowledgeable teacher explaining concepts to a learner."
        " Adapt your explanation depth to the user's apparent level — if they ask a basic"
        " question, start simple; if they use technical terms, go deeper."
        " The user may switch between languages mid-conversation. Mirror their language"
        " — if they write in French, reply in French. Only translate if explicitly asked."
        " Format responses with markdown (code blocks, lists, headings) for readability"
        " in a terminal."
        " Use analogies and examples to clarify difficult ideas. If the user seems confused,"
        " offer to rephrase or break it down further."
    ),
    "politic": (
        "You are a neutral political analyst providing factual, balanced information."
        " Present multiple perspectives on political issues fairly, citing sources where"
        " possible. Distinguish clearly between established facts and opinions or theories."
        " The user may switch between languages mid-conversation. Mirror their language"
        " — if they write in French, reply in French. Only translate if explicitly asked."
        " Format responses with markdown (code blocks, lists, headings) for readability"
        " in a terminal."
        " Avoid endorsing any party, candidate, or ideology. If asked for analysis, explain"
        " the reasoning behind different positions rather than advocating for one."
    )
}

DEFAULT_SYSTEM_PROMPT = BUILTIN_PROMPTS["default"]


MAX_CONTEXT_SIZE = 4192000  # 4M tokens maximum limit (prevent OOM)
MAX_READ_FILE_SIZE = 102400    # 100KB max per agentic read_file page
MAX_READ_LINES = 2000          # max lines per agentic read_file page
MAX_READ_LINE_LENGTH = 2000    # per-line truncation in read_file pages
MAX_WRITE_FILE_SIZE = 1048576  # 1MB max for agentic write_file tool
MAX_FILE_INCLUSION_SIZE = 5 * 1024 * 1024  # 5MB max for @file inclusions
DEFAULT_OLLAMA_HOST    = 'http://127.0.0.1:11434'
DEFAULT_LLAMACPP_HOST  = 'http://127.0.0.1:8080'
DEFAULT_LMSTUDIO_HOST  = 'http://127.0.0.1:1234'
DEFAULT_GEMINI_HOST    = 'https://generativelanguage.googleapis.com/v1beta/openai'
DEFAULT_OPENCODEZEN_HOST = 'https://opencode.ai/zen'
DEFAULT_OPENCODEGO_HOST  = 'https://opencode.ai/zen/go'
DEFAULT_MISTRAL_HOST     = 'https://api.mistral.ai'
DEFAULT_DEEPSEEK_HOST    = 'https://api.deepseek.com'
DEFAULT_OLLAMA_PORT    =  11434
DEFAULT_LLAMACPP_PORT  =  8080
DEFAULT_LMSTUDIO_PORT  =  1234

# Cloud backends share the OpenAI-compatible /v1/models shape but need an API key.
CLOUD_BACKENDS = {"gemini", "opencodezen", "opencodego", "mistral", "deepseek"}
CLOUD_API_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "opencodezen": "OPENCODEZEN_API_KEY",
    "opencodego": "OPENCODEGO_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
CLOUD_HOST_ENV = {
    "gemini": "GEMINI_HOST",
    "opencodezen": "OPENCODEZEN_HOST",
    "opencodego": "OPENCODEGO_HOST",
    "mistral": "MISTRAL_HOST",
    "deepseek": "DEEPSEEK_HOST",
}
CLOUD_DEFAULT_HOST = {
    "gemini": DEFAULT_GEMINI_HOST,
    "opencodezen": DEFAULT_OPENCODEZEN_HOST,
    "opencodego": DEFAULT_OPENCODEGO_HOST,
    "mistral": DEFAULT_MISTRAL_HOST,
    "deepseek": DEFAULT_DEEPSEEK_HOST,
}



# ============================================================================
# ============= THEME & COLOR CONFIGURATION ==================================
# ============================================================================

BUILTIN_THEMES = {
    "default": {
        "muted": "\033[90m",
        "info": "\033[36m",
        "success": "\033[32;1m",
        "warning": "\033[33;1m",
        "error": "\033[31;1m",
        "reset": "\033[0m"
    },
    "minimal": {
        "muted": "",
        "info": "",
        "success": "",
        "warning": "",
        "error": "",
        "reset": ""
    },
    "emacs_dark": {
        "muted": "\033[90m",
        "info": "\033[94m",
        "success": "\033[92m",
        "warning": "\033[93m",
        "error": "\033[91m",
        "reset": "\033[0m"
    },
    "vim_dark": {
        "muted": "\033[38;5;245m",
        "info": "\033[38;5;75m",
        "success": "\033[38;5;71m",
        "warning": "\033[38;5;221m",
        "error": "\033[38;5;196m",
        "reset": "\033[0m"
    },
    "high_contrast": {
        "muted": "\033[90m",
        "info": "\033[96m",
        "success": "\033[92;1m",
        "warning": "\033[93;1m",
        "error": "\033[91;1m",
        "reset": "\033[0m"
    }
}

THEME_FILE = os.path.expanduser("~/.ollamaquery/themes.json")

# ============================================================================
# ============= COMMAND REGISTRY =============================================
# ============================================================================

COMMANDS = {
    # === Core Commands ===
    'help': {
        'aliases': ['/?', '/help'],
        'category': 'Core',
        'description': 'Show this help message',
        'usage': '/help',
        'handler': None  # Handled inline in ChatLoop
    },
    'quit': {
        'aliases': ['/quit', '/exit', 'quit', 'exit'],
        'category': 'Core',
        'description': 'Exit the chat session',
        'usage': '/quit',
        'handler': None
    },
    'clear': {
        'aliases': ['/clear'],
        'category': 'Core',
        'description': 'Clear conversation context and attached image',
        'usage': '/clear',
        'handler': None
    },
    'stats': {
        'aliases': ['/stats', '/usage'],
        'category': 'Core',
        'description': 'Show usage statistics',
        'usage': '/stats [reset]',
        'handler': None
    },

    # === Model Management ===
    'listmodel': {
        'aliases': ['/listmodel'],
        'category': 'Model',
        'description': 'List available models',
        'usage': '/listmodel [filter]',
        'handler': None
    },
    'listmodelall': {
        'aliases': ['/listmodelall'],
        'category': 'Model',
        'description': 'List models with capabilities',
        'usage': '/listmodelall',
        'handler': None
    },
    'switchmodel': {
        'aliases': ['/switchmodel'],
        'category': 'Model',
        'description': 'Switch to another model',
        'usage': '/switchmodel <model_name>',
        'handler': None
    },

    # === context management ===
   'dumpcontext': {
        'aliases': ['/dumpcontext'],
        'category': 'I/O',
        'description': 'Dump current conversation context to a file for inspection',
        'usage': '/dumpcontext <filepath>',
        'handler': None
    },
    # === Settings ===
    'contextsizeset': {
        'aliases': ['/contextsizeset'],
        'category': 'Settings',
        'description': 'Set context window size (0 for default)',
        'usage': '/contextsizeset <tokens>',
        'handler': None
    },
    'thinkingon': {
        'aliases': ['/thinkingon'],
        'category': 'Settings',
        'description': 'Enable reasoning/thinking output',
        'usage': '/thinkingon',
        'handler': None
    },
    'thinkingoff': {
        'aliases': ['/thinkingoff'],
        'category': 'Settings',
        'description': 'Disable reasoning/thinking output',
        'usage': '/thinkingoff',
        'handler': None
    },
    'reasoning': {
        'aliases': ['/reasoning'],
        'category': 'Settings',
        'description': 'Set model reasoning effort (low/medium/high) or off/on',
        'usage': '/reasoning [off|on|low|medium|high]',
        'handler': None
    },
    'debug': {
        'aliases': ['/debug'],
        'category': 'Settings',
        'description': 'Configure per-category debug levels',
        'usage': '/debug <category> <level> | /debug list | /debug status',
        'handler': None
    },
    'agentic': {
        'aliases': ['/agentic'],
        'category': 'Settings',
        'description': 'Configure agentic mode and sub-options',
        'usage': '/agentic [on|off|full|auto|sandbox|verbose|thinking|trace|log|acl|iterations|timeout|status]',
        'handler': None
    },
    'listtool': {
        'aliases': ['/listtool'],
        'category': 'Model',
        'description': 'List available agentic tools',
        'usage': '/listtool',
        'handler': None
    },
    # === I/O Operations ===
    'image': {
        'aliases': ['/image'],
        'category': 'I/O',
        'description': 'Attach or clear an image for multimodal models',
        'usage': '/image <path> | /image clear',
        'handler': None
    },
    'curl': {
        'aliases': ['/curl'],
        'category': 'I/O',
        'description': 'Fetch and convert web content to plain text',
        'usage': '/curl <url>',
        'handler': None
    },
    'cwd': {
        'aliases': ['/cwd'],
        'category': 'I/O',
        'description': 'Change working directory',
        'usage': '/cwd [path]',
        'handler': None
    },
    'ls': {
        'aliases': ['/ls'],
        'category': 'I/O',
        'description': 'List directory contents',
        'usage': '/ls [args]',
        'handler': None
    },

    # === Advanced ===
    'spawnshell': {
        'aliases': ['/spawnshell'],
        'category': 'Advanced',
        'description': 'Spawn interactive shell session (exit to return)',
        'usage': '/spawnshell',
        'handler': None
    },
    'compact': {
        'aliases': ['/compact'],
        'category': 'Core',
        'description': 'Compact conversation history to save context space',
        'usage': '/compact [now|force [index]|llm|threshold <v>|list|status]',
        'handler': None
    },
    'drop': {
        'aliases': ['/drop'],
        'category': 'Core',
        'description': 'Drop specific message(s) by index (use /tokencount to see indices)',
        'usage': '/drop <index> [index2 ...]',
        'handler': None
    },
    'tokencount': {
        'aliases': ['/tokencount'],
        'category': 'Core',
        'description': 'Show token breakdown of current conversation',
        'usage': '/tokencount',
        'handler': None
    },
    'sessions': {
        'aliases': ['/sessions'],
        'category': 'Core',
        'description': 'List saved sessions',
        'usage': '/sessions [count]',
        'handler': None
    },
    'resume': {
        'aliases': ['/resume'],
        'category': 'Core',
        'description': 'Resume a previous session',
        'usage': '/resume <session_id>',
        'handler': None
    },
    'save': {
        'aliases': ['/save'],
        'category': 'Core',
        'description': 'Save current session to disk',
        'usage': '/save',
        'handler': None
    },
}

# Category order for help display
COMMAND_CATEGORIES = ['Core', 'Model', 'Settings', 'I/O', 'Advanced']


def get_command_aliases() -> List[str]:
    """Return flat list of all command aliases for readline completion."""
    aliases = []
    for cmd in COMMANDS.values():
        aliases.extend(cmd['aliases'])
    return aliases


def get_commands_by_category(category: Optional[str] = None) -> Dict:
    """Return commands grouped by category, or filtered by category.

    Args:
        category: Optional single category name to filter by.

    Returns:
        When category is given, a dict of commands in that category; otherwise
        a dict mapping category names to sorted (name, info) lists.
    """
    if category:
        return {k: v for k, v in COMMANDS.items() if v['category'] == category}

    # Group by category in defined order
    grouped = {}
    for cat in COMMAND_CATEGORIES:
        cat_cmds = {k: v for k, v in COMMANDS.items() if v['category'] == cat}
        if cat_cmds:
            grouped[cat] = sorted(cat_cmds.items(), key=lambda x: x[0])
    return grouped


def format_help_text(compact: bool = False) -> str:
    """Generate formatted help text for display.

    Args:
        compact: If True, render one-line-per-category alias lists; otherwise
            the detailed /help layout with descriptions and usage.
    """
    lines = []
    grouped = get_commands_by_category()

    for category in COMMAND_CATEGORIES:
        if category not in grouped:
            continue

        if compact:
            # One-liner format for welcome message
            cmds = [info['aliases'][0] for _, info in grouped[category]]
            lines.append(colorize(f"{category}: " + ', '.join(cmds), 'muted'))
        else:
            # Detailed format for /help
            lines.append(f"\n{colorize(category + ':', 'info')}")
            for name, info in grouped[category]:
                aliases = ', '.join(info['aliases'])
                desc = info['description']
                lines.append(f"  {colorize(f'{aliases:<28}', 'success')} {desc}")
                if info.get('usage') and info['usage'] != info['aliases'][0]:
                    lines.append(f"    {colorize('Usage: ' + info['usage'], 'muted')}")

    return '\n'.join(lines)


#
# === Color management
#


def colors_enabled() -> bool:
    """Check if colors should be used (TTY check + NO_COLOR env var)."""
    if os.environ.get('NO_COLOR'):
        return False
    return sys.stdout.isatty()


def load_custom_themes() -> Dict:
    """Load custom themes from JSON file.

    Returns:
        Dict of theme name -> color mapping, or {} if the file is absent/invalid.
    """
    if not os.path.exists(THEME_FILE):
        return {}
    try:
        with open(THEME_FILE, 'r') as f:
            custom = json.load(f)
            if isinstance(custom, dict):
                return custom
    except Exception:
        pass
    return {}


def get_theme(theme_name: Optional[str] = None) -> Dict:
    """Get theme color dictionary.

    Args:
        theme_name: Optional theme key; falls back to OLLAMAQUERY_THEME env or 'default'.
    """
    if os.environ.get('NO_COLOR'):
        return BUILTIN_THEMES["minimal"]

    if theme_name is None:
        theme_name = os.environ.get('OLLAMAQUERY_THEME', 'default')

    custom_themes = load_custom_themes()
    if theme_name in custom_themes:
        theme = custom_themes[theme_name]
        for key in ["muted", "info", "success", "warning", "error", "reset"]:
            if key not in theme:
                theme[key] = BUILTIN_THEMES["default"][key]
        return theme

    return BUILTIN_THEMES.get(theme_name, BUILTIN_THEMES["default"])


def colorize(text: str, role: str, theme: Optional[Dict] = None,
             force_color: bool = False, is_prompt: bool = False) -> str:
    """Apply color to text using active theme.

    Args:
        text: The text to colorize.
        role: Color role key (e.g. 'success', 'warning', 'error').
        theme: Optional theme dict; defaults to get_theme().
        force_color: Apply colors even when the terminal is not a TTY.
        is_prompt: Wrap codes in readline byte markers so they aren't counted.
    """
    if theme is None:
        theme = get_theme()

    if not colors_enabled() and not force_color:
        return text

    start_code = theme.get(role, '')
    reset_code = theme['reset']

    # Wrap color codes in \x01 and \x02 so readline ignores their length
    if is_prompt and READLINE_AVAILABLE:
        start_code = f"\x01{start_code}\x02" if start_code else ""
        reset_code = f"\x01{reset_code}\x02" if reset_code else ""

    return f"{start_code}{text}{reset_code}"


# ============================================================================
# ============= RETRY UTILITY ================================================
# ============================================================================

def _request_with_retry(req: Request, max_retries: int = 3, delay: float = 1,
                        timeout: float = 120, **kwargs: dict) -> object:
    """Open URL with retry on transient network errors.

    Retries on URLError, HTTPError (5xx only), ConnectionError, TimeoutError,
    and OSError. Does NOT retry HTTP 4xx client errors.

    Args:
        req: URL string or Request object to pass to urlopen
        max_retries: Maximum number of attempts (default 3)
        delay: Seconds to wait between retries (default 1)
        timeout: Socket timeout in seconds for urlopen (default 120)
        **kwargs: Additional arguments passed to urlopen (e.g. timeout)

    Returns:
        Same as urlopen(req, **kwargs)

    Raises:
        HTTPError: For 4xx errors or if all retries exhausted
        URLError: If all retries exhausted
    """
    max_retries = max(1, max_retries)
    for attempt in range(1, max_retries + 1):
        try:
            return urlopen(req, timeout=timeout, **kwargs)
        except (URLError, HTTPError, ConnectionResetError, ConnectionError, TimeoutError, OSError) as e:
            if isinstance(e, HTTPError) and e.code < 500:
                raise  # Don't retry 4xx client errors
            if attempt < max_retries:
                sys.stderr.write(
                    colorize(f"\n[RETRY] Request failed ({e}), "
                             f"retrying in {delay}s (next attempt {attempt + 1}/{max_retries})\n",
                             'warning')
                )
                time.sleep(delay)
                continue
            sys.stderr.write(
                colorize(f"\n[RETRY] Request failed ({e}); "
                         f"giving up after {max_retries} attempt(s).\n",
                         'warning')
            )
            raise


# ============================================================================
# ============= CONTEXT WINDOW TRACKING ======================================
# ============================================================================

def _extract_context_size_from_show(data: dict) -> int:
    """Extract context window size from Ollama /api/show response.

    Prefers the model's own architecture-specific context_length key
    (e.g. mistral3.context_length), then falls back to any
    .context_length in model_info, then modelfile/parameters.
    """
    model_info = data.get("model_info", {}) or {}
    arch = model_info.get("general.architecture", "")

    if arch:
        arch_key = f"{arch}.context_length"
        val = model_info.get(arch_key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)

    for key, val in model_info.items():
        if key.endswith(".context_length") and isinstance(val, (int, float)) and val > 0:
            return int(val)

    for source in ("modelfile", "parameters"):
        text = data.get(source, "") or ""
        m = re.search(r'(?:PARAMETER\s+)?num_ctx\s+(\d+)', text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if val > 0:
                return val

    return data.get("context_size", 0)


def _get_ollama_ps_context_size(models: list, model_name: str) -> int:
    """Search /api/ps model list for context_size or context_length."""
    for model in models:
        if model.get("name", "").startswith(model_name):
            for field in ("context_size", "context_length"):
                val = model.get(field, 0)
                if isinstance(val, (int, float)) and val > 0:
                    return int(val)
    return 0


def get_ollama_context_size(base_url: str, model_name: str) -> int:
    """Get the actual context window size from Ollama's running model."""
    try:
        url = f"{base_url}/api/ps"
        with _request_with_retry(Request(url)) as response:
            data = json.loads(response.read().decode('utf-8'))
            size = _get_ollama_ps_context_size(data.get("models", []), model_name)
            if size > 0:
                return size

        url = f"{base_url}/api/show"
        payload = json.dumps({"name": model_name}).encode('utf-8')
        req = Request(url, data=payload,
                     headers={'Content-Type': 'application/json'}, method='POST')
        with _request_with_retry(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return _extract_context_size_from_show(data)
    except Exception:
        sys.stderr.write(colorize(f"[WARNING] Failed to get Ollama context size for '{model_name}'\n", 'warning'))
    return 0


# update the ctx.context_window_size by querying the LLM server
# ctx is an object which contain ctx.context_window_size
def refresh_context_window_size(ctx: 'CommandContext') -> bool:
    """Fetch and update context window size from the backend.

    Args:
        ctx: The shared CommandContext (uses backend/base_url/model).

    Returns:
        True if a positive context size was stored, False otherwise.
    """
    if ctx.backend == "ollama":
        size = get_ollama_context_size(ctx.base_url, ctx.model)
    elif ctx.backend == "lmstudio":
        size = 0
    elif ctx.backend == "gemini":
        size = 1048576  # Gemini 3.5 Flash: 1M token context
    elif ctx.backend in ("opencodezen", "opencodego"):
        size = 131072  # OpenCode models typically have 128K context
    elif ctx.backend == "mistral":
        size = 131072  # Mistral models: 32K-128K context
    elif ctx.backend == "deepseek":
        size = 131072  # DeepSeek: 128K context (deepseek-v4)
    else:
        size = get_llamacpp_context_size(ctx.base_url)

    if size > 0:
        ctx.context_window_size = size
        return True
    return False

def refresh_ollama_context_window_size_from_ps(ctx: 'CommandContext') -> None:
    """Re-check /api/ps to get the real context window size now that the model is loaded.

    Called after a query completes (model is guaranteed to be in /api/ps).
    Only updates if /api/ps returns a positive value (which is the runtime
    context_length, more reliable than /api/show's model_info metadata).
    """
    try:
        url = f"{ctx.base_url}/api/ps"
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with _request_with_retry(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            size = _get_ollama_ps_context_size(data.get("models", []), ctx.model)
            if size > 0:
                ctx.context_window_size = size
    except Exception:
        pass


def context_bar(current: int, maximum: int, width: int = 20) -> str:
    """Render a simple [====    ] NN% bar."""
    if maximum == 0:
        return ""

    pct = min(current / maximum, 1.0)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)

    if pct < 0.6:
        color = 'success'
    elif pct < 0.8:
        color = 'warning'
    else:
        color = 'error'

    return colorize(f"[{bar}] {current}/{maximum} ({pct:.0%})", color)


# ============================================================================
# ============= UTILITY FUNCTIONS ===========================================
# ============================================================================

def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def prepare_image_data(image_path: str) -> Optional[str]:
    """Reads an image file and returns its base64 encoded string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64-encoded image string, or None if the file is missing/unreadable.
    """
    if not image_path or not os.path.isfile(image_path):
        return None

    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(colorize(f"[ERROR] Loading image {image_path}: {e}", 'error'), file=sys.stderr)
        return None


def guess_image_mime(b64_data: str) -> str:
    """Guess image MIME type from base64-encoded data prefix.

    Args:
        b64_data: Base64-encoded image data.

    Returns:
        One of "jpeg"/"png"/"gif"/"webp" (defaults to "jpeg").
    """
    if b64_data.startswith('/9j/'):
        return "jpeg"
    if b64_data.startswith('iVBOR'):
        return "png"
    if b64_data.startswith('R0lGOD'):
        return "gif"
    if b64_data.startswith('UklGR'):
        return "webp"
    return "jpeg"


def fetch_models_ollama(base_url: str) -> list:
    """Fetch available models from Ollama API.

    Args:
        base_url: Ollama server URL.

    Returns:
        List of model dicts, or [] on failure.
    """
    try:
        url = f"{base_url}/api/tags"
        with _request_with_retry(Request(url, headers={'User-Agent': 'Mozilla/5.0'})) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('models', [])
    except Exception:
        return []


def fetch_models_llamacpp(base_url: str, api_key: Optional[str] = None) -> list:
    """Fetch available models from Llama.cpp API (or Gemini OAI-compatible endpoint).

    Args:
        base_url: Server URL.
        api_key: Optional API key for cloud backends.

    Returns:
        List of {"name": ...} dicts, or [] on failure.
    """
    try:
        url = f"{base_url}/v1/models"
        headers = {'User-Agent': 'Mozilla/5.0'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        with _request_with_retry(Request(url, headers=headers)) as response:
            data = json.loads(response.read().decode('utf-8'))
            models = data.get('data', [])
            if not models:
                models = data.get('models', [])
            raw_names = [m.get('id', m.get('name', 'unknown')) for m in models]
            # Strip leading 'models/' prefix from Gemini API
            return [{'name': n[7:] if n.startswith('models/') else n} for n in raw_names]
    except Exception:
        return []

def get_llamacpp_context_size(base_url: str) -> int:
    """Get context size from Llama.cpp /slots endpoint."""
    try:
        url = f"{base_url}/slots"
        with _request_with_retry(Request(url)) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data and isinstance(data, list):
                # Return the context size from the first slot
                return data[0].get('n_ctx', -1)
    except Exception:
        sys.stderr.write(colorize("[WARNING] Failed to get Llama.cpp context size\n", 'warning'))
    return -1

def get_message_token_count_llamacpp(base_url: str, text: str) -> int:
    """Get exact token count using the /tokenize endpoint (no GPU overhead)."""
    global _TOKEN_COUNT_WARNED
    try:
        url = f"{base_url}/tokenize"
        payload = json.dumps({"content": text}).encode('utf-8')
        req = Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with _request_with_retry(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            return len(data.get('tokens', []))
    except Exception as e:
        if not _TOKEN_COUNT_WARNED:
            sys.stderr.write(colorize(f"[WARNING] Token counting failed (llama.cpp): {e}\n", 'warning'))
            _TOKEN_COUNT_WARNED = True
        return estimate_token_count(text)


def get_message_token_count_ollama(base_url: str, text: str, model: str) -> int:
    """Get exact token count using the /api/tokenize endpoint (no GPU overhead).

    Falls back to a heuristic estimate when the endpoint is unavailable. The
    "unsupported" determination (404) is cached so subsequent messages don't
    re-probe the endpoint on every token count.
    """
    global _TOKEN_COUNT_WARNED, _OLLAMA_TOKENIZE_SUPPORTED
    if not model:
        return estimate_token_count(text)
    if _OLLAMA_TOKENIZE_SUPPORTED is False:
        return estimate_token_count(text)
    try:
        url = f"{base_url}/api/tokenize"
        payload = json.dumps({"model": model, "content": text}).encode('utf-8')
        req = Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with _request_with_retry(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            _OLLAMA_TOKENIZE_SUPPORTED = True
            return len(data.get('tokens', []))
    except Exception as e:
        if isinstance(e, HTTPError) and e.code == 404:
            # Stock Ollama has no /api/tokenize — don't re-probe on every message.
            _OLLAMA_TOKENIZE_SUPPORTED = False
            if not _TOKEN_COUNT_WARNED:
                sys.stderr.write(colorize(
                    "[WARNING] Exact token counting unsupported on this Ollama build "
                    "(no /api/tokenize). Using estimates.", 'warning') + "\n")
                _TOKEN_COUNT_WARNED = True
        else:
            if not _TOKEN_COUNT_WARNED:
                sys.stderr.write(colorize(f"[WARNING] Token counting failed (ollama): {e}\n", 'warning'))
                _TOKEN_COUNT_WARNED = True
        return estimate_token_count(text)


def is_available_ollama_model(base_url: str, model_name: str) -> bool:
    """
    Check if a model exists on the Ollama server.

    Args:
        base_url: Ollama server URL (e.g., 'http://192.168.1.20:11434')
        model_name: Model name to check (e.g., 'qwen3:8b')

    Returns:
        True if model exists, False otherwise
    """
    try:
        models = fetch_models_ollama(base_url)
        model_names = [m.get('name', '') for m in models]
        return model_name in model_names
    except Exception:
        return False

def is_available_llamacpp_model(base_url: str, model_name: str, api_key: str = None) -> bool:
    """
    Check if a model exists on the Llama.cpp (or Gemini) server.

    Args:
        base_url: Server URL
        model_name: Model name to check
        api_key: Optional API key for cloud backends

    Returns:
        True if model exists, False otherwise
    """
    try:
        models = fetch_models_llamacpp(base_url, api_key=api_key)
        model_names = [m.get('name', '') for m in models]
        return model_name in model_names
    except Exception:
        return False


def parse_size(size_bytes: Optional[int]) -> str:
    """Parse size from the API into human-readable format.

    Args:
        size_bytes: Numeric size in bytes (or None/falsy).

    Returns:
        Human-readable size string, or "N/A" when unparseable.
    """
    if not size_bytes:
        return "N/A"
    try:
        size_bytes = int(size_bytes)
        if size_bytes > 0:
            if size_bytes >= 1024**3:
                return f"{size_bytes / (1024**3):.1f} GB"
            if size_bytes >= 1024**2:
                return f"{size_bytes / (1024**2):.1f} MB"
            if size_bytes >= 1024:
                return f"{size_bytes / 1024:.1f} KB"
            return f"{size_bytes} B"
    except (TypeError, ValueError):
        pass
    return "N/A"


_TOKEN_COUNT_WARNED = False
# Ollama's stock build does NOT ship /api/tokenize (never merged upstream), so
# probing it on every message wastes an HTTP round-trip and a 404. Cache the
# determination: None=unknown, True=supported, False=unsupported (use estimates).
_OLLAMA_TOKENIZE_SUPPORTED = None


def estimate_token_count(text: str) -> int:
    """Estimate token count from text using regex-based heuristic.

    Falls back to a safe overestimate when API tokenization is unavailable.
    Detects code content and uses a higher multiplier for safety.
    """
    if not text:
        return 0
    tokens = len(re.findall(r'\b\w+\b|[^\w\s]', text))
    code_keywords = {'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'return', 'for', 'while', 'try', 'except', 'with', 'as', 'pass', 'raise', 'lambda', 'yield', 'async', 'await'}
    words = set(re.findall(r'\b\w+\b', text.lower()))
    if words.intersection(code_keywords):
        return max(1, int(tokens * 2.0))
    return max(1, int(tokens * 1.5))


def fetch_model_info_ollama(base_url: str, model_name: str) -> Dict:
    """Fetch detailed model information via Ollama /api/show endpoint.

    Args:
        base_url: Ollama server URL.
        model_name: Name of the model to inspect.

    Returns:
        Raw /api/show JSON dict, or {} on failure.
    """
    try:
        url = f"{base_url}/api/show"
        payload = json.dumps({"name": model_name}).encode('utf-8')
        req = Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with _request_with_retry(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return {}


class CommandContext:
    """Singleton that holds all shared state for the application.

    Replaces the scattered self.* attributes across ChatLoop, ModelQuery,
    and other classes with a single centralized state object.

    Design rationale: factory methods (create_completer, create_tool_registry,
    etc.) live here as thin convenience wrappers rather than in separate factory
    classes because the codebase is a single file and the indirection would add
    no benefit. Token counting heuristic (calculate_context_tokens) lives here
    because it is context-state-dependent, not a pure text utility.
    All print() calls in business logic are intentional for CLI mode;
    a GUI port would inject a callback/emitter pattern instead.

    Call map:
      create_completer() → ChatCompleter
      create_query_handler() → ModelQuery
      create_executor() → Executor
      create_tool_registry() → ToolRegistry
      update_stats / get_cumulative_stats / estimate_tokens / calculate_context_tokens
    """
    _instance = None
    _initialized = False

    def __new__(cls) -> 'CommandContext':
        """Return the singleton instance, creating it on first access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the singleton's shared state once (no-op on repeat)."""
        if CommandContext._initialized:
            return
        CommandContext._initialized = True
        self.shell_timeout: int = 5
        self.debug_manager = DebugManager()
        # Connection info
        self._base_url: str = ""
        self._backend: str = "ollama"
        self._model: str = "llama3"
        self.api_key: str = ""

        # Session state
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.context_size: Optional[int] = None
        self.current_images: List[str] = []
        self.force_no_thinking: bool = False
        self.reasoning_effort: Optional[str] = None  # None | "low" | "medium" | "high"
        self.models: List[str] = []

        # Execution state
        self.debug_mode: bool = False
        self.stream_enabled: bool = True

        # Statistics (cumulative)
        self.total_queries: int = 0
        self.total_tokens_generated: int = 0
        self.total_prompt_tokens: int = 0
        self.total_time_spent: float = 0.0
        self.total_chars_generated: int = 0
        self.query_history: list = []
        self.max_history: int = 50
        # Agentic mode
        self.agentic_mode: bool = False
        self.auto_confirm: bool = False
        self.agentic_verbose: bool = False
        self.agentic_show_thinking: bool = False
        self.agentic_trace: bool = False
        self.agentic_logging: bool = True
        self.agentic_max_iterations: int = 50
        self.agentic_step_timeout: int = 300
        self.agentic_timeout_max: int = 600
        self.agentic_progress_grace: int = 15
        self.agentic_max_thinking_tokens: int = 2048
        self.agentic_heartbeat_tokens: int = 10
        self.agentic_compact_threshold: float = 0.85  # Agentic auto-compact trigger (later than compaction_threshold to keep working memory)
        self.agentic_consecutive_timeouts: int = 0
        self.agentic_has_executed_tool: bool = False
        self.agentic_last_tool_name: str = ""
        self.lazy_tool: bool = True  # Enabled by default: many models embed tool calls after thinking/preamble
        self.supports_vision: Optional[bool] = None  # None = unknown (treated as capable)

        # Context window tracking
        self.context_window_size: int = 0  # Will be fetched from server
        self.current_context_tokens: int = 0  # Updated after each query
        self.compaction_threshold: float = COMPACTION_THRESHOLD  # Auto-compact trigger (0.0-1.0)

        # Session-scoped path access policy (see /agentic acl)
        self.path_acl: PathAcl = PathAcl()

    # === Properties ===
    @property
    def base_url(self) -> str:
        """The resolved backend API base URL."""
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        """Set the backend API base URL."""
        self._base_url = value

    @property
    def backend(self) -> str:
        """The active backend name (ollama/llamacpp/lmstudio/cloud)."""
        return self._backend

    @backend.setter
    def backend(self, value: str) -> None:
        """Set the active backend name."""
        self._backend = value

    @property
    def model(self) -> str:
        """The currently selected model name."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the currently selected model name."""
        self._model = value

    # === Helper Methods ===
    def reset(self) -> None:
        """Reset session state without changing connection info or user preferences."""
        self.current_images = []
        self.total_queries = 0
        self.total_tokens_generated = 0
        self.total_prompt_tokens = 0
        self.total_time_spent = 0.0
        self.total_chars_generated = 0
        self.query_history = []
        self.context_window_size = 0
        self.current_context_tokens = 0
        self.supports_vision = None

    def create_completer(self) -> 'ChatCompleter':
        """Create a ChatCompleter using this context's connection info."""
        return ChatCompleter(self.base_url, self.backend, api_key=self.api_key)

    def create_query_handler(self) -> 'ModelQuery':
        """Create a ModelQuery using this context's connection info."""
        return ModelQuery(self.base_url, self.backend)

    def create_executor(self) -> 'Executor':
        """Create an Executor for agentic tool execution."""
        return Executor()

    def create_tool_registry(self) -> 'ToolRegistry':
        """Create a ToolRegistry for agentic mode."""
        executor = self.create_executor()
        return ToolRegistry(ctx=self, executor=executor)

    def update_stats(self, tokens: int, prompt_tokens: int, time_spent: float, chars: int) -> None:
        """Update cumulative statistics."""
        self.total_queries += 1
        self.total_tokens_generated += tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_time_spent += time_spent
        self.total_chars_generated += chars

        # Rolling history
        entry = {
            "timestamp": time.time(),
            "tokens": tokens,
            "prompt_tokens": prompt_tokens,
            "time": time_spent,
            "tps": tokens / time_spent if time_spent > 0 else 0.0
        }
        self.query_history.append(entry)
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)

    def get_cumulative_stats(self) -> Dict:
        """Return summary of all tracked usage."""
        return {
            "total_queries": self.total_queries,
            "total_completion_tokens": self.total_tokens_generated,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_tokens": self.total_tokens_generated + self.total_prompt_tokens,
            "total_time_seconds": self.total_time_spent,
            "avg_tps": self.total_tokens_generated / self.total_time_spent if self.total_time_spent > 0 else 0.0,
            "avg_tokens_per_query": self.total_tokens_generated / self.total_queries if self.total_queries > 0 else 0.0
        }

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text. Delegates to standalone estimator."""
        return estimate_token_count(text)

    def tokenize(self, text: str) -> int:
        """Get exact token count via the backend's tokenize endpoint.

        Falls back to heuristic estimation if the endpoint is unavailable.
        Result should be cached by the caller (e.g., in msg['_tokens']).
        """
        if not text:
            return 0
        if self.backend == "llamacpp":
            return get_message_token_count_llamacpp(self.base_url, text)
        elif self.backend == "ollama":
            return get_message_token_count_ollama(self.base_url, text, self.model)
        return self.estimate_tokens(text)

    def stamp_tokens(self, msg: dict) -> dict:
        """Compute and cache the token count for a message dict.

        Stores the count in msg['_tokens'] so it only needs to be computed once.
        Returns the same dict (mutated in place) for convenience.
        """
        if '_tokens' in msg:
            return msg
        content = msg.get("content", "")
        if isinstance(content, list):
            extracted = ""
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        extracted += part.get("text", "")
            content = extracted
        tokens = self.tokenize(content)
        if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                tc_text = f"{tc.get('id', '')} {fn.get('name', '')} {fn.get('arguments', '')}"
                tokens += self.tokenize(tc_text)
        if isinstance(msg.get("images"), list):
            tokens += len(msg["images"]) * 1024
        tokens += 2  # role overhead
        msg['_tokens'] = tokens
        return msg

    def calculate_context_tokens(self, messages: list) -> int:
        """Calculate total tokens in conversation context.

        Uses cached _tokens when available (exact), falls back to estimation.
        """
        total = 0
        for msg in messages:
            if '_tokens' in msg:
                total += msg['_tokens']
            else:
                content = msg.get("content", "")
                image_count = 0
                if isinstance(content, list):
                    extracted = ""
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                extracted += part.get("text", "")
                            elif part.get("type") == "image_url":
                                image_count += 1
                    content = extracted
                if isinstance(msg.get("images"), list):
                    image_count += len(msg["images"])
                total += self.estimate_tokens(content)
                if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        tc_text = f"{tc.get('id', '')} {fn.get('name', '')} {fn.get('arguments', '')}"
                        total += self.estimate_tokens(tc_text)
                total += image_count * 1024
                total += 2
        return total


# ============================================================================
# ============= CONTEXT COMPACTION  ===========================================
# ============================================================================

COMPACTION_THRESHOLD = 0.75  # Trigger auto-compaction at 75% of context window
COMPACTION_TARGET = 0.50     # Compact down to 50% of context window
COMPACTION_KEEP_RECENT = 6   # Always keep last 6 messages (3 user/assistant turns)


def sanitize_tool_pairing(messages: list) -> list:
    """Repair orphaned tool-role messages so strict chat templates don't 500.

    gpt-oss (and similar) chat templates hard-raise when a 'tool' role message
    is not preceded by an assistant message carrying tool_calls. History
    compaction / merging can split such pairs, leaving an orphaned tool result
    that makes every subsequent request fail with a Jinja exception. This
    walker demotes any orphaned tool message to a 'user' note, preserving its
    content so no information is lost.

    Consecutive tool messages after a single assistant-with-tool_calls are
    valid (parallel results) and are left untouched. Only a tool message whose
    most recent non-tool predecessor was a user or a plain assistant is
    repaired.

    Mutates messages in place and returns the same list.
    """
    last_non_tool_ok = False  # most recent non-tool msg was assistant with tool_calls
    for msg in messages:
        role = msg.get('role')
        if role == 'tool':
            if not last_non_tool_ok:
                name = msg.get('name', '')
                content = msg.get('content', '')
                if isinstance(content, list):
                    content = " ".join(
                        p.get('text', '') for p in content if isinstance(p, dict)
                    )
                prefix = f"Tool result ({name}):" if name else "Tool result:"
                msg['role'] = 'user'
                msg['content'] = f"{prefix}\n{content}"
                msg.pop('tool_call_id', None)
        else:
            last_non_tool_ok = (
                role == 'assistant'
                and bool(msg.get('tool_calls'))
            )
    return messages


def _split_compaction_window(messages: list, keep_recent: int) -> tuple:
    """Split messages into (system, middle, recent) for compaction.

    The keep-recent window is shifted left so it never *starts* on a tool
    result whose pairing assistant tool_call lives in the dropped middle.
    Strict chat templates (gpt-oss) raise on such an orphaned tool message.

    Returns:
        (system_msg, middle, recent). `middle` may be empty when the whole
        conversation fits inside the keep-recent window.
    """
    system_msg = messages[0]
    start = max(1, len(messages) - keep_recent)
    while start > 1 and messages[start].get('role') == 'tool':
        start -= 1
    return system_msg, messages[1:start], messages[start:]


def _build_compaction_summary(middle: list) -> str:
    """Render the mechanical excerpt-based summary of the dropped middle turns.

    Tool results collapse to a status line; user/assistant messages keep a short
    preview. This is the Phase-1 fallback when no LLM summary is available.
    """
    summary_parts = []
    for msg in middle:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if role == 'tool':
            name = msg.get('name', 'unknown')
            success = 'OK' if 'ERROR' not in content[:100] else 'FAILED'
            summary_parts.append(f"[Tool {name}: {success}]")
        elif role == 'assistant':
            preview = content[:200].replace('\n', ' ')
            if len(content) > 200:
                preview += '...'
            summary_parts.append(f"Assistant: {preview}")
        elif role == 'user':
            if content.startswith("Tool result:") or content.startswith("<tool_result"):
                tool_lines = content.split('\n')[:2]
                summary_parts.append(f"[Tool observation: {tool_lines[0][:80]}]")
            else:
                preview = content[:150].replace('\n', ' ')
                if len(content) > 150:
                    preview += '...'
                summary_parts.append(f"User: {preview}")

    return (
        "[CONVERSATION HISTORY COMPACTED]\n"
        "The following is a condensed summary of earlier conversation turns:\n"
        + "\n".join(summary_parts)
    )


def compact_messages(messages: list, ctx: 'CommandContext', target_tokens: int = 0,
                     keep_recent: int = COMPACTION_KEEP_RECENT, force: bool = False) -> list:
    """Compact conversation history to fit within token budget.

    Strategy:
    1. Always keep messages[0] (system prompt)
    2. Always keep the last `keep_recent` messages
    3. Replace everything in between with a condensed summary message
    4. Tool result messages are summarized most aggressively (name + status only)

    Args:
        messages: Full message list (not mutated — returns a new list)
        ctx: CommandContext for token estimation
        target_tokens: Target token count after compaction (0 = use COMPACTION_TARGET)
        keep_recent: Number of recent messages to always preserve
        force: If True, compact even when already under the token budget

    Returns:
        A new compacted message list
    """
    if len(messages) <= keep_recent + 1:
        return list(messages)

    if target_tokens <= 0 and ctx.context_window_size > 0:
        target_tokens = int(ctx.context_window_size * COMPACTION_TARGET)

    current_tokens = ctx.calculate_context_tokens(messages)
    if target_tokens > 0 and current_tokens <= target_tokens and not force:
        return list(messages)

    system_msg, middle, recent = _split_compaction_window(messages, keep_recent)

    if not middle:
        return list(messages)

    summary_text = _build_compaction_summary(middle)

    summary_tokens = ctx.estimate_tokens(summary_text)
    system_tokens = system_msg.get('_tokens', ctx.estimate_tokens(system_msg.get('content', '')))
    recent_tokens = sum(m.get('_tokens', ctx.estimate_tokens(m.get('content', ''))) for m in recent)
    available = target_tokens - system_tokens - recent_tokens - 100

    if available > 0 and summary_tokens > available:
        ratio = available / summary_tokens
        max_chars = int(len(summary_text) * ratio)
        summary_text = summary_text[:max_chars] + "\n[... older history truncated ...]"

    compacted_msg = {'role': 'user', 'content': summary_text}
    ctx.stamp_tokens(compacted_msg)
    result = [system_msg, compacted_msg]
    if recent and recent[-1].get('role') == 'user':
        result.append({'role': 'assistant', 'content': '[Context continued...]'})
    result.extend(recent)
    sanitize_tool_pairing(result)
    return result


# LLM-assisted tool-result summarization (qwen36 improvement A).
SUMMARIZE_TOOL_MIN_CHARS = 1000  # Only summarize tool results larger than this
SUMMARIZE_TOOL_MAX = 3           # Cap tool results summarized per invocation
SUMMARIZE_TOOL_EXCLUDE = {"list_directory", "diff", "patch", "edit_file", "apply_patch"}


def _extract_sync_content(ctx: 'CommandContext', response: dict) -> str:
    """Extract assistant text content from a sync response (backend-aware).

    Handles ollama's `message.content`, OpenAI-compatible `choices[0].message
    .content`, and plain-string responses. Multi-modal content lists are joined
    from their `text` parts.

    Args:
        ctx: CommandContext (uses backend for shape selection).
        response: Dict or str returned by a query call.

    Returns:
        Extracted text content (never None).
    """
    content = ""
    if isinstance(response, dict):
        if ctx.backend == "ollama":
            content = response.get('message', {}).get('content', '')
        else:
            choices = response.get('choices', [])
            if choices:
                content = choices[0].get('message', {}).get('content', '')
    elif isinstance(response, str):
        content = response
    if isinstance(content, list):
        content = "".join(p.get('text', '') for p in content if isinstance(p, dict))
    return content or ""


def _extract_sync_tool_calls(ctx: 'CommandContext', response: dict) -> list:
    """Extract native API tool calls from a sync response (backend-aware).

    Returns the raw `message.tool_calls` (ollama) or `choices[0].message
    .tool_calls` (OpenAI-compatible) list, or an empty list when absent.

    Args:
        ctx: CommandContext (uses backend for shape selection).
        response: Dict or str returned by a query call.
    """
    if not isinstance(response, dict):
        return []
    if ctx.backend == "ollama":
        return response.get('message', {}).get('tool_calls', []) or []
    choices = response.get('choices', [])
    if choices:
        return choices[0].get('message', {}).get('tool_calls', []) or []
    return []


def _signal_abort(step_cancel: dict) -> None:
    """Abort a running agentic step: set the cancel event and close its HTTP response.

    Args:
        step_cancel: The {"event": threading.Event, "close": callable} dict
            threaded into `query_sync_stream` as its `cancel` kwarg.
    """
    step_cancel["event"].set()
    close = step_cancel.get("close")
    if close:
        try:
            close()
        except Exception:
            pass


def _llm_summarize_tool_result(ctx: 'CommandContext', query_handler: 'ModelQuery',
                               name: str, content: str) -> Optional[str]:
    """Ask the model to summarize a single tool result. Returns summary text or None.

    Args:
        ctx: CommandContext (uses model for the call).
        query_handler: ModelQuery for the summarization call.
        name: Tool name that produced the result.
        content: Raw tool result content.

    Returns:
        Summary text, or None on failure/empty response.
    """
    system = (
        "You are a context optimizer. Summarize the tool result below, keeping only "
        "information relevant to the ongoing task: key findings, file paths, errors, "
        "and decisions. Drop verbose output. Be concise — a few sentences at most."
    )
    user = f"[{name}]\n{content[:4000]}"
    try:
        response = query_handler.query_sync(
            [{'role': 'system', 'content': system},
             {'role': 'user', 'content': user}],
            ctx.model,
            temperature=0.3,
        )
    except Exception:
        return None
    summary = _extract_sync_content(ctx, response).strip()
    return summary or None


def summarize_tool_results(messages: list, ctx: 'CommandContext',
                           query_handler: 'ModelQuery') -> list:
    """Summarize large tool results via the LLM to preserve key findings.

    Replaces the content of eligible tool-result messages (in place) with a
    model-generated summary, so compaction doesn't blindly drop their content.
    Only the oldest half of tool results are considered (newest stay raw), and
    only results larger than SUMMARIZE_TOOL_MIN_CHARS whose tool is not in
    SUMMARIZE_TOOL_EXCLUDE are summarized.

    Args:
        messages: Conversation list (mutated in place).
        ctx: CommandContext for token stamping.
        query_handler: ModelQuery for the summarization call.

    Returns:
        messages (tool result contents replaced in place).
    """
    tool_idxs = [i for i, m in enumerate(messages) if m.get('role') == 'tool']
    if not tool_idxs:
        return messages

    eligible = tool_idxs[:len(tool_idxs) // 2] if len(tool_idxs) > 1 else []

    candidates = []
    for i in eligible:
        m = messages[i]
        name = m.get('name', '')
        content = m.get('content', '')
        if name in SUMMARIZE_TOOL_EXCLUDE:
            continue
        if not isinstance(content, str) or len(content) <= SUMMARIZE_TOOL_MIN_CHARS:
            continue
        candidates.append(i)

    candidates = candidates[:SUMMARIZE_TOOL_MAX]

    for i in candidates:
        m = messages[i]
        name = m.get('name', '')
        content = m.get('content', '')
        summary = _llm_summarize_tool_result(ctx, query_handler, name, content)
        if not summary:
            continue
        m['content'] = f"[Summarized tool result ({name})]\n{summary}"
        m.pop('_tokens', None)
        ctx.stamp_tokens(m)

    return messages


# LLM-assisted conversation summarization (opus6 item 2 / `/compact llm`).
LLM_COMPACT_TRANSCRIPT_CHARS = 6000    # Cap the excerpt sent to the model
LLM_COMPACT_BREAKDOWN_CHARS = 4000    # Cap the token breakdown kept in the summary message
LLM_COMPACT_BREAKDOWN_TEXT = 80       # Per-line text excerpt in the breakdown
LLM_COMPACT_SUMMARY_MARKER = "[CONVERSATION SUMMARY (LLM)]"


def _cap_line(text: str, max_len: int) -> str:
    """Strip and truncate a single line to at most `max_len` chars."""
    text = text.strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _render_token_breakdown(middle: list, ctx: 'CommandContext',
                            max_chars: int = LLM_COMPACT_BREAKDOWN_CHARS) -> str:
    """Render a `/tokencount`-style breakdown of the messages being compacted.

    Every line carries the role, token count (· exact / ~ estimated), an [OK] or
    [KO] marker for tool results, and ~`LLM_COMPACT_BREAKDOWN_TEXT` chars of the
    text — enough for a model reading the compacted summary to reconstruct what
    happened step by step (queries, attempts, tool calls and their outcomes).
    Tool-call JSON in assistant lines is long, so the excerpt budget (~80 chars)
    is sized to expose at least the tool name and the start of its arguments.

    Args:
        middle: The older message list being compacted.
        ctx: CommandContext (token estimation for unstamped messages).
        max_chars: Total output cap.

    Returns:
        A formatted breakdown string.
    """
    lines = []
    for i, msg in enumerate(middle):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if not isinstance(content, str):
            content = str(content)
        if '_tokens' in msg:
            tokens = msg['_tokens']
            marker = "·"
        else:
            tokens = ctx.estimate_tokens(content) + 2
            marker = "~"
        if role == 'tool':
            ok = 'OK' if 'ERROR' not in content[:100] else 'KO'
            excerpt = _cap_line(content.split('\n')[0] if content else '', LLM_COMPACT_BREAKDOWN_TEXT)
            text = f"[{ok}] {excerpt}"
        else:
            text = _cap_line(content.replace('\n', ' '), LLM_COMPACT_BREAKDOWN_TEXT)
        lines.append(f"  [{i:3d}] {role:10s} {marker}{tokens:6d} tok  {text}")
    breakdown = "\n".join(lines)
    if len(breakdown) > max_chars:
        breakdown = breakdown[:max_chars] + "\n[...]"
    return breakdown


def _extract_gen_tokens(ctx: 'CommandContext', response: object) -> int:
    """Extract the number of generated (completion) tokens from a sync response.

    Args:
        ctx: CommandContext (backend selects the shape).
        response: The query_sync response dict.

    Returns:
        Generated token count, or 0 when unavailable.
    """
    if not isinstance(response, dict):
        return 0
    if ctx.backend == "ollama":
        return int(response.get("eval_count", 0) or 0)
    usage = response.get("usage") or {}
    return int(usage.get("completion_tokens", 0) or 0)


def _render_conversation_transcript(middle: list,
                                    max_chars: int = LLM_COMPACT_TRANSCRIPT_CHARS) -> str:
    """Render a compact text transcript of older turns for the LLM summarizer.

    Tool results reduce to a status line plus a short outcome excerpt (they are
    otherwise handled separately by `summarize_tool_results`); user/assistant
    text is kept near-verbatim so the model can follow the actual conversation.
    """
    lines = []
    for msg in middle:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if role == 'tool':
            name = msg.get('name', 'tool')
            ok = 'OK' if 'ERROR' not in content[:100] else 'FAILED'
            excerpt = content.split('\n')[0] if isinstance(content, str) and content else ''
            if excerpt:
                excerpt = _cap_line(excerpt, 100)
                lines.append(f"[tool {name}: {ok}] {excerpt}")
            else:
                lines.append(f"[tool {name}: {ok}]")
            continue
        if not isinstance(content, str):
            content = str(content)
        if role == 'user' and (content.startswith("Tool result:") or content.startswith("<tool_result")):
            first = content.split('\n')[0][:80]
            lines.append(f"[tool observation: {first}]")
            continue
        text = content.replace('\n', ' ')
        if len(text) > 300:
            text = text[:300] + "..."
        lines.append(f"{role}: {text}")
    transcript = "\n".join(lines)
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n[...]"
    return transcript


def _llm_summarize_conversation(ctx: 'CommandContext', query_handler: 'ModelQuery',
                                middle: list) -> tuple:
    """Ask the model for a faithful narrative summary of older conversation turns.

    Args:
        ctx: CommandContext (uses model for the call).
        query_handler: ModelQuery for the summarization call.
        middle: Older message list to summarize.

    Returns:
        (summary, gen_tokens, elapsed_sec): the summary text (or None on
        failure/empty response), the number of generated tokens, and the wall
        time of the summarization call.
    """
    transcript = _render_conversation_transcript(middle)
    if not transcript.strip():
        return None, 0, 0.0
    system = (
        "You are a context optimizer for a CLI chat session. The conversation is "
        "approaching the context limit, so the EARLIER turns below must be compacted "
        "into a short, faithful summary. Retain everything that still matters for the "
        "rest of the session: the user's requirements and preferences, decisions made, "
        "file paths, commands run, errors, and open questions. A few sentences are "
        "enough. Do not narrate the process — only keep information that would be lost "
        "if the transcript were deleted."
    )
    user = f"EARLIER CONVERSATION TO SUMMARIZE:\n\n{transcript}"
    start = time.time()
    try:
        response = query_handler.query_sync(
            [{'role': 'system', 'content': system},
             {'role': 'user', 'content': user}],
            ctx.model,
            temperature=0.3,
        )
    except Exception:
        return None, 0, 0.0
    elapsed = time.time() - start
    summary = _extract_sync_content(ctx, response).strip()
    return (summary or None), _extract_gen_tokens(ctx, response), elapsed


def _extract_compaction_log(content: object) -> str:
    """Return the body of a prior compaction summary message for re-embedding.

    A previous `llm_compact_messages` result starts with the summary marker and
    an intro line; the dated entries after it are extracted so they can be kept
    verbatim when a new compaction entry is appended — the log grows by roughly
    one entry per `/compact llm`, but earlier entries are never re-summarized
    away, so the session model keeps its full memory of what was achieved.

    Args:
        content: The prior compacted message content (or any non-string).

    Returns:
        The entries block without the marker/intro, or '' when `content` is
        not a prior compaction summary.
    """
    if not isinstance(content, str) or not content.startswith(LLM_COMPACT_SUMMARY_MARKER):
        return ''
    lines = content.split('\n')
    rest = lines[1:]
    if rest and (rest[0].startswith('Earlier conversation compacted')
                 or rest[0].startswith('Compaction log')):
        rest = rest[1:]
    while rest and not rest[0].strip():
        rest = rest[1:]
    while rest and not rest[-1].strip():
        rest.pop()
    return '\n'.join(rest)


def llm_compact_messages(messages: list, ctx: 'CommandContext', query_handler: 'ModelQuery',
                         keep_recent: int = COMPACTION_KEEP_RECENT) -> list:
    """Compress history with an LLM-generated narrative summary of the older turns.

    Unlike `compact_messages` (excerpt previews), the dropped middle is handed to
    the model for a faithful summary, preserving facts the excerpt-based approach
    would drop. The compacted message embeds the narrative summary followed by
    the `/tokencount`-style breakdown of the compacted turns (roles, token
    counts, OK/KO tool results, ~50 chars of text) plus the before/after context
    size and the summarization token rate, so the session model can reconstruct
    what was achieved. When a prior `/compact llm` summary message exists, its
    dated entries are kept verbatim and the new entry is appended — the summary
    becomes a dated log that grows by ~one entry per compaction. On LLM failure
    (or when there is nothing to compact) falls back to the mechanical
    `compact_messages`.

    Args:
        messages: Full message list (not mutated — returns a new list).
        ctx: CommandContext for token estimation/stamping.
        query_handler: ModelQuery used for the summarization call.
        keep_recent: Number of recent messages to always preserve.

    Returns:
        A new compacted message list.
    """
    if len(messages) <= keep_recent + 1:
        return list(messages)

    system_msg, middle, recent = _split_compaction_window(messages, keep_recent)
    if not middle:
        return list(messages)

    before_tokens = ctx.calculate_context_tokens(messages)
    summary, gen_tokens, elapsed = _llm_summarize_conversation(ctx, query_handler, middle)
    if not summary:
        return compact_messages(messages, ctx, keep_recent=keep_recent)

    # Append to the dated compaction log if a prior summary message exists, so
    # earlier entries survive verbatim instead of being re-summarized away.
    prev_log = _extract_compaction_log(middle[0].get('content', '')) if middle else ''
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    entry = (
        f"[{timestamp}]\n"
        + summary
        + "\n\nToken breakdown of the compacted turns:\n"
        + _render_token_breakdown(middle, ctx)
    )
    body = prev_log + "\n\n" + entry if prev_log else entry
    base_text = (
        LLM_COMPACT_SUMMARY_MARKER + "\n"
        "Compaction log (oldest first):\n\n"
        + body
    )
    compacted_msg = {'role': 'user', 'content': base_text}
    ctx.stamp_tokens(compacted_msg)
    result = [system_msg, compacted_msg]
    if recent and recent[-1].get('role') == 'user':
        result.append({'role': 'assistant', 'content': '[Context continued...]'})
    result.extend(recent)
    sanitize_tool_pairing(result)
    after_tokens = ctx.calculate_context_tokens(result)

    saved = before_tokens - after_tokens
    pct = (saved / before_tokens * 100) if before_tokens else 0.0
    speed = (gen_tokens / elapsed) if elapsed > 0 else 0.0
    compacted_msg['content'] = (
        base_text
        + f"\n\nContext: {before_tokens} → {after_tokens} tokens ({pct:.0f}% saved)"
        + f" | summary generation: {speed:.1f} tok/s ({gen_tokens} tok in {elapsed:.1f}s)"
    )
    ctx.stamp_tokens(compacted_msg)
    return result


# ============================================================================
# ============= SESSION PERSISTENCE  ==========================================
# ============================================================================

SESSION_DIR = os.path.expanduser("~/.ollamaquery.d/sessions")
SESSION_INDEX = os.path.join(SESSION_DIR, "index.json")
MAX_SESSIONS = 50


class SessionManager:
    """Manages session persistence: save, list, resume."""

    def __init__(self) -> None:
        """Ensure the session directory exists and load the session index."""
        os.makedirs(SESSION_DIR, exist_ok=True)
        self._index = self._load_index()

    def _load_index(self) -> list:
        """Load the saved session index JSON, or [] when missing/invalid."""
        if os.path.exists(SESSION_INDEX):
            try:
                with open(SESSION_INDEX, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_index(self) -> None:
        """Persist the current session index JSON to disk."""
        with open(SESSION_INDEX, 'w') as f:
            json.dump(self._index, f, indent=2)

    def save_session(self, messages: list, ctx: 'CommandContext') -> str:
        """Save current session to disk. Returns session ID."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(SESSION_DIR, f"{session_id}.json")

        summary = ""
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if not content.startswith("[CONVERSATION HISTORY COMPACTED]"):
                    summary = content[:100].replace('\n', ' ')
                    break

        session_data = {
            "id": session_id,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "model": ctx.model,
            "backend": ctx.backend,
            "base_url": ctx.base_url,
            "message_count": len(messages),
            "total_tokens": ctx.current_context_tokens,
            "summary": summary,
            "messages": messages
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False)

        self._index = [e for e in self._index if e['id'] != session_id]
        self._index.insert(0, {
            "id": session_id,
            "created": session_data["created"],
            "model": ctx.model,
            "message_count": len(messages),
            "summary": summary
        })

        if len(self._index) > MAX_SESSIONS:
            for old in self._index[MAX_SESSIONS:]:
                old_path = os.path.join(SESSION_DIR, f"{old['id']}.json")
                if os.path.exists(old_path):
                    try:
                        os.unlink(old_path)
                    except OSError:
                        pass
            self._index = self._index[:MAX_SESSIONS]

        self._save_index()
        return session_id

    def list_sessions(self, limit: int = 10) -> list:
        """Return recent session metadata."""
        return self._index[:limit]

    def load_session(self, session_id: str) -> Optional[dict]:
        """Load a session by ID (exact or prefix match)."""
        matches = [e for e in self._index if e['id'].startswith(session_id)]
        if not matches:
            return None
        target = matches[0]
        filepath = os.path.join(SESSION_DIR, f"{target['id']}.json")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None


# ============================================================================
# ============= COMPOSABLE AGENTIC SYSTEM PROMPT  ============================
# ============================================================================
# System prompt is assembled from blocks: role + tool defs + format + examples + rules.
# This avoids monolithic prompt strings and allows per-model format/style selection.

AGENTIC_ROLE_BLOCK = """You are a capable AI agent with access to tools. You operate in a terminal environment.

## Your capabilities
You have tools that let you read/write files, execute Python code, fetch URLs, search files, and apply diffs. Use them whenever you need information from the outside world."""

# Strict format: bare JSON only, no surrounding text, no code blocks.
# For models that reliably follow strict JSON-only instructions (Qwen3.5, GPT-OSS).
AGENTIC_FORMAT_STRICT = """## Output format
When you need to perform an action, respond with a JSON object:
{"tool": "tool_name", "arguments": {"arg1": "value1", ...}}

After the tool runs, you will receive the result as an observation. Use it to decide the next step.

## CRITICAL rules
- Output ONLY the JSON tool call — no surrounding text, no explanations, no markdown fences. If you add text before the JSON, the system will treat it as a final answer and ignore the tool call.
- Do NOT chain multiple commands with `&&`, `|`, `;` etc. Each tool call runs in isolation.
- You may make multiple tool calls sequentially — each result feeds back in.
- Format your final answer with markdown for terminal readability (code blocks, lists, headings)."""

# Soft format: allows code blocks, preamble, multi-tool-per-response.
# For models like Nemotron-Cascade that refuse bare JSON output.
AGENTIC_FORMAT_SOFT = """## Output format
When you need to perform an action, output a JSON tool call in a code block. Use one code block per tool call.

After the tool runs, you will receive the result as an observation. Use it to decide the next step.

## Rules
- You may output multiple tool calls in a single response (each in its own code block).
- Each tool call runs in isolation. Break multi-step tasks into separate calls.
- When you have the final answer, respond in plain text with markdown formatting."""

# ReAct protocol description matching each format style.
AGENTIC_EXAMPLE_STRICT = """## ReAct protocol (Think → Act → Observe → Answer)
1. **Think** about what the user needs and which tool can help
2. **Act** by outputting a JSON tool call
3. **Observe** the tool result (it will be shown to you)
4. **Repeat** if more actions are needed
5. When you have the answer, respond in plain text (no JSON) — that is your final answer"""

AGENTIC_EXAMPLE_SOFT = """## Protocol
1. **Think** about what steps are needed
2. **Act** by outputting the JSON tool call in a code block
3. **Observe** the tool result
4. **Repeat** if more actions are needed
5. When you have the final answer, respond in plain text with markdown formatting"""

# Native tools API format: for models that receive tool schemas via the OpenAI
# `tools` parameter (send_tools_api=True). opencode-style — the system prompt
# never describes JSON tool-call syntax; the model emits native `tool_calls`
# from its own training. Telling such models to write bare JSON into the message
# (AGENTIC_FORMAT_STRICT) causes hybrid/XML-garbage output (e.g. Qwen3.6 mixing
# `{"tool": ...}` with `</function></tool_call>`).
AGENTIC_FORMAT_NATIVE = """## Tool use
You have access to tools defined through the function-calling interface. When you need to perform an action, call the appropriate function. After a tool runs, you will receive the result as an observation — use it to decide the next step.

## CRITICAL rules
- Do NOT write tool calls as JSON text or XML in your reply. Issue them as proper function calls through the tool interface, then continue once the result is returned.
- Do NOT chain multiple commands with `&&`, `|`, `;` etc. Each tool call runs in isolation.
- You may make multiple tool calls sequentially — each result feeds back in.
- Format your final answer with markdown for terminal readability (code blocks, lists, headings)."""

AGENTIC_EXAMPLE_NATIVE = """## ReAct protocol (Think → Act → Observe → Answer)
1. **Think** about what the user needs and which tool can help
2. **Act** by calling the appropriate function through the tool interface
3. **Observe** the tool result (it will be shown to you)
4. **Repeat** if more actions are needed
5. When you have the answer, respond in plain text (no tool calls) — that is your final answer"""

# Shared rules block — common to all models.
AGENTIC_RULES_BLOCK = """## General rules
- Be precise with file paths. If you create a file in a subdirectory, use the same path when compiling or reading it later.
- Mirror the user's language — if they write in French, reply in French."""

# Registry mapping format style names to their blocks.
AGENTIC_FORMAT_REGISTRY = {
    "strict": AGENTIC_FORMAT_STRICT,
    "soft": AGENTIC_FORMAT_SOFT,
}

AGENTIC_EXAMPLE_REGISTRY = {
    "strict": AGENTIC_EXAMPLE_STRICT,
    "soft": AGENTIC_EXAMPLE_SOFT,
}

# Registry mapping model name substrings to format style.
AGENTIC_PROMPT_STYLE_REGISTRY = {
    "nemotron-cascade": "soft",
}

AGENTIC_PROMPT_STYLE_DEFAULT = "strict"

# Registry mapping model name substrings to tool delivery strategy.
# "openai": Pass tools via native OpenAI tools API parameter, strip inline tool defs from system prompt.
# "inline": Embed tool descriptions in system prompt, don't use native tools API.
# IMPORTANT: Longer/more-specific substrings must come before shorter ones
# to avoid false matches (e.g. "qwen3.5" before "qwen3").
TOOL_FORMAT_REGISTRY = [
    ("qwen3.5", "openai"),
    ("qwen3", "openai"),
    ("granite4", "openai"),
    ("granite-code", "openai"),
    ("rnj", "openai"),
    ("ministral", "inline"),
    ("glm-4", "inline"),
    ("glm4", "inline"),
    ("llama", "openai"),
    ("gpt-oss", "openai"),
    ("nemotron", "inline"),
    ("gemini", "openai"),
    ("opencodezen", "openai"),
    ("opencodego", "openai"),
    ("mistral", "openai"),
    ("deepseek", "openai"),
]

TOOL_FORMAT_DEFAULT = "inline"


def get_tool_format(model_name: str) -> str:
    """Determine tool delivery strategy for a model."""
    lower = model_name.lower()
    for key, fmt in TOOL_FORMAT_REGISTRY:
        if key in lower:
            return fmt
    return TOOL_FORMAT_DEFAULT


def get_prompt_style(model_name: str) -> str:
    """Determine which format style a model should use."""
    lower = model_name.lower()
    for key, style in AGENTIC_PROMPT_STYLE_REGISTRY.items():
        if key in lower:
            return style
    return AGENTIC_PROMPT_STYLE_DEFAULT


def get_agentic_prompt(model_name: str, tool_defs_block: str = "",
                       include_tool_defs: bool = True) -> str:
    """Assemble the agentic system prompt from composable blocks.

    Args:
        model_name: Used to select format style and tool-delivery strategy.
        tool_defs_block: Tool definitions block (from ToolRegistry.get_system_prompt_block()).
            Inserted after the role block so models see available tools before format instructions.
        include_tool_defs: If False, skip tool_defs_block (for models using native tools API).

    Format selection:
        - openai tool format (native tools API): AGENTIC_FORMAT_NATIVE — never
          describes JSON tool-call syntax (opencode-style).
        - otherwise: the style-registry format (strict/soft) with inline JSON.
    """
    tool_format = get_tool_format(model_name)
    style = get_prompt_style(model_name)
    blocks = [AGENTIC_ROLE_BLOCK]
    if include_tool_defs and tool_defs_block:
        blocks.append(tool_defs_block)
    if tool_format == "openai":
        blocks.append(AGENTIC_FORMAT_NATIVE)
        blocks.append(AGENTIC_EXAMPLE_NATIVE)
    else:
        blocks.append(AGENTIC_FORMAT_REGISTRY[style])
        blocks.append(AGENTIC_EXAMPLE_REGISTRY[style])
    blocks.append(AGENTIC_RULES_BLOCK)
    return "\n\n".join(blocks)


# Model-specific inference parameters for agentic/tool-calling mode.
# Keys are substrings matched against the lowercased model name.
# Values are passed as top-level fields to the llama.cpp OpenAI-compatible API.
# Sources: HuggingFace model card "Best Practices" sections.
MODEL_INFERENCE_PARAMS_REGISTRY = {
    "nemotron-cascade": {
        # https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B
        # HF recommends temperature=1.0 for general use, but we use 0.6 for
        # more deterministic tool-calling (inspired by Qwen coding best practices).
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "glm-4.7": {
        # https://medium.com/@zh.milo/glm-4-7-flash-the-ultimate-2026-guide-to-local-ai-coding-assistant
        # Fetched via: curl https://r.jina.ai/<URL>
        # EMPIRICAL VALIDATION (May 2026): Tested with 6-agentic-test suite
        # on glm-4.7-flash:q4_K_m at OLLAMA_HOST=http://192.168.1.20:11434.
        # Key findings:
        #   - temp=0.7 is optimal (validated): fast thinking (~5-13s vs qwen3's
        #     67-111s for simple queries), reliable tool calling
        #   - Lower temps (0.1, 0.5) cause overthinking/meta-reasoning loops
        #     similar to qwen3
        #   - min_p=0.01 is correct: prevents llama.cpp default 0.05 from
        #     over-pruning vocabulary during tool call JSON generation
        #   - top_p=1.0 works well: model's RL alignment handles token filtering
        #   - Intelligent debugging: model used netstat to discover correct
        #     listening IP instead of trying to start listeners
        #   - 5/6 E2E tests passed (port scanner overcame 127.0.0.1 vs
        #     192.168.1.20 mismatch automatically)
        #   - Web server test failed due to NULL pointer in accept() C code
        # See doc/model-parameters.md for full test results.
        "temperature": 0.7,
        "top_p": 1.0,
        "min_p": 0.01,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "qwen3.6": {
        # https://huggingface.co/Qwen/Qwen3.6-35B-A3B#best-practices
        # HF thinking-mode sampling for PRECISE CODING / agentic tool use:
        #   temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,
        #   repeat_penalty=1.0 (general tasks use temp=1.0 / presence=1.5;
        #   instruct mode uses temp=0.7 / top_p=0.80).
        # MUST be matched before the generic "qwen3" key (substring collision —
        # "qwen3" is contained in "qwen3.6"). The qwen3-8B values (0.5/0.9)
        # over-constrain sampling and correlate with degenerate/XML-hybrid
        # tool-call output.
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "qwen3.5": {
        # https://huggingface.co/Qwen/Qwen3.5-9B#best-practices
        # HF: temp=0.6, top_p=0.95, top_k=20 (thinking mode for precise coding tasks)
        # CAUTION: Gemini 3.5 community advice says temp=0.0-0.5, top_p=0.8-0.9
        # to reduce overthinking. Not official — evaluate before adopting.
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "qwen3": {
        # https://huggingface.co/Qwen/Qwen3-8B/blob/main/README.md
        # HF thinking mode: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
        # EMPIRICAL VALIDATION (May 2026): Tested with 6-agentic-test suite
        # on qwen3:8b at OLLAMA_HOST=http://192.168.1.20:11434.
        # Key findings:
        #   - temp=0.5 is best compromise: reliable tool calls for complex
        #     multi-step (write→compile→run) without overthinking loops
        #   - temp=0.1 is 7x faster for simple queries but causes
        #     meta-reasoning loops on networking code generation
        #   - Penalties (repeat=1.2, presence=0.3) backfire — increase
        #     verbose think blocks by 2-5x without improving output
        #   - top_k=20 prevents token sampling from wandering into
        #     low-probability tokens during JSON tool call generation
        # See doc/model-parameters.md for full test results.
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "deepseek": {
        # https://ollama.com/library/deepseek-r1
        # DeepSeek-R1-Distill-Qwen-8B is a reasoning model based on Qwen3-8B.
        # Ollama modelfile: temperature=0.6, top_p=0.95
        # EMPIRICAL VALIDATION (May 2026): Tested on deepseek-r1:8b
        # with 6-agentic-test suite.
        # Key findings:
        #   - temp=0.6 (Ollama default) works but verbose thinking
        #     (238s for simple query)
        #   - temp=0.7 is faster (57.5s) with same accuracy
        #   - Automatically outputs JSON tool calls without
        #     instruction following issues
        #   - Tool calls bypass the verbose reasoning, making
        #     fetch_url/write_file much faster than text responses
        #   - 5/6 E2E tests passed (web server fails like all models)
        #   - Compared to qwen3:8b: similar thinking verbosity but
        #     better structured reasoning output
        # Cloud DeepSeek API only consumes temperature/top_p (see
        # build_request_payload), so top_k/min_p are harmless there and
        # only apply to local llama.cpp runs. See doc/model-parameters.md.
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "nemotron": {
        # https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
        # "temperature=0.6 and top_p=0.95 are recommended for tool calling"
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "gpt-oss": {
        # CAUTION: Gemini 3.5 community advice (not official HF source):
        # temp=0.0 (strict JSON) or 0.7 (CoT), top_p=1.0.
        "temperature": 0.7,
        "top_p": 1.0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "gemini": {
        # Lower temperature for deterministic tool calling; top_k helps
        # reduce wandering during JSON generation.
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "mistral": {
        # Mistral API does not support top_k or min_p
        "temperature": 0.7,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
}

DEFAULT_INFERENCE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}


def get_inference_params(model_name: str) -> dict:
    """Look up inference params for a model by matching its name against the registry."""
    lower = model_name.lower()
    for key, params in MODEL_INFERENCE_PARAMS_REGISTRY.items():
        if key in lower:
            return dict(params)
    return dict(DEFAULT_INFERENCE_PARAMS)


_VISION_ERROR_KEYWORDS = [
    "does not support images", "does not support vision",
    "image processing", "multimodal", "image input",
    "vision is not supported", "this model does not support image",
]


def _extract_error_text(obj: object, depth: int = 0) -> str:
    """Recursively extract error text from nested structures.

    Args:
        obj: Any nested dict/list/str to walk.
        depth: Internal recursion guard (stops at depth 5).

    Returns:
        Flattened error text string.
    """
    if depth > 5:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        parts = []
        for key in ["message", "error", "text", "type", "detail", "details"]:
            if key in obj:
                parts.append(_extract_error_text(obj[key], depth + 1))
        return " ".join(parts)
    if isinstance(obj, list):
        return " ".join(_extract_error_text(item, depth + 1) for item in obj)
    return str(obj)


def check_vision_error(response: dict) -> bool:
    """Check if an API response dict indicates the model doesn't support vision.

    Args:
        response: Dict returned by the backend.

    Returns:
        True when the error text matches vision-unsupported keywords.

    Works for Ollama (/api/chat returns {"error": "..."}),
    and OpenAI-compatible APIs (/v1/chat/completions returns {"error": {"message": "..."}}).
    """
    if not isinstance(response, dict):
        return False
    error_text = _extract_error_text(response.get("error", ""))
    return any(kw in error_text.lower() for kw in _VISION_ERROR_KEYWORDS)


_TOOLS_ERROR_KEYWORDS = [
    "does not support tools", "tool calls are not supported",
    "tool_use", "tools is not supported", "tools not supported",
    "this model does not support tools",
]


def check_tools_error(response: dict) -> bool:
    """Check if an API response dict indicates the model doesn't support native tools.

    Args:
        response: Dict returned by the backend.

    Returns:
        True when the error text matches tools-unsupported keywords.

    Works for Ollama (/api/chat returns {"error": "..."}),
    and OpenAI-compatible APIs (/v1/chat/completions returns {"error": {"message": "..."}}).
    """
    if not isinstance(response, dict):
        return False
    error_text = _extract_error_text(response.get("error", ""))
    return any(kw in error_text.lower() for kw in _TOOLS_ERROR_KEYWORDS)


# ============================================================================
# ============= SHELL APPROVAL GATE (opencode-style + home guard + CWD leniency)
# ============================================================================
# Port of shell_permission_parser.py (opencode) + Hermes home-fold normalization.
# Whole-string executed via shell=True; approval is per-component.
# Design: ~/ hard deny (bypass-immune) → opencode breakdown → CWD leniency → prompt
# Single-file, stdlib-only, no packaging.

# --- Default permission config: opencode-style, pipes/redirs allowed ---
DEFAULT_SHELL_PERMISSION = {
    "bash": {
        "*": "ask",
        "git *": "allow",
        "grep *": "allow",
        "ls *": "allow",
        "cat *": "allow",
        "head *": "allow",
        "tail *": "allow",
        "wc *": "allow",
        "sort *": "allow",
        "uniq *": "allow",
        "cut *": "allow",
        "awk *": "allow",
        "sed *": "allow",
        "find *": "allow",
        "glob *": "allow",
        "echo *": "allow",
        "printf *": "allow",
        "pwd *": "allow",
        "whoami *": "allow",
        "uname *": "allow",
        "date *": "allow",
        "env *": "allow",
        "which *": "allow",
        "npm *": "allow",
        "pip *": "allow",
        "pip3 *": "allow",
        "python *": "allow",
        "python3 *": "allow",
        "node *": "allow",
        "cargo *": "allow",
        "go *": "allow",
        "make *": "allow",
        "gcc *": "allow",
        "g++ *": "allow",
        "javac *": "allow",
        "java *": "allow",
        "curl *": "allow",
        "wget *": "allow",
        "tar *": "allow",
        "gzip *": "allow",
        "gunzip *": "allow",
        "zip *": "allow",
        "unzip *": "allow",
        "diff *": "allow",
        "patch *": "allow",
        "chmod *": "ask",
        "chown *": "ask",
        "rm *": "ask",
        # Home hard-deny overrides the ask above (last-match-wins needs explicit deny after)
        "rm -rf ~": "deny",
        "rm -rf ~/ *": "deny",
        "rm -rf ~/*": "deny",
        "rm -rf $HOME *": "deny",
        "rm -rf $HOME/*": "deny",
        "rm -rf ${HOME} *": "deny",
    }
}

_SHELL_SESSION_APPROVED = []  # list of {"permission":"bash","pattern":..., "action":"allow"}

_ANSI_RE_SHELL = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

def _shell_strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from a shell command string."""
    return _ANSI_RE_SHELL.sub("", text)

def _shell_rewrite_home(cmd: str) -> str:
    """Rewrite an absolute $HOME prefix in a command to the `~` shorthand."""
    home = os.path.expanduser("~")
    if home and home != "~":
        cmd = cmd.replace(home + "/", "~/").replace(home, "~")
    return cmd

def _shell_normalize_for_home(cmd: str) -> str:
    """Normalization subset from hermes_style_approval.py for home guard."""
    cmd = _shell_strip_ansi(cmd)
    cmd = cmd.replace("\x00", "")
    cmd = unicodedata.normalize("NFKC", cmd)
    cmd = re.sub(r"\\\r?\n", "", cmd)
    cmd = _shell_rewrite_home(cmd)
    cmd = re.sub(r"\\([^\n])", r"\1", cmd)
    cmd = re.sub(r"''|\"\"", "", cmd)
    cmd = re.sub(r"\$\{IFS\b[^}]*\}|\$IFS\b", " ", cmd)
    return cmd

def _shell_escape_regex(pattern: str) -> str:
    """Escape regex metacharacters in a shell glob pattern."""
    out = []
    for ch in pattern:
        if ch in r".+^${}()|[\]\\":
            out.append("\\")
        out.append(ch)
    return "".join(out)

def _shell_wildcard_match(input_str: str, pattern: str) -> bool:
    """Match a shell-style glob (with `*`/`?`) against an input string."""
    normalized = input_str.replace("\\", "/")
    escaped = _shell_escape_regex(pattern.replace("\\", "/"))
    escaped = escaped.replace("*", ".*").replace("?", ".")
    if escaped.endswith(" .*"):
        escaped = escaped[:-3] + "( .*)?"
    return re.match("^" + escaped + "$", normalized, re.S) is not None

_SHELL_ARITY = {
    "cat": 1, "cd": 1, "chmod": 1, "chown": 1, "cp": 1, "echo": 1, "env": 1,
    "export": 1, "grep": 1, "kill": 1, "killall": 1, "ln": 1, "ls": 1,
    "mkdir": 1, "mv": 1, "ps": 1, "pwd": 1, "rm": 1, "rmdir": 1, "sleep": 1,
    "source": 1, "tail": 1, "touch": 1, "unset": 1, "which": 1,
    "aws": 3, "az": 3, "bazel": 2, "brew": 2, "bun": 2, "bun run": 3,
    "bun x": 3, "cargo": 2, "cargo add": 3, "cargo run": 3, "cdk": 2,
    "cf": 2, "cmake": 2, "composer": 2, "consul": 2, "consul kv": 3,
    "crictl": 2, "deno": 2, "deno task": 3, "doctl": 3,
    "docker": 2, "docker builder": 3, "docker compose": 3,
    "docker container": 3, "docker image": 3, "docker network": 3,
    "docker volume": 3, "eksctl": 2, "eksctl create": 3, "firebase": 2,
    "flyctl": 2, "gcloud": 3, "gh": 3, "git": 2, "git config": 3,
    "git remote": 3, "git stash": 3, "go": 2, "gradle": 2, "helm": 2,
    "heroku": 2, "hugo": 2, "ip": 2, "ip addr": 3, "ip link": 3,
    "ip netns": 3, "ip route": 3, "kind": 2, "kind create": 3, "kubectl": 2,
    "kubectl kustomize": 3, "kubectl rollout": 3, "kustomize": 2, "make": 2,
    "mc": 2, "mc admin": 3, "minikube": 2, "mongosh": 2, "mysql": 2,
    "mvn": 2, "ng": 2, "npm": 2, "npm exec": 3, "npm init": 3, "npm run": 3,
    "npm view": 3, "nvm": 2, "nx": 2, "openssl": 2, "openssl req": 3,
    "openssl x509": 3, "pip": 2, "pipenv": 2, "pnpm": 2, "pnpm dlx": 3,
    "pnpm exec": 3, "pnpm run": 3, "poetry": 2, "podman": 2,
    "podman container": 3, "podman image": 3, "psql": 2, "pulumi": 2,
    "pulumi stack": 3, "pyenv": 2, "python": 2, "rake": 2, "rbenv": 2,
    "redis-cli": 2, "rustup": 2, "serverless": 2, "sfdx": 3, "skaffold": 2,
    "sls": 2, "sst": 2, "swift": 2, "systemctl": 2, "terraform": 2,
    "terraform workspace": 3, "tmux": 2, "turbo": 2, "ufw": 2, "vault": 2,
    "vault auth": 3, "vault kv": 3, "vercel": 2, "volta": 2, "wp": 2,
    "yarn": 2, "yarn dlx": 3, "yarn run": 3,
}

def _shell_arity_prefix(tokens: list) -> list:
    """Return the command tokens up to the arity of the longest matching command prefix."""
    for length in range(len(tokens), 0, -1):
        prefix = " ".join(tokens[:length])
        arity = _SHELL_ARITY.get(prefix)
        if arity is not None:
            return tokens[:arity]
    if not tokens:
        return []
    return tokens[:1]

_SHELL_CWD = {"cd", "chdir", "popd", "pushd", "push-location", "set-location"}

_SHELL_WORD = "word"
_SHELL_REDIR = "redir"
_SHELL_CTRL = "ctrl"

class _ShellToken:
    """A shell token with kind, span, and optional nested tokens."""

    __slots__ = ("kind", "text", "start", "end", "nested")
    def __init__(self, kind: str, text: str, start: int, end: int, nested: Optional[list] = None) -> None:
        """Initialize a shell token.

        Args:
            kind: One of _SHELL_WORD / _SHELL_REDIR / _SHELL_CTRL.
            text: The token's raw text.
            start: Character offset of the token start.
            end: Character offset of the token end.
            nested: Optional list of tokens nested inside a $() expansion.
        """
        self.kind = kind
        self.text = text
        self.start = start
        self.end = end
        self.nested = nested or []

def _shell_find_closing(s: str, i: int, open_: str, close: str) -> int:
    """Find the index of the matching close character for a `$(` / `)` group.

    Args:
        s: The command string.
        i: Index just after the opening `$(`.
        open_: The open character (e.g. '(').
        close: The close character (e.g. ')').

    Returns:
        Index of the matching close char, or len(s) if unbalanced.
    """
    depth = 1
    n = len(s)
    j = i
    while j < n:
        c = s[j]
        if c == "'":
            k = s.find("'", j + 1)
            j = n if k == -1 else k + 1
            continue
        if c == '"':
            k = j + 1
            while k < n:
                if s[k] == "\\":
                    k += 2
                    continue
                if s[k] == '"':
                    break
                k += 1
            j = k + 1
            continue
        if c == "\\":
            j += 2
            continue
        if c == "$" and s[j:j + 2] == "$(":
            depth += 1
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return n

def _shell_scan_word(s: str, i: int) -> tuple:
    """Scan a shell word starting at index i, honoring quotes and $() nesting.

    Args:
        s: The command string.
        i: Starting index of the word.

    Returns:
        (j, parts, nested) where j is the end index, parts are the raw word
        segments, and nested is the list of nested tokens.
    """
    n = len(s)
    j = i
    parts = []
    nested = []
    while j < n:
        c = s[j]
        if c in " \t\r\n" or c in "|;&<>":
            break
        if c == "'":
            k = s.find("'", j + 1)
            if k == -1:
                k = n
            parts.append(s[j:k + 1])
            j = k + 1
            continue
        if c == '"':
            k = j + 1
            while k < n:
                if s[k] == "\\":
                    k += 2
                    continue
                if s[k] == '"':
                    break
                k += 1
            parts.append(s[j:k + 1])
            j = k + 1
            continue
        if c == "\\":
            parts.append(s[j:j + 2])
            j += 2
            continue
        if c == "`":
            k = s.find("`", j + 1)
            if k == -1:
                k = n
            parts.append(s[j:k + 1])
            nested.append(s[j + 1:k])
            j = k + 1
            continue
        if c == "$" and s[j:j + 2] == "$(":
            k = _shell_find_closing(s, j + 2, "(", ")")
            parts.append(s[j:k + 1])
            nested.append(s[j + 2:k])
            j = k + 1
            continue
        if c == "$" and s[j:j + 2] == "${":
            k = _shell_find_closing(s, j + 2, "{", "}")
            parts.append(s[j:k + 1])
            j = k + 1
            continue
        if c == "$":
            parts.append(c)
            j += 1
            continue
        if c == "(":
            k = _shell_find_closing(s, j + 1, "(", ")")
            parts.append(s[j:k + 1])
            nested.append(s[j + 1:k])
            j = k + 1
            continue
        parts.append(c)
        j += 1
    return "".join(parts), j, nested

_SHELL_REDIR_OPS = ("&>>", "<<-", "<<<", ">>", ">&", "<&", "<>", ">|", "<<", "&>", ">", "<")

def _shell_tokenize(s: str) -> list:
    """Tokenize a shell command string into _ShellToken objects."""
    tokens = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c in " \t\r\n":
            i += 1
            continue
        two = s[i:i + 2]
        if two == "||" or two == "&&" or two == "|&" or two == ";;":
            tokens.append(_ShellToken(_SHELL_CTRL, two, i, i + 2))
            i += 2
            continue
        if c in "<>" and s[i + 1:i + 2] == "(":
            k = _shell_find_closing(s, i + 2, "(", ")")
            tok = _ShellToken(_SHELL_WORD, s[i:k + 1], i, k + 1)
            tok.nested.append(s[i + 2:k])
            tokens.append(tok)
            i = k + 1
            continue
        if c == "&":
            op = None
            for cand in ("&>>", "&&", "&>"):
                if s.startswith(cand, i):
                    op = cand
                    break
            if op is None:
                op = "&"
            kind = _SHELL_CTRL if op == "&&" or op == "&" else _SHELL_REDIR
            tokens.append(_ShellToken(kind, op, i, i + len(op)))
            i += len(op)
            continue
        if c == "|":
            tokens.append(_ShellToken(_SHELL_CTRL, "|", i, i + 1))
            i += 1
            continue
        if c == ";":
            tokens.append(_ShellToken(_SHELL_CTRL, ";", i, i + 1))
            i += 1
            continue
        if c in "<>":
            op = None
            for cand in _SHELL_REDIR_OPS:
                if s.startswith(cand, i):
                    op = cand
                    break
            tokens.append(_ShellToken(_SHELL_REDIR, op, i, i + len(op)))
            i += len(op)
            continue
        text, j, nested = _shell_scan_word(s, i)
        tokens.append(_ShellToken(_SHELL_WORD, text, i, j, nested))
        i = j
    return tokens

class _ShellNode:
    """A parsed shell command node with source, glob pattern, and arity tokens."""

    __slots__ = ("source", "pattern", "tokens")
    def __init__(self, source: str, pattern: str, tokens: list) -> None:
        """Initialize a shell node.

        Args:
            source: Original command source text.
            pattern: Glob pattern used for rule matching.
            tokens: Arity-reduced token list.
        """
        self.source = source
        self.pattern = pattern
        self.tokens = tokens

def _shell_arity_tokens(tokens: list) -> list:
    """Drop redirection operators and their targets from a token list.

    Args:
        tokens: List of _ShellToken.

    Returns:
        List of word texts excluding redirect operators/operands.
    """
    skip = set()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.kind == _SHELL_REDIR:
            skip.add(i)
            if i + 1 < len(tokens) and tokens[i + 1].kind == _SHELL_WORD:
                skip.add(i + 1)
            i += 2
            continue
        i += 1
    out = []
    for idx, t in enumerate(tokens):
        if idx in skip or t.kind != _SHELL_WORD:
            continue
        if t.text.isdigit() and idx + 1 < len(tokens) and tokens[idx + 1].kind == _SHELL_REDIR and t.end == tokens[idx + 1].start:
            continue
        out.append(t.text)
    return out

def _shell_breakdown(command: str) -> list:
    """Split a shell command into _ShellNode segments at control operators.

    Args:
        command: The raw command string.

    Returns:
        List of _ShellNode, recursively including $()/subshell inner segments.
    """
    tokens = _shell_tokenize(command)
    segments = []
    current = []
    for t in tokens:
        if t.kind == _SHELL_CTRL:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(t)
    if current:
        segments.append(current)
    nodes = []
    for seg in segments:
        if len(seg) == 1 and seg[0].kind == _SHELL_WORD and seg[0].text.startswith(("(", "$(", "<(", ">(")):
            for inner in seg[0].nested:
                if inner.strip():
                    nodes.extend(_shell_breakdown(inner))
            continue
        source = command[seg[0].start:seg[-1].end].strip()
        tokens_ = _shell_arity_tokens(seg)
        nodes.append(_ShellNode(source, source, tokens_))
        for t in seg:
            if t.kind == _SHELL_WORD:
                for inner in t.nested:
                    if inner.strip():
                        nodes.extend(_shell_breakdown(inner))
    return nodes

def _shell_from_config(permission: object) -> list:
    """Flatten an opencode-style permission config into a rules list.

    Args:
        permission: str or dict permission spec (e.g. "ask" or {"*": "ask"}).

    Returns:
        List of {"permission", "pattern", "action"} rule dicts.
    """
    rules = []
    if isinstance(permission, str):
        permission = {"*": permission}
    for key, value in permission.items():
        if isinstance(value, str):
            rules.append({"permission": key, "pattern": "*", "action": value})
        elif isinstance(value, dict):
            for pattern, action in value.items():
                rules.append({"permission": key, "pattern": pattern, "action": action})
    return rules

def _shell_as_bash_rules(config: object) -> list:
    """Normalize a bash permission config into a rules list.

    Args:
        config: str, dict, or list of rule dicts.

    Returns:
        List of {"permission", "pattern", "action"} rule dicts.
    """
    if isinstance(config, str):
        return _shell_from_config({"bash": {"*": config}})
    if isinstance(config, dict):
        if "bash" in config:
            return _shell_from_config(config)
        return _shell_from_config({"bash": config})
    return list(config)

def _shell_evaluate(permission: str, pattern: str, *rulesets: tuple) -> Optional[dict]:
    """Evaluate a permission/pattern against rule sets, last match wins.

    Args:
        permission: The permission string to match.
        pattern: The command pattern to match.
        *rulesets: Ordered iterables of rule dicts.

    Returns:
        The winning rule dict, or None if no rule matched.
    """
    found = None
    for ruleset in rulesets:
        for rule in ruleset:
            if _shell_wildcard_match(permission, rule["permission"]) and _shell_wildcard_match(pattern, rule["pattern"]):
                found = rule
    if found is not None:
        return found
    return {"permission": permission, "pattern": "*", "action": "ask"}

def shell_check_command(command: str, config: object, approved: Optional[list] = None) -> Dict:
    """Break a command down and evaluate it against the shell permission rules.

    Args:
        command: The raw command string.
        config: Permission config (str/dict/list).
        approved: Optional list of pre-approved rule dicts.

    Returns:
        Verdict dict with command, nodes, patterns, always, effect, details.
    """
    rules = _shell_as_bash_rules(config)
    approved = list(approved) if approved else []
    nodes = _shell_breakdown(command)
    patterns = []
    always = []
    for node in nodes:
        if not node.tokens:
            continue
        cmd = node.tokens[0].strip("'\"")
        if cmd in _SHELL_CWD:
            continue
        patterns.append(node.pattern)
        always.append(" ".join(_shell_arity_prefix(node.tokens)) + " *")
    all_rulesets = (rules, approved)
    effect = "allow"
    details = []
    for pattern in patterns:
        rule = _shell_evaluate("bash", pattern, *all_rulesets)
        details.append({"pattern": pattern, "action": rule["action"], "rule": rule["pattern"]})
        if rule["action"] == "deny":
            effect = "deny"
            break
        if rule["action"] == "ask":
            effect = "ask"
    return {"command": command, "nodes": [{"source": n.source, "pattern": n.pattern, "tokens": n.tokens} for n in nodes], "patterns": patterns, "always": always, "effect": effect, "details": details}

# --- CWD leniency helpers ---
def _shell_path_inside_cwd(path_str: str, cwd: Optional[str] = None) -> bool:
    """Return True if a shell operand resolves inside the given working directory.

    Args:
        path_str: A command operand (may be quoted or a glob).
        cwd: Directory to test against (defaults to os.getcwd()).
    """
    if cwd is None:
        cwd = os.getcwd()
    # Strip quotes
    p = path_str.strip().strip("'\"")
    # Skip flags, globs without slash, bare commands
    if not p or p.startswith("-"):
        return True
    # Expand ~ and vars: treat as outside if contains ~
    if "~" in p or "$HOME" in p or "${HOME" in p:
        return False
    # If no slash and no dot, it's likely a cmd arg not a path -> treat as inside
    if "/" not in p and "." not in p and "*" not in p:
        return True
    # Resolve against cwd
    try:
        abs_p = os.path.abspath(os.path.join(cwd, os.path.expanduser(p)))
        real_cwd = os.path.realpath(cwd)
        real_p = os.path.realpath(abs_p)
        # For globs, check base directory
        if "*" in p or "?" in p:
            base = os.path.dirname(abs_p.split("*")[0].split("?")[0])
            if not base:
                base = cwd
            real_p = os.path.realpath(os.path.abspath(base))
        return os.path.commonpath([real_cwd, real_p]) == real_cwd
    except Exception:
        return False

def _shell_all_operands_inside_cwd(nodes: list, cwd: Optional[str] = None) -> bool:
    """Return True when every non-flag operand of the parsed nodes is inside CWD.

    Args:
        nodes: List of _ShellNode (or dict nodes) from _shell_breakdown.
        cwd: Directory to test against (defaults to os.getcwd()).
    """
    if cwd is None:
        cwd = os.getcwd()
    for n in nodes:
        toks = n.get("tokens") if isinstance(n, dict) else n.tokens
        if not toks:
            continue
        head = toks[0].strip("'\"") if toks[0] else ""
        if head in _SHELL_CWD:
            continue
        for tok in toks[1:]:
            # Skip flags
            if tok.startswith("-"):
                continue
            if not _shell_path_inside_cwd(tok, cwd):
                return False
    return True

# --- Home/root destructive guard (bypass-immune) ---
def _shell_is_home_destructive(nodes: list) -> bool:
    """Return True if any parsed node is an `rm` targeting ~ or the filesystem root.

    Args:
        nodes: List of _ShellNode (or dict nodes) from _shell_breakdown.
    """
    for n in nodes:
        toks = n.get("tokens") if isinstance(n, dict) else n.tokens
        if not toks:
            continue
        head = toks[0].strip("'\"").lower()
        if head != "rm":
            continue
        for tok in toks[1:]:
            low = tok.lower()
            # Home markers
            if "~" in tok or "$home" in low or "${home" in low:
                return True
            # Root filesystem markers
            stripped = tok.strip("'\"")
            if stripped in ("/", "/*", "/ *", "//", "///"):
                return True
            if stripped.startswith("/") and stripped.strip("/ ") in ("", "*"):
                return True
        # Also check source for folded home/root
        src = n.get("source") if isinstance(n, dict) else n.source
        if "~" in src or "$HOME" in src or "${HOME" in src:
            # Verify it's rm with home operand, not echo
            if head == "rm":
                return True
        # Direct rm -rf / detection on source
        if re.search(r'\brm\s+.*\s/+(\s|$|;|\||&)', src):
            # Only hard-block when operand is exactly root or root glob, not /tmp etc.
            if re.search(r'\brm\s+[^;|&]*\s/(?:\s|$|;|\||&|"|\\\')', src):
                # Check that it's not /tmp, /home etc. — only bare /
                # Use token check above for precision, fallback to root pattern
                toks_src = [t.strip("'\"") for t in toks[1:] if not t.startswith("-")]
                if "/" in toks_src or "/*" in toks_src:
                    return True
    return False


def _shell_operand_realpath(tok: str) -> Optional[str]:
    """Best-effort resolve of a shell token to an absolute realpath.

    Args:
        tok: A command operand string.

    Returns:
        A realpath string, or None if the token is not a resolvable path
        (flags, env assignments, bare words, URLs, empty tokens).
    """
    t = tok.strip().strip("'\"")
    t = t.lstrip("<>")  # strip redirection markers (>>/etc/foo)
    if not t or t.startswith("-"):
        return None
    if "*" in t or "?" in t:
        base = t.split("*")[0].split("?")[0].rstrip("/")
        if not base:
            return None
        t = base
    if "~" in t:
        t = os.path.expanduser(t)
    if "=" in t or "$" in t:
        return None  # assignments / env refs are not simple paths
    if not t.startswith("/"):
        if "/" not in t and "." not in t and "*" not in t and "?" not in t and "~" not in t:
            return None  # bare word arg, not a path
        t = os.path.join(os.getcwd(), t)
    try:
        return os.path.realpath(t)
    except Exception:
        return None


def _shell_acl_scan(command: str, ctx: Optional['CommandContext']) -> Optional[dict]:
    """Scan command operands against PathAcl (run_command path awareness).

    Only *explicit* ACL rules trigger here — /proc, /sys, /etc, home dotfiles,
    plus anything the user added. The generic outside-CWD default deny is NOT
    applied (the shell gate already owns that). Sensitive virtual files
    (/proc/*/mem etc.) are hard-denied.

    Args:
        command: The raw command string.
        ctx: CommandContext with a path_acl.

    Returns:
        None (no concern) or a dict {"approved": bool, "reason": str|None}.
    """
    acl = getattr(ctx, "path_acl", None) if ctx is not None else None
    if acl is None:
        return None
    deny_info = None
    ask_info = None
    for node in _shell_breakdown(command):
        toks = node.tokens
        if not toks:
            continue
        for tok in toks[1:]:
            rp = _shell_operand_realpath(tok)
            if rp is None:
                continue
            if _is_blocked_system_file(rp):
                acl.log_event("run_command", rp, "deny", None)
                return {"approved": False, "reason": f"System file blocked (sensitive virtual file): {rp}"}
            decision, rule = acl.evaluate("read", rp, explicit_only=True)
            if rule is None:
                continue
            if decision == "deny" and deny_info is None:
                deny_info = (tok, rp, rule)
            elif decision == "ask" and ask_info is None:
                ask_info = (tok, rp, rule)
    if deny_info:
        _, rp, rule = deny_info
        acl.log_event("run_command", rp, "deny", rule)
        shown = rule["path"] if rule.get("kind") == "prefix" else "home dotfile"
        return {"approved": False, "reason": f"denied by path ACL rule: {rule['action']} {shown}"}
    if ask_info:
        _, rp, rule = ask_info
        granted = acl.prompt("read", rp, rule, tool="run_command", context=command)
        acl.log_event("run_command", rp, "allow" if granted else "deny", rule)
        if not granted:
            shown = rule["path"] if rule.get("kind") == "prefix" else "home dotfile"
            return {"approved": False, "reason": f"denied by user (path ACL ask {shown})"}
    return None


def _shell_hard_block(command: str, verdict: dict) -> Optional[Dict]:
    """Check home/root destructive and explicit-rule denies (bypass-immune).

    Args:
        command: The raw command string.
        verdict: The opencode breakdown verdict dict.

    Returns:
        A deny result dict when hard-blocked, or None to continue the gate.
    """
    # Normalize for home check (fold resolved home)
    norm = _shell_normalize_for_home(command)
    norm_verdict = shell_check_command(norm, DEFAULT_SHELL_PERMISSION, _SHELL_SESSION_APPROVED)
    # Home/root guard: either original or normalized hits destructive rm (bypass-immune, contains 'rejected' for test compat)
    if _shell_is_home_destructive(verdict["nodes"]) or _shell_is_home_destructive(norm_verdict["nodes"]):
        return {"approved": False, "effect": "deny", "message": "BLOCKED (rejected): recursive delete of home directory (~/ or $HOME) or root filesystem is never allowed", "verdict": verdict}
    # Also catch opencode deny (explicit rules) — include 'rejected' for compat with safety_blocklist test
    if verdict["effect"] == "deny":
        return {"approved": False, "effect": "deny", "message": "BLOCKED (rejected): denied by permission rules (%s)" % ", ".join(d["pattern"] for d in verdict["details"] if d["action"]=="deny"), "verdict": verdict}
    return None


def _shell_prompt_approval(command: str, verdict: dict) -> Dict:
    """Interactively ask the user to approve a command (once/session/always/deny).

    Args:
        command: The raw command string.
        verdict: The opencode breakdown verdict dict.

    Returns:
        {"approved", "effect", "message", "verdict"} result dict.
    """
    # Prompt user: once / session / always / deny
    print(colorize(f"\n[Shell approval] `{command}`", 'warning'), file=sys.stderr)
    for d in verdict["details"]:
        if d["action"] == "ask":
            print(colorize(f"  → {d['pattern']} needs approval (rule: {d['rule']})", 'muted'), file=sys.stderr)
    try:
        reply = input(colorize("Allow? [y/N/s/a/d] (y=once, s=session, a=always, d=deny): ", 'warning')).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return {"approved": False, "effect": "ask", "message": "Denied (no input)", "verdict": verdict}
    if reply in ("y", "yes", "o", "once"):
        return {"approved": True, "effect": "allow", "message": None, "verdict": verdict}
    if reply in ("s", "session"):
        for pat in verdict["patterns"]:
            _SHELL_SESSION_APPROVED.append({"permission": "bash", "pattern": pat, "action": "allow"})
        return {"approved": True, "effect": "allow", "message": None, "verdict": verdict}
    if reply in ("a", "always"):
        for pat in verdict["patterns"]:
            _SHELL_SESSION_APPROVED.append({"permission": "bash", "pattern": pat, "action": "allow"})
        # Also store always pattern for future sessions (same list, session-scoped)
        for pat in verdict["always"]:
            _SHELL_SESSION_APPROVED.append({"permission": "bash", "pattern": pat, "action": "allow"})
        return {"approved": True, "effect": "allow", "message": None, "verdict": verdict}
    return {"approved": False, "effect": "ask", "message": "Denied by user", "verdict": verdict}


def check_shell_approval(command: str, ctx: Optional['CommandContext'] = None, executor_mode: str = "host") -> Dict:
    """Gate: home hard-deny → opencode breakdown → CWD leniency → prompt.

    Args:
        command: The raw command string.
        ctx: Optional CommandContext (for path ACL + auto_confirm).
        executor_mode: "host" or "container" (container skips the gate).

    Returns:
        dict: {"approved": bool, "effect": str, "message": str|None, "verdict": dict}
    """
    if not command or not command.strip():
        return {"approved": False, "effect": "deny", "message": "Empty command", "verdict": None}
    if executor_mode == "container":
        return {"approved": True, "effect": "allow", "message": None, "verdict": None}

    # Build verdict via opencode
    verdict = shell_check_command(command, DEFAULT_SHELL_PERMISSION, _SHELL_SESSION_APPROVED)
    nodes = verdict["nodes"]

    # Home/root guard + explicit-rule denies (bypass-immune)
    blocked = _shell_hard_block(command, verdict)
    if blocked:
        return blocked

    # Apply the path ACL on automatic allow paths. Explicit user approval via
    # the shell prompt below already overrides it (the user said yes to the
    # exact command), so those returns are left untouched.
    def _final_allow(**extra: dict) -> dict:
        """Final allow wrapper that applies the path-ACL operand scan."""
        res = _shell_acl_scan(command, ctx)
        if res is not None and not res["approved"]:
            return {"approved": False, "effect": "deny", "message": res["reason"], "verdict": verdict}
        return {"approved": True, "effect": "allow", "message": None, "verdict": verdict, **extra}

    if verdict["effect"] == "allow":
        return _final_allow()
    # effect == ask → check CWD leniency
    if _shell_all_operands_inside_cwd(nodes):
        return _final_allow(cwd_leniency=True)
    # Auto-confirm (yolo) respects home deny but allows ask
    if ctx is not None and getattr(ctx, "auto_confirm", False):
        return _final_allow(auto=True)
    # Interactive prompt (TTY required)
    if not sys.stdin.isatty():
        return {"approved": False, "effect": "ask", "message": "Approval required but no TTY — denied (use /agentic auto to allow)", "verdict": verdict}
    return _shell_prompt_approval(command, verdict)

# ============================================================================
# ============= EXECUTOR (CONTAINER/HOST)  ===================================
# ============================================================================

class Executor:
    """Runs shell commands on host or inside a container sandbox.

    Call map:
      run() → _run_shell() or _pre_pull_image()
      _run_shell() → subprocess.run / podman|docker exec
    """

    def __init__(self, mode: str = "host", container_runtime: Optional[str] = None,
                 container_image: Optional[str] = None) -> None:
        """Initialize the executor with host/container mode and runtime settings.

        Args:
            mode: "host" or "container".
            container_runtime: podman/docker runtime (defaults to env or podman).
            container_image: Image for container mode (defaults to python:3.12-alpine).
        """
        self.mode = mode
        self.runtime = container_runtime or os.environ.get("OLLAMAQUERY_CONTAINER_RT", "podman")
        self.image = container_image or os.environ.get("OLLAMAQUERY_CONTAINER_IMAGE",
                                                       "docker.io/library/python:3.12-alpine")

    def _pre_pull_image(self) -> None:
        """Pull the container image with a separate timeout so pulls don't consume command timeout."""
        try:
            subprocess.run(
                [self.runtime, "pull", self.image],
                capture_output=True, timeout=120, check=False
            )
        except Exception:
            pass

    def run(self, command: str, timeout: int = 120) -> dict:
        """Run a shell command on the host or inside a container.

        Args:
            command: The command string to run.
            timeout: Timeout in seconds (default 120).

        Returns:
            dict with stdout, stderr, returncode.
        """
        if self.mode == "container":
            self._pre_pull_image()
            cwd_bind = os.getcwd()
            wrapped = (
                f"{self.runtime} run --rm "
                f"-v {shlex.quote(cwd_bind)}:/workspace:Z "
                f"-w /workspace "
                f"{shlex.quote(self.image)} "
                f"sh -c {shlex.quote(command)}"
            )
            return self._run_shell(wrapped, timeout, is_container=True)
        return self._run_shell(command, timeout)

    def _run_shell(self, command: str, timeout: int, is_container: bool = False) -> dict:
        """Execute a command, applying the shell approval gate on host mode.

        Args:
            command: The command string to run.
            timeout: Timeout in seconds.
            is_container: Whether to run via the container runtime (no gate).

        Returns:
            dict with stdout, stderr, returncode.
        """
        if not is_container:
            ctx = CommandContext() if CommandContext._initialized else None
            res = check_shell_approval(command, ctx=ctx, executor_mode=self.mode)
            if not res["approved"]:
                return {"stdout": "", "stderr": res.get("message") or "Blocked by shell approval gate", "returncode": -1}
        try:
            if is_container:
                args_list = shlex.split(command)
                proc = subprocess.run(
                    args_list, shell=False,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=timeout
                )
            else:
                proc = subprocess.run(
                    command, shell=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=timeout, executable="/bin/bash"
                )
            return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"Timed out after {timeout}s", "returncode": -1}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "returncode": -1}


# ============================================================================
# ============= AGENTIC TOOL DEFINITIONS      ================================
# ============================================================================

AGENTIC_TOOL_DEFS = {
    "fetch_url": {
        "description": "Fetch a URL and return its content as plain text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL (http/https only)"}
            },
            "required": ["url"]
        }
    },
    "read_file": {
        "description": "Read a text file, optionally paging through large files by 1-based line offset. Files over 100KB auto-page. A paged result reports the served line range and the `next` offset to continue from (read_file(file_path=..., offset=...)). Lines over 2000 chars are truncated. Path relative to CWD. Reading outside CWD (e.g. /proc, /sys, /etc, home dotfiles) may prompt the user for approval — if denied, use run_command instead or ask the user to allow it via /agentic acl.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path relative to CWD; absolute paths outside CWD may require approval"},
                "offset": {"type": "integer", "description": "Optional 1-based line number to start reading from (default 1). Use the `next` offset from a previous paged result to continue."},
                "limit": {"type": "integer", "description": "Optional maximum number of lines to read (default 2000, capped at 2000)"}
            },
            "required": ["file_path"]
        }
    },
    "write_file": {
        "description": "Write text content to a file. Creates subdirectories if needed. Overwrites existing files.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path relative to CWD"},
                "content": {"type": "string", "description": "Text content to write"}
            },
            "required": ["file_path", "content"]
        }
    },
    "list_directory": {
        "description": "List files and directories. Directories have a trailing '/'.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list (default: '.')"}
            },
            "required": ["path"]
        }
    },
    "glob": {
        "description": "Find files matching a glob pattern (e.g. '**/*.py').",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match"}
            },
            "required": ["pattern"]
        }
    },
    "run_python": {
        "description": "Execute Python 3 code (inline or from a file). Returns stdout/stderr. Default timeout: 10s, max: 300s.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Inline Python code to run"},
                "file_path": {"type": "string", "description": "Path to .py file to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 10, max 300). Increase for long-running scripts."}
            }
        }
    },
    "run_command": {
        "description": "Execute a single shell command (compiler, build tool, etc.). Returns stdout/stderr. Default timeout: 10s, max: 300s.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 10, max 300). Increase for long-running compilations or tests."}
            },
            "required": ["command"]
        }
    },
    "diff": {
        "description": "Generate a unified diff between two files. Pure Python.",
        "parameters": {
            "type": "object",
            "properties": {
                "file1": {"type": "string", "description": "Original file"},
                "file2": {"type": "string", "description": "Modified file"},
                "label1": {"type": "string", "description": "Optional label for file1"},
                "label2": {"type": "string", "description": "Optional label for file2"}
            },
            "required": ["file1", "file2"]
        }
    },
    "patch": {
        "description": "Apply a unified diff to a file in-place using the `patch` command. Destructive — user confirmation required.",
        "parameters": {
            "type": "object",
            "properties": {
                "diff": {"type": "string", "description": "Unified diff text to apply"},
                "target": {"type": "string", "description": "File to patch"}
            },
            "required": ["diff", "target"]
        }
    },
    "edit_file": {
        "description": "Make a precise text replacement in an existing file. Finds exact old_string and replaces with new_string. Requires exactly one match.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path relative to CWD"},
                "old_string": {"type": "string", "description": "Exact text to find (must match exactly once)"},
                "new_string": {"type": "string", "description": "Replacement text"}
            },
            "required": ["file_path", "old_string", "new_string"]
        }
    },
    "apply_patch": {
        "description": "Apply a unified diff to the filesystem. Accepts standard unified diff format (---/+++ headers, @@ hunks) or OpenCode-style markers (*** Add File: path, *** Update File: path, *** Delete File: path, *** Move to: path). Creates/modifies/deletes files automatically based on diff content. Pure Python, no external dependencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "patch_text": {"type": "string", "description": "Unified diff text to apply. File paths are parsed from diff headers or OpenCode markers."}
            },
            "required": ["patch_text"]
        }
    }
}

DESTRUCTIVE_TOOLS = {"write_file", "run_python", "run_command", "patch", "edit_file", "apply_patch"}

# Tools whose re-execution can cause harm; used by the agentic timeout escalation
# policy to abort (rather than extend) when the last executed tool was one of these.
AGENTIC_TIMEOUT_ABORT_TOOLS = {"run_command", "patch", "edit_file"}

# Absolute ceiling for `agentic_timeout_max`. The escalation policy doubles the
# max each time the current budget reaches it, up to this hard cap; at the cap it
# stops growing (State B aborts, State A aborts after two stalled retries).
AGENTIC_TIMEOUT_MAX_CEILING = 7200


# ============================================================================
# ============= AGENTIC TOOL HANDLERS         ================================
# ============================================================================

def _tool_handle_fetch_url(self, args: dict) -> dict:
    """Fetch a URL and return its text content.

    Args:
        args: Tool arguments dict with "url".

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    url = args["url"]
    text, _tool = fetch_and_convert_url(url)
    return {"success": True, "output": text, "error": None}


# First path components that hint a missing-leading-slash system path
# (e.g. "proc/self/cgroup" → "/proc/self/cgroup").
SYSTEM_PATH_HINTS = ("proc", "sys")
# Sensitive virtual files that are never served (hard deny, ACL bypass-immune).
SYSTEM_READ_BLOCKED_BASENAMES = {"mem", "environ", "kcore", "pagemap", "maps", "smaps"}


def _is_blocked_system_file(abspath: str) -> bool:
    """True for sensitive /proc or /sys virtual files (secrets / huge dumps).

    Args:
        abspath: Absolute path to test.

    Returns:
        True when the path is a hard-blocked virtual file.
    """
    base = os.path.basename(abspath)
    if base in SYSTEM_READ_BLOCKED_BASENAMES:
        return True
    # /proc/<pid>/fd/* can leak open descriptors; refuse the fd tree.
    parts = abspath.split(os.sep)
    if "/proc" in parts and len(parts) >= 4 and parts[-2] == "fd":
        return True
    return False


def _resolve_tool_path(raw_path: str, allow_home: bool = False) -> tuple:
    """Resolve raw_path handling ~, absolute, and symlink.

    Args:
        raw_path: The path string from a tool argument.
        allow_home: Whether explicit `~`/home paths are permitted.

    Returns:
        (abspath, allowed, error). When allow_home=True, paths
        explicitly starting with ~ or absolute inside $HOME are allowed
        even though they are outside CWD (needed for ~/.vimrc etc.).
        All other paths must be inside CWD via realpath commonpath check.
    """
    expanded = os.path.expanduser(raw_path)
    if os.path.isabs(expanded):
        abspath = os.path.abspath(expanded)
    else:
        abspath = os.path.abspath(os.path.join(os.getcwd(), expanded))
    base_real = os.path.realpath(os.getcwd())
    file_real = os.path.realpath(abspath)
    # Check CWD containment
    try:
        if os.path.commonpath([base_real, file_real]) == base_real:
            return abspath, True, None
    except ValueError:
        pass
    # Optionally allow explicit home reads (~/... or /home/... when requested via ~)
    if allow_home:
        try:
            home_real = os.path.realpath(os.path.expanduser("~"))
            is_home_request = raw_path.startswith("~") or raw_path.startswith(home_real) or expanded.startswith(home_real)
            if is_home_request and os.path.commonpath([home_real, file_real]) == home_real:
                return abspath, True, None
        except ValueError:
            pass
    return abspath, False, "Path traversal denied"


def _is_home_dotfile(abspath: str) -> bool:
    """True if abspath is inside $HOME (but outside CWD) and any component is dotfile.

    Args:
        abspath: Absolute path to test.

    Returns:
        True when the file is a home dotfile outside the current working directory.

    Dotfiles inside the current working directory (e.g. ./.gitignore, ./.env)
    are NOT considered home dotfiles even when CWD itself is inside $HOME —
    they are project files and should not trigger the guard. Only files that
    are in $HOME but not in CWD are guarded (e.g. ~/.vimrc, ~/.ssh/id_rsa).
    """
    try:
        home_real = os.path.realpath(os.path.expanduser("~"))
        cwd_real = os.path.realpath(os.getcwd())
        file_real = os.path.realpath(abspath)
        # If inside CWD, it's a project dotfile — not a home dotfile
        try:
            if os.path.commonpath([cwd_real, file_real]) == cwd_real:
                return False
        except ValueError:
            pass
        if os.path.commonpath([home_real, file_real]) != home_real:
            return False
        rel = os.path.relpath(file_real, home_real)
        for part in rel.split(os.sep):
            if part.startswith(".") and part not in (".", "..") and part != "":
                return True
        return False
    except ValueError:
        return False


def _raw_is_home_explicit(raw_path: str) -> bool:
    """True if a raw tool path is an explicit home request (`~` prefix or an
    absolute path under $HOME). Relative `../` escapes that happen to land in
    $HOME are NOT explicit and stay denied.

    Args:
        raw_path: The original tool argument string.

    Returns:
        True when the path is an explicit home request.
    """
    if raw_path is None:
        return False
    if raw_path.startswith("~"):
        return True
    home_real = os.path.realpath(os.path.expanduser("~"))
    expanded = os.path.expanduser(raw_path)
    if expanded.startswith(home_real):
        return True
    return False


# ============================================================================
# ============= PATH ACL (session-scoped allow/ask/deny)  ====================
# ============================================================================

class PathAcl:
    """Session-scoped access policy for file tools and shell command operands.

    Rules are dicts:
      {op: 'read'|'write'|'any', action: 'allow'|'ask'|'deny',
       kind: 'prefix'|'home_dotfile', path: realpath (prefix kind),
       source: 'default'|'session'}

    Evaluation precedence (highest wins):
      1. explicit rule match — most-specific (longest realpath), session over default
      2. implicit default — inside CWD → allow; read inside $HOME (non-dotfile) → allow
      3. everything else → deny

    `ask` is bypass-immune: auto_confirm never auto-approves it (same as the
    dotfile gate). Non-TTY runs deny `ask` silently and record the denial.

    Default policy (steering-first): CWD + $HOME allowed; home dotfiles,
    /proc, /sys, /etc are `ask` (reminder prompt); everything else outside
    CWD/$HOME is denied.
    """

    def __init__(self) -> None:
        """Initialize the ACL with an empty log and the default rules."""
        self.rules: list = []
        self.log: list = []
        self._install_defaults()

    def _install_defaults(self) -> None:
        """Install the default allow/ask/deny policy (CWD, $HOME, /proc, /sys, /etc)."""
        for root in ("/proc", "/sys", "/etc"):
            self.add("read", "ask", root, source="default")
            self.add("write", "deny", root, source="default")
        home = os.path.realpath(os.path.expanduser("~"))
        for op in ("read", "write"):
            self.rules.append({"op": op, "action": "ask", "kind": "home_dotfile",
                               "path": home, "source": "default"})

    def add(self, op: str, action: str, path: str, source: str = "session") -> None:
        """Add/replace a prefix rule for `path` (read|write|any).

        Args:
            op: "read", "write", or "any".
            action: "allow", "ask", or "deny".
            path: Real path (or ~-expanded) the rule applies to.
            source: "session" (user) or "default".
        """
        rp = os.path.realpath(os.path.expanduser(path))
        self.rules = [r for r in self.rules
                      if not (r["source"] == source and r.get("kind") == "prefix"
                              and r["op"] in (op, "any") and r["path"] == rp)]
        self.rules.append({"op": op, "action": action, "kind": "prefix",
                           "path": rp, "source": source})

    def remove(self, path: str) -> None:
        """Remove session rules whose path equals `path` (realpath'd).

        Args:
            path: The path to drop from session rules.

        Default rules are preserved — use reset() to restore defaults.
        """
        rp = os.path.realpath(os.path.expanduser(path))
        self.rules = [r for r in self.rules
                      if not (r["source"] == "session"
                              and r.get("kind") == "prefix" and r["path"] == rp)]

    def reset(self) -> None:
        """Clear all rules and log, reinstalling the defaults."""
        self.rules = []
        self.log = []
        self._install_defaults()

    def evaluate(self, op: str, realpath: str, explicit_only: bool = False) -> tuple:
        """Resolve a (op, realpath) to (decision, rule|None).

        Args:
            op: Operation ("read"/"write"/"any").
            realpath: Realpath of the target file.
            explicit_only: If True, skip implicit CWD/$HOME defaults.

        Returns:
            (action, rule|None) tuple.
        """
        best = None
        best_key = None
        for rule in self.rules:
            if rule["op"] != "any" and rule["op"] != op:
                continue
            if rule.get("kind") == "prefix":
                try:
                    if os.path.commonpath([rule["path"], realpath]) != rule["path"]:
                        continue
                except ValueError:
                    continue
                key = (1 if rule["source"] == "session" else 0, len(rule["path"]))
            elif rule.get("kind") == "home_dotfile":
                if not _is_home_dotfile(realpath):
                    continue
                key = (1 if rule["source"] == "session" else 0, len(rule["path"]) + 1)
            else:
                continue
            if best_key is None or key > best_key:
                best_key = key
                best = rule
        if best is not None:
            return best["action"], best
        if explicit_only:
            return "allow", None
        # Implicit defaults
        cwd_real = os.path.realpath(os.getcwd())
        try:
            if os.path.commonpath([cwd_real, realpath]) == cwd_real:
                return "allow", None
        except ValueError:
            pass
        if op == "read":
            home_real = os.path.realpath(os.path.expanduser("~"))
            try:
                if os.path.commonpath([home_real, realpath]) == home_real \
                        and not _is_home_dotfile(realpath):
                    return "allow", None
            except ValueError:
                pass
        return "deny", None

    def log_event(self, tool: str, path: str, decision: str, rule: Optional[dict] = None) -> None:
        """Record a path ACL decision in the rolling log (capped at 100 entries).

        Args:
            tool: Tool name (e.g. "read_file", "run_command").
            path: The realpath being accessed.
            decision: The resulting decision ("allow"/"ask"/"deny").
            rule: Optional matching rule dict.
        """
        self.log.append({
            "tool": tool, "path": path, "decision": decision,
            "rule": (rule["path"] if rule and rule.get("kind") == "prefix"
                     else "home_dotfile" if rule else None),
            "ts": time.strftime("%H:%M:%S"),
        })
        if len(self.log) > 100:
            self.log = self.log[-100:]

    def prompt(self, op: str, realpath: str, rule: Optional[dict], tool: str = "file", context: str = "") -> bool:
        """Interactive `ask` gate. Bypass-immune (auto_confirm does not
        auto-approve). Returns True if the user granted session access.

        Args:
            op: Operation ("read"/"write").
            realpath: The realpath being accessed.
            rule: The matching rule dict (for display).
            tool: Tool name for the log.
            context: Extra context line to show.

        Returns:
            True when the user granted access (adding a session allow rule).
        """
        if not sys.stdin.isatty():
            return False
        op_label = "read" if op == "read" else "write"
        print(colorize(f"\n[Path approval] {op_label} {realpath}", 'warning'), file=sys.stderr)
        if context:
            print(colorize(f"  from: {context}", 'muted'), file=sys.stderr)
        if rule and rule.get("kind") == "prefix":
            print(colorize(f"  rule: {rule['action']} {rule['path']}", 'muted'), file=sys.stderr)
        else:
            print(colorize("  rule: ask home dotfile", 'muted'), file=sys.stderr)
        try:
            reply = input(colorize("Allow? [y/N/s/a/d] (s/a = this session, d = deny): ", 'warning')).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if reply in ("y", "yes", "o", "once", "s", "session", "a", "always"):
            if rule and rule.get("kind") == "prefix":
                self.add(op, "allow", rule["path"], source="session")
            else:
                self.add(op, "allow", realpath, source="session")
            return True
        return False


def _path_acl_decision(ctx: Optional['CommandContext'], op: str, realpath: str, tool: str, context: str = "", raw_path: Optional[str] = None) -> tuple:
    """Apply PathAcl to an operation. Returns (decision, message|None).

    Args:
        ctx: CommandContext with a path_acl.
        op: Operation ("read"/"write").
        realpath: The realpath being accessed.
        tool: Tool name for logging.
        context: Extra context line for the prompt.
        raw_path: The original tool path (used for home-explicitness gating).

    Returns:
        (decision, message|None) tuple.

    'ask' rules are resolved via the interactive gate (bypass-immune).
    Sensitive virtual files (/proc/*/mem etc.) are hard-denied regardless.
    `raw_path`, when provided, gates the implicit $HOME read-allow: only an
    explicit home request (leading `~` or absolute under $HOME) benefits from
    it — a relative `../` escape that happens to land in $HOME stays denied.
    """
    acl = getattr(ctx, "path_acl", None) if ctx is not None else None
    if acl is None:
        return "allow", None
    if _is_blocked_system_file(realpath):
        acl.log_event(tool, realpath, "deny", None)
        return "deny", "System file blocked (sensitive virtual file)"
    decision, rule = acl.evaluate(op, realpath)
    # Reject implicit home-read allowance for non-explicit home requests.
    # CWD is always allowed — this only applies to $HOME paths OUTSIDE CWD,
    # so relative `../` escapes that land in $HOME stay denied.
    if decision == "allow" and rule is None and op == "read":
        cwd_real = os.path.realpath(os.getcwd())
        try:
            in_cwd = os.path.commonpath([cwd_real, realpath]) == cwd_real
        except ValueError:
            in_cwd = False
        if not in_cwd:
            home_real = os.path.realpath(os.path.expanduser("~"))
            try:
                in_home = os.path.commonpath([home_real, realpath]) == home_real
            except ValueError:
                in_home = False
            if in_home and not _is_home_dotfile(realpath) and not _raw_is_home_explicit(raw_path):
                decision = "deny"
    if decision == "ask":
        granted = acl.prompt(op, realpath, rule, tool=tool, context=context)
        decision = "allow" if granted else "deny"
    acl.log_event(tool, realpath, decision, rule)
    if decision == "deny":
        if rule is not None:
            shown = rule["path"] if rule.get("kind") == "prefix" else "home dotfile"
            return "deny", f"denied by path ACL rule: {rule['action']} {shown}"
        return "deny", "Path traversal denied"
    return "allow", None

def _tool_handle_read_file(self, args: dict) -> dict:
    """Read a file (with ACL gate + optional paging).

    Args:
        args: Tool arguments with "file_path", optional "offset"/"limit".

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    raw = args["file_path"]
    filepath, _allowed, _err = _resolve_tool_path(raw, allow_home=True)
    ctx = self._ctx if self._ctx is not None else (CommandContext() if CommandContext._initialized else None)
    # Models frequently drop the leading slash on system paths
    # ("proc/self/cgroup" → "/proc/self/cgroup"). If the CWD-relative
    # resolution doesn't exist but the root-relative one does, use it.
    if not os.path.lexists(filepath) and not os.path.isabs(raw):
        stripped = raw.lstrip("./")
        first = stripped.split("/", 1)[0]
        if first in SYSTEM_PATH_HINTS:
            alt_path, _alt_allowed, _alt_err = _resolve_tool_path("/" + stripped, allow_home=True)
            if os.path.lexists(alt_path):
                filepath = alt_path
    decision, acl_err = _path_acl_decision(ctx, "read", os.path.realpath(filepath),
                                           tool="read_file", context=raw, raw_path=raw)
    if decision == "deny":
        return {"success": False, "output": "", "error": acl_err or "Path traversal denied"}
    if not os.path.isfile(filepath):
        return {"success": False, "output": "", "error": "File not found"}
    try:
        try:
            offset = int(args["offset"]) if args.get("offset") is not None else None
        except (TypeError, ValueError):
            offset = None
        try:
            limit = int(args["limit"]) if args.get("limit") is not None else None
        except (TypeError, ValueError):
            limit = None
        size = os.path.getsize(filepath)
        # Paged mode kicks in for large files or an explicit offset/limit.
        if size > MAX_READ_FILE_SIZE or offset is not None or limit is not None:
            return _read_file_page(filepath, raw, offset, limit)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_READ_FILE_SIZE + 1)
        truncated = len(content) > MAX_READ_FILE_SIZE
        if truncated:
            content = content[:MAX_READ_FILE_SIZE]
            content += f"\n... [truncated: file exceeds {MAX_READ_FILE_SIZE} chars; showing first {MAX_READ_FILE_SIZE}]"
        return {"success": True, "output": content, "error": None}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

def _read_file_page(filepath: str, raw: str, offset: Optional[int], limit: Optional[int]) -> dict:
    """Read a paged window of lines from a (possibly large) text file.

    Mirrors opencode's paged `read`: returns at most `limit` lines (capped at
    MAX_READ_LINES) and at most MAX_READ_FILE_SIZE bytes, truncating any line
    longer than MAX_READ_LINE_LENGTH. A leading header reports the served line
    range and, when more content follows, the exact `next` 1-based offset so
    the model can chain `read_file` calls.

    Args:
        filepath: Resolved absolute path of the file.
        raw: The path string as the model passed it (echoed in the header).
        offset: 1-based starting line (None → 1).
        limit: Max lines to return (None → MAX_READ_LINES).

    Returns:
        {"success": True, "output": <paged text>}.
    """
    start = offset if offset is not None and offset >= 1 else 1
    cap = min(limit, MAX_READ_LINES) if limit is not None and limit >= 1 else MAX_READ_LINES

    collected = []
    seen = 0
    bytes_used = 0
    more = False
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            seen += 1
            if seen < start:
                continue
            if len(collected) >= cap:
                more = True
                break
            text = line.rstrip("\r\n")
            if len(text) > MAX_READ_LINE_LENGTH:
                text = text[:MAX_READ_LINE_LENGTH] + f" ... (line truncated to {MAX_READ_LINE_LENGTH} chars)"
            size_bytes = len(text.encode("utf-8", "replace")) + (1 if collected else 0)
            if bytes_used + size_bytes > MAX_READ_FILE_SIZE:
                more = True
                break
            collected.append(text)
            bytes_used += size_bytes

    if not collected:
        return {"success": True,
                "output": f"[read_file {raw}: no lines at offset {start} (end of file).]",
                "error": None}

    end = start + len(collected) - 1
    body = "\n".join(collected)
    if more:
        header = (f"[read_file {raw}: lines {start}-{end}, more available — "
                  f"use read_file(file_path=\"{raw}\", offset={end + 1}) to read the next page.]")
    else:
        header = f"[read_file {raw}: lines {start}-{end} (end of file).]"
    return {"success": True, "output": header + "\n" + body, "error": None}


def _tool_handle_write_file(self, args: dict) -> dict:
    """Write a file after the path-ACL write gate.

    Args:
        args: Tool arguments with "file_path" and "content".

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    # Write policy is enforced by PathAcl: CWD allowed, home dotfiles ask,
    # everything else outside CWD denied (system paths /etc /proc /sys deny).
    filepath, _allowed, _err = _resolve_tool_path(args["file_path"], allow_home=True)
    ctx = self._ctx if self._ctx is not None else (CommandContext() if CommandContext._initialized else None)
    decision, acl_err = _path_acl_decision(ctx, "write", os.path.realpath(filepath),
                                           tool="write_file", context=args["file_path"])
    if decision == "deny":
        return {"success": False, "output": "", "error": acl_err or "Path traversal denied"}
    content = args["content"]
    if len(content) > MAX_WRITE_FILE_SIZE:
        return {"success": False, "output": "", "error": "Content too large (max 1MB)"}
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "output": f"Written {len(content)} bytes to {args['file_path']}", "error": None}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def _tool_handle_list_directory(self, args: dict) -> dict:
    """List directory entries (ACL-gated), sorted, with dirs suffixed by '/'.

    Args:
        args: Tool arguments with optional "path" (default ".").

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    path, _allowed, _err = _resolve_tool_path(args.get("path", "."), allow_home=True)
    ctx = self._ctx if self._ctx is not None else (CommandContext() if CommandContext._initialized else None)
    decision, acl_err = _path_acl_decision(ctx, "read", os.path.realpath(path),
                                           tool="list_directory", context=path, raw_path=args.get("path", "."))
    if decision == "deny":
        return {"success": False, "output": "", "error": acl_err or "Path traversal denied"}
    if not os.path.isdir(path):
        return {"success": False, "output": "", "error": "Not a directory"}
    try:
        entries = []
        for name in sorted(os.listdir(path)):
            suffix = "/" if os.path.isdir(os.path.join(path, name)) else ""
            entries.append(f"{name}{suffix}")
        return {"success": True, "output": "\n".join(entries), "error": None}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def _tool_handle_glob(self, args: dict) -> dict:
    """Glob for files matching a pattern, restricted to the working directory.

    Args:
        args: Tool arguments with "pattern".

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    base_dir = os.path.abspath(os.getcwd())
    try:
        matches = sorted(glob.glob(args["pattern"], recursive=True))
        safe_matches = []
        for m in matches:
            abs_m = os.path.abspath(m)
            if os.path.commonpath([base_dir, abs_m]) == base_dir:
                safe_matches.append(m)
        return {"success": True, "output": "\n".join(safe_matches), "error": None}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def _tool_handle_run_python(self, args: dict) -> dict:
    """Run Python 3 code (inline or from a file) through the executor.

    Args:
        args: Tool arguments with "code" or "file_path"/"file", optional "timeout".

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    if "code" in args:
        command = f"python3 -c {shlex.quote(args['code'])}"
    elif "file_path" in args:
        command = f"python3 {shlex.quote(args['file_path'])}"
    elif "file" in args:
        command = f"python3 {shlex.quote(args['file'])}"
    else:
        return {"success": False, "output": "", "error": "Provide either 'code' or 'file'"}
    cmd_timeout = int(args.get("timeout", 10))
    if cmd_timeout <= 0 or cmd_timeout > 300:
        return {"success": False, "output": "", "error": f"Invalid timeout: {cmd_timeout}. Timeout is in seconds (1-300). Use a value between 1 and 300."}
    result = self.executor.run(command, timeout=cmd_timeout)
    output = result["stdout"]
    if result["stderr"]:
        output += f"\n[stderr]\n{result['stderr']}"
    return {"success": result["returncode"] == 0, "output": output, "error": result["stderr"] or None}


def _tool_handle_run_command(self, args: dict) -> dict:
    """Run a shell command through the executor (with the approval gate).

    Args:
        args: Tool arguments with "command", optional "timeout".

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    command = args["command"]
    # Gate is now in Executor._run_shell (opencode breakdown + home guard + CWD leniency)
    # Pipes/redirections are allowed when per-component rules permit; no operator ban here.
    cmd_timeout = int(args.get("timeout", 10))
    if cmd_timeout <= 0 or cmd_timeout > 300:
        return {"success": False, "output": "", "error": f"Invalid timeout: {cmd_timeout}. Timeout is in seconds (1-300). Use a value between 1 and 300."}
    result = self.executor.run(command, timeout=cmd_timeout)
    output = result["stdout"]
    if result["stderr"]:
        output += f"\n[stderr]\n{result['stderr']}"
    return {"success": result["returncode"] == 0, "output": output, "error": result["stderr"] or None}


def _tool_handle_diff(self, args: dict) -> dict:
    """Generate a unified diff between two files (pure Python).

    Args:
        args: Tool arguments with "file1"/"file2", optional "label1"/"label2".

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    filepath1, ok1, err1 = _resolve_tool_path(args["file1"], allow_home=False)
    if not ok1:
        return {"success": False, "output": "", "error": err1}
    filepath2, ok2, err2 = _resolve_tool_path(args["file2"], allow_home=False)
    if not ok2:
        return {"success": False, "output": "", "error": err2}
    label1 = args.get("label1", args["file1"])
    label2 = args.get("label2", args["file2"])
    try:
        with open(filepath1, "r", encoding="utf-8", errors="replace") as f:
            lines1 = f.readlines()
        with open(filepath2, "r", encoding="utf-8", errors="replace") as f:
            lines2 = f.readlines()
        diff = list(difflib.unified_diff(lines1, lines2, fromfile=label1, tofile=label2))
        output = "".join(diff) if diff else "Files are identical"
        return {"success": True, "output": output, "error": None}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def _tool_handle_patch(self, args: dict) -> dict:
    """Apply a unified diff to a file via the `patch` command (destructive).

    Args:
        args: Tool arguments with "diff" and optional "target".

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    import tempfile
    diff_text = args["diff"]
    diff_path = None
    target_arg = args.get("target", "").strip()
    if target_arg:
        target_raw, ok, err = _resolve_tool_path(target_arg, allow_home=False)
        if not ok:
            return {"success": False, "output": "", "error": err}
        target = shlex.quote(target_raw)
    else:
        sections = _parse_patch_sections(diff_text)
        for sec in sections:
            abspath, ok, err = _resolve_tool_path(sec["path"], allow_home=False)
            if not ok:
                return {"success": False, "output": "", "error": f"Path traversal denied within diff headers: {sec['path']}"}
        target = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            f.write(diff_text)
            diff_path = f.name
        command = f"patch -i {shlex.quote(diff_path)} {target}" if target else f"patch -i {shlex.quote(diff_path)}"
        result = self.executor.run(command, timeout=120)
        if result["returncode"] == 0:
            return {"success": True, "output": result["stdout"], "error": None}
        return {"success": False, "output": result["stdout"], "error": result["stderr"]}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}
    finally:
        if diff_path and os.path.exists(diff_path):
            os.unlink(diff_path)


def _tool_handle_edit_file(self, args: dict) -> dict:
    """Replace an exact old_string with new_string in a file (single match).

    Args:
        args: Tool arguments with "file_path"/"old_string"/"new_string".

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    filepath, _allowed, _err = _resolve_tool_path(args["file_path"], allow_home=True)
    ctx = self._ctx if self._ctx is not None else (CommandContext() if CommandContext._initialized else None)
    decision, acl_err = _path_acl_decision(ctx, "write", os.path.realpath(filepath),
                                           tool="edit_file", context=args["file_path"])
    if decision == "deny":
        return {"success": False, "output": "", "error": acl_err or "Path traversal denied"}
    if not os.path.isfile(filepath):
        return {"success": False, "output": "", "error": f"File not found: {args['file_path']}"}
    old = args["old_string"]
    new_string = args["new_string"]
    if not old:
        return {"success": False, "output": "", "error": "old_string must not be empty"}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if os.name == 'nt':
            content = content.replace('\r\n', '\n')
            old = old.replace('\r\n', '\n')
        count = content.count(old)
        if count == 0:
            return {"success": False, "output": "", "error": "old_string not found in file"}
        if count > 1:
            return {"success": False, "output": "", "error": f"Found {count} matches. Provide more surrounding context in old_string to make the match unique."}
        content = content.replace(old, new_string, 1)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "output": f"Replaced 1 occurrence in {args['file_path']}", "error": None}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def _parse_opencode_marker(line: str) -> Optional[tuple]:
    """Parse an OpenCode-style `*** <Op> to: path` marker line.

    Args:
        line: A patch line starting with `*** `.

    Returns:
        (operation, path, destination) tuple where operation is one of
        "add"/"modify"/"delete"/"move", path is the target file, and
        destination is set only for "move" operations. Returns None if the
        line is not a recognized marker.
    """
    marker = line[4:].strip()
    if marker.startswith("Add File:"):
        return "add", marker[len("Add File:"):].strip(), None
    if marker.startswith("Update File:"):
        return "modify", marker[len("Update File:"):].strip(), None
    if marker.startswith("Delete File:"):
        return "delete", marker[len("Delete File:"):].strip(), None
    if marker.startswith("Move to:"):
        dest = marker[len("Move to:"):].strip()
        # The marker tells us the destination; the source comes from ---/+++
        # headers or a path before this.
        return "move", dest, dest
    return None


def _parse_unified_diff_section(lines: list, i: int) -> Optional[tuple]:
    """Parse a standard unified diff section starting at a `--- ` line.

    Args:
        lines: Patch lines (kept with line endings).
        i: Index of the `--- ` line in `lines`.

    Returns:
        (path, operation, hunks, next_index) tuple, or None if the lines at
        index `i` do not form a valid `--- ` / `+++ ` section. `next_index`
        is the index just past the parsed section's hunks.
    """
    n = len(lines)
    old_path = lines[i][4:].strip()
    if i + 1 >= n or not lines[i + 1].startswith("+++ "):
        return None
    new_path = lines[i + 1][4:].strip()
    i += 2

    # Strip "a/" and "b/" prefixes commonly used by git
    src = old_path[2:] if old_path.startswith(("a/", "b/")) else old_path
    dst = new_path[2:] if new_path.startswith(("a/", "b/")) else new_path
    path = dst if dst != "/dev/null" else src
    is_new = old_path == "/dev/null" or old_path.endswith("/dev/null")
    is_delete = new_path == "/dev/null" or new_path.endswith("/dev/null")

    # Collect hunks
    body_start = i
    while i < n:
        if lines[i].startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            break
        i += 1
    body = "".join(lines[body_start:i])
    hunks, _, _ = _parse_unified_hunks(body)

    if is_delete:
        operation = "delete"
        hunks = []
    elif is_new:
        operation = "add"
    else:
        operation = "modify"
    return path, operation, hunks, i


def _parse_patch_sections(patch_text: str) -> list:
    """Parse patch text into sections for each file.

    Args:
        patch_text: The raw unified-diff / OpenCode-marker patch text.

    Returns:
        List of dicts:
          {"path": str, "operation": "add"|"modify"|"delete"|"move",
           "destination": str (for move), "hunks": [{"start": int, "old_count": int, "new_lines": [str]}]}
    """
    lines = patch_text.splitlines(keepends=True)
    sections = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # OpenCode-style marker: *** Add/Update/Delete/Move to: path
        if line.startswith("*** "):
            marker = _parse_opencode_marker(line)
            if marker is None:
                i += 1
                continue
            op, path, dest = marker
            if op == "delete":
                sections.append({"path": path, "operation": "delete", "hunks": []})
                i += 1
                continue

            i += 1
            # Collect the diff body (---/+++ lines + hunks) that follows the marker
            body_start = i
            while i < n and not lines[i].startswith("*** "):
                if i > body_start and lines[i].startswith("--- ") and lines[i-1].startswith("*** "):
                    break
                i += 1
            body = "".join(lines[body_start:i])
            hunks, detected_path, is_delete = _parse_unified_hunks(body)
            if detected_path and op != "move":
                path = detected_path
            if is_delete:
                sections.append({"path": path, "operation": "delete", "hunks": []})
            elif op == "move":
                source_path = detected_path or path
                sections.append({"path": source_path, "operation": "move", "destination": dest, "hunks": hunks})
            else:
                sections.append({"path": path, "operation": op, "hunks": hunks})
            continue

        # Standard unified diff section: starts with "--- "
        parsed = _parse_unified_diff_section(lines, i)
        if parsed:
            path, operation, hunks, next_i = parsed
            sections.append({"path": path, "operation": operation, "hunks": hunks})
            i = next_i
            continue
        i += 1

    return sections


def _collect_hunk_lines(lines: list, i: int, n: int, old_count: int, new_count: int) -> tuple:
    """Collect old/new lines for a numbered @@ hunk.

    Args:
        lines: Diff body lines (kept with endings).
        i: Index of the first line after the @@ header.
        n: Total line count.
        old_count / new_count: Expected removed/added line counts from the header.

    Returns:
        (i, old_lines, new_lines) where i is the index just past the hunk.
    """
    old_lines = []
    new_lines = []
    old_collected = 0
    new_collected = 0
    while i < n and (old_collected < old_count or new_collected < new_count):
        cl = lines[i]
        if cl.strip() == "" and not cl.startswith(("+", "-")):
            new_lines.append(cl[1:] if cl.startswith(" ") else cl)
            old_lines.append(cl[1:] if cl.startswith(" ") else cl)
            new_collected += 1
            old_collected += 1
            i += 1
            continue
        if cl.startswith("+") or cl.startswith(" "):
            new_lines.append(cl[1:] if cl.startswith("+") else cl[1:])
            new_collected += 1
        if cl.startswith("-") or cl.startswith(" "):
            old_lines.append(cl[1:] if cl.startswith("-") else cl[1:])
            old_collected += 1
        if not (cl.startswith(("+", "-", " ")) or cl.startswith("\\ ")):
            break
        i += 1
    return i, old_lines, new_lines


def _collect_bare_hunk_lines(lines: list, i: int, n: int) -> tuple:
    """Collect old/new lines for a bare `@@` (match-anywhere) hunk.

    Args:
        lines: Diff body lines (kept with endings).
        i: Index of the first line after the bare @@ marker.
        n: Total line count.

    Returns:
        (i, old_lines, new_lines) where i is the index just past the hunk.
    """
    old_lines = []
    new_lines = []
    while i < n:
        cl = lines[i]
        if cl.startswith('@@') or cl.startswith('*** '):
            break
        if cl.startswith('--- ') and i + 1 < n and lines[i + 1].startswith('+++ '):
            break
        if cl.startswith('+'):
            new_lines.append(cl[1:])
        elif cl.startswith('-'):
            old_lines.append(cl[1:])
        elif cl.startswith(' ') or cl.strip() == '':
            context = cl[1:] if cl.startswith(' ') else cl
            old_lines.append(context)
            new_lines.append(context)
        elif cl.startswith('\\ '):
            pass  # "\ No newline at end of file"
        else:
            break
        i += 1
    return i, old_lines, new_lines


def _parse_unified_hunks(body: str) -> tuple:
    """Parse @@ hunks from a unified diff body.

    Args:
        body: The diff body text (---/+++ headers + hunks).

    Returns:
        (hunks, detected_path, is_delete).
        Each hunk: {"start": int, "old_count": int, "old_lines": [str], "new_lines": [str]}
    """
    import re
    hunks = []
    lines = body.splitlines(keepends=True)
    i = 0
    n = len(lines)
    detected_path = None
    is_delete = False

    while i < n:
        line = lines[i]

        # Check for new file marker
        if line.startswith("new file mode"):
            is_delete = False
            i += 1
            continue

        # Extract file path from /dev/null detection
        if line.startswith("--- "):
            p = line[4:].split('\t')[0].strip()
            if p != "/dev/null":
                detected_path = p[2:] if p.startswith(("a/", "b/")) else p
            else:
                is_delete = False
            i += 1
            continue
        if line.startswith("+++ "):
            p = line[4:].split('\t')[0].strip()
            if p != "/dev/null":
                detected_path = p[2:] if p.startswith(("a/", "b/")) else p
            else:
                is_delete = True
            i += 1
            continue

        # Parse hunk header: @@ -start,count +start,count @@. A bare `@@` (no
        # line numbers) is the opencode "match anywhere" form and is stored with
        # start=-1 so the applier knows to search rather than use a line number.
        if line.startswith('@@'):
            m = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if m:
                start = int(m.group(1))
                old_count = int(m.group(2) or 1)
                i += 1

                i, old_lines, new_lines = _collect_hunk_lines(lines, i, n, old_count, int(m.group(4) or 1))

                hunks.append({"start": start, "old_count": old_count, "old_lines": old_lines, "new_lines": new_lines})
                continue

            # Bare @@: collect lines until the next hunk/header as a match-anywhere hunk.
            i += 1
            i, old_lines, new_lines = _collect_bare_hunk_lines(lines, i, n)
            hunks.append({"start": -1, "old_count": len(old_lines), "old_lines": old_lines, "new_lines": new_lines})
            continue

        i += 1

    return hunks, detected_path, is_delete


def _resolve_section_path(base_dir: str, path: str) -> str:
    """Resolve a patch section path and validate it stays inside the base dir.

    Args:
        base_dir: Realpath of the working directory.
        path: The patch's target path.

    Returns:
        The absolute path, or raises ValueError on traversal.
    """
    abspath = os.path.abspath(os.path.join(os.getcwd(), path))
    if os.path.commonpath([base_dir, os.path.realpath(abspath)]) != base_dir:
        raise ValueError(f"Path traversal denied: {path}")
    return abspath


def _apply_hunk(content: list, hunk: dict, path: str) -> tuple:
    """Apply a single hunk to a file's line list.

    Args:
        content: List of lines (mutated in place).
        hunk: {"start", "old_count", "old_lines", "new_lines"}.
        path: Target path (for skip messages).

    Returns:
        (applied: bool, message: Optional[str]).
    """
    start_idx = hunk["start"] - 1
    old_count = hunk["old_count"]
    new_lines = hunk["new_lines"]

    if start_idx < 0:
        # Bare @@ hunk: match the removed/context lines anywhere in the file.
        expected_clean = [line.rstrip('\r\n') for line in hunk.get("old_lines", [])]
        if not expected_clean:
            # Pure addition with no anchor: append at the end of the file.
            content.extend(new_lines)
            return True, None
        found = -1
        window_len = len(expected_clean)
        for k in range(len(content) - window_len + 1):
            window = [line.rstrip('\r\n') for line in content[k:k + window_len]]
            if window == expected_clean:
                found = k
                break
        if found < 0:
            return False, f"SKIPPED {path} hunk (no match for removed lines)"
        start_idx = found
        old_count = window_len

    if start_idx + old_count > len(content):
        old_count = len(content) - start_idx
    if old_count < 0:
        old_count = 0

    existing = content[start_idx:start_idx + old_count]
    expected = hunk.get("old_lines", [])
    existing_clean = [line.rstrip('\r\n') for line in existing]
    expected_clean = [line.rstrip('\r\n') for line in expected]
    if expected and existing_clean != expected_clean:
        return False, f"SKIPPED {path} hunk at line {hunk['start']} (context mismatch)"

    content[start_idx:start_idx + old_count] = new_lines
    return True, None


def _apply_unified_diff(patch_text: str) -> Dict:
    """Apply a unified diff patch to the filesystem. Pure Python.

    Args:
        patch_text: The diff text to apply.

    Returns:
        {"success": bool, "output": str, "error": str|None}.

    Supports standard unified diff (---/+++ headers, @@ hunks)
    and OpenCode-style markers (*** Add/Update/Delete/Move to: path).
    """
    sections = _parse_patch_sections(patch_text)
    if not sections:
        return {"success": False, "output": "", "error": "No valid patch sections found in patch_text"}

    applied = []
    base_dir = os.path.realpath(os.getcwd())
    for sec in sections:
        path = sec["path"]
        try:
            abspath = _resolve_section_path(base_dir, path)
        except ValueError as e:
            return {"success": False, "output": "", "error": str(e)}

        op = sec.get("operation", "modify")

        if op == "delete":
            if os.path.isfile(abspath):
                os.unlink(abspath)
            applied.append(f"Deleted {path}")
            continue

        if op == "move":
            dest = sec["destination"]
            try:
                absdest = _resolve_section_path(base_dir, dest)
            except ValueError as e:
                return {"success": False, "output": "", "error": str(e)}
            if os.path.isfile(abspath):
                os.makedirs(os.path.dirname(absdest), exist_ok=True)
                os.rename(abspath, absdest)
            applied.append(f"Moved {path} -> {dest}")
            continue

        # Read existing content or start empty for new files
        if os.path.isfile(abspath):
            with open(abspath, "r") as f:
                content = f.readlines()
        elif op == "add":
            content = []
        else:
            return {"success": False, "output": "", "error": f"File not found: {path}"}

        # Apply hunks in reverse order to preserve line numbers
        applied_hunks = 0
        for hunk in sorted(sec["hunks"], key=lambda h: h["start"], reverse=True):
            ok, message = _apply_hunk(content, hunk, path)
            if ok:
                applied_hunks += 1
            elif message:
                applied.append(message)

        if applied_hunks == 0 and sec["hunks"]:
            # Nothing matched — report failure so the model can retry instead of
            # silently "succeeding" without changing the file.
            return {"success": False, "output": "\n".join(applied),
                    "error": f"No hunks matched {path}; patch not applied"}

        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "w") as f:
            f.writelines(content)

        action = "Added" if op == "add" else "Patched"
        applied.append(f"{action} {path}")

    return {"success": True, "output": "\n".join(applied), "error": None}


def _tool_handle_apply_patch(self, args: dict) -> dict:
    """Apply a unified diff (or OpenCode markers) to the filesystem.

    Args:
        args: Tool arguments with "patch_text".

    Returns:
        {"success": bool, "output": str, "error": str|None}.
    """
    patch_text = args["patch_text"]
    if not patch_text.strip():
        return {"success": False, "output": "", "error": "patch_text must not be empty"}
    try:
        return _apply_unified_diff(patch_text)
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


# ============================================================================
# ============= TOOL REGISTRY                  ================================
# ============================================================================

TOOL_ARG_ALIASES = {
    "read_file": {"file_path": ["file", "path", "filename", "filepath"],
                          "offset": ["start", "start_line", "line"],
                          "limit": ["max_lines", "lines", "count"]},
    "write_file": {"file_path": ["file", "path", "filename", "filepath"], "content": ["file_content"]},
    "run_python": {"file_path": ["file", "path", "filename", "filepath"]},
    "run_command": {"command": ["cmd", "shell"]},
    "list_directory": {"path": ["directory", "dir"]},
    "edit_file": {"file_path": ["file", "path", "filename", "filepath"]},
    "apply_patch": {},
}

class ToolRegistry:
    """Registers and executes agentic tools with confirmation support.

    Call map:
      get_system_prompt_block() → AGENTIC_TOOL_DEFS
      execute() → _confirm() then handler
      list_tools_str() → AGENTIC_TOOL_DEFS
    """

    def __init__(self, ctx: Optional['CommandContext'] = None,
                 executor: Optional[Executor] = None) -> None:
        """Initialize the tool registry with context and executor.

        Args:
            ctx: Shared CommandContext (used for auto-confirm + path ACL).
            executor: Executor for run_python/run_command (defaults to a new one).
        """
        self._ctx = ctx
        self.executor = executor or Executor()
        self._handlers = {
            "fetch_url": _tool_handle_fetch_url,
            "read_file": _tool_handle_read_file,
            "write_file": _tool_handle_write_file,
            "list_directory": _tool_handle_list_directory,
            "glob": _tool_handle_glob,
            "run_python": _tool_handle_run_python,
            "run_command": _tool_handle_run_command,
            "diff": _tool_handle_diff,
            "patch": _tool_handle_patch,
            "edit_file": _tool_handle_edit_file,
            "apply_patch": _tool_handle_apply_patch,
        }

    def get_system_prompt_block(self) -> str:
        """Build tool definitions section for embedding in the agentic system prompt."""
        lines = ["## Available tools\n"]
        for name, defn in AGENTIC_TOOL_DEFS.items():
            lines.append(f"### {name}")
            lines.append(f"{defn['description']}\n")
            lines.append("Parameters:")
            props = defn["parameters"]["properties"]
            required = set(defn["parameters"].get("required", []))
            for pname, pdef in props.items():
                req_mark = " (REQUIRED)" if pname in required else ""
                lines.append(f"  - {pname} ({pdef['type']}): {pdef['description']}{req_mark}")
            lines.append(f"\nCall format: {{\"tool\": \"{name}\", \"arguments\": {{...}}}}\n")
        return "\n".join(lines)

    def list_tools_str(self) -> str:
        """Format the tool list for /listtool display."""
        lines = []
        for name, defn in AGENTIC_TOOL_DEFS.items():
            destructive = "! " if name in DESTRUCTIVE_TOOLS else "  "
            lines.append(f"{destructive}{name:<16} {defn['description']}")
        return "\n".join(lines)

    def _confirm(self, tool_name: str, args: dict) -> bool:
        """Ask the user before running a destructive tool (unless auto-confirm).

        Args:
            tool_name: Name of the tool to run.
            args: The tool's arguments (shown in the prompt).

        Returns:
            True if the tool may run.
        """
        if tool_name not in DESTRUCTIVE_TOOLS:
            return True
        if self._ctx and self._ctx.auto_confirm:
            return True
        args_display = ", ".join(f"{k}={v!r}" for k, v in args.items())
        prompt = f"\n[Agentic] Run {tool_name}({args_display})? [y/N] "
        try:
            reply = input(prompt).strip().lower()
            return reply in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def execute(self, tool_name: str, args: dict) -> dict:
        """Dispatch a tool call to its handler with alias normalization + confirmation.

        Args:
            tool_name: Name of the tool to execute.
            args: Tool argument dict.

        Returns:
            {"success": bool, "output": str, "error": str|None}.
        """
        if tool_name not in self._handlers:
            return {"success": False, "output": "", "error": f"Unknown tool '{tool_name}'"}
        # Normalize argument name aliases (e.g. "path" -> "file")
        if tool_name in TOOL_ARG_ALIASES:
            for canonical, aliases in TOOL_ARG_ALIASES[tool_name].items():
                if canonical not in args:
                    for alias in aliases:
                        if alias in args:
                            args[canonical] = args.pop(alias)
                            break
        if not self._confirm(tool_name, args):
            return {"success": False, "output": "", "error": "Cancelled by user"}
        try:
            return self._handlers[tool_name](self, args)
        except KeyError as e:
            return {"success": False, "output": "", "error": f"Missing required argument: {e}"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}


# ============================================================================
# ============= AGENTIC SESSION LOGGER          ================================
# ============================================================================

class AgenticLogger:
    """Logs agentic session turns to a structured JSONL file.

    Automatically cleans up log files older than AGENTIC_LOG_RETENTION_DAYS
    (default 1) on initialization to prevent unbounded disk growth.

    Call map:
      __init__() → _cleanup_old_logs()
      write() → appends JSONL line
      close() → flushes file
    """

    AGENTIC_LOG_RETENTION_DAYS = 1

    def __init__(self) -> None:
        """Open a fresh timestamped JSONL log after pruning old ones."""
        log_dir = os.path.expanduser("~/.ollamaquery.d/agentic")
        os.makedirs(log_dir, exist_ok=True)
        self._cleanup_old_logs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"{timestamp}.jsonl")
        self.file = open(self.path, "w", encoding="utf-8")

    def _cleanup_old_logs(self, log_dir: str) -> None:
        """Remove log files older than AGENTIC_LOG_RETENTION_DAYS."""
        cutoff = time.time() - (self.AGENTIC_LOG_RETENTION_DAYS * 86400)
        try:
            for fname in os.listdir(log_dir):
                fpath = os.path.join(log_dir, fname)
                if fname.endswith(".jsonl") and os.path.isfile(fpath):
                    if os.path.getmtime(fpath) < cutoff:
                        os.unlink(fpath)
        except OSError:
            pass

    def write(self, **data: dict) -> None:
        """Append a JSONL entry with a timestamp.

        Args:
            **data: Arbitrary fields to log (type, iteration, model_response, ...).
        """
        data["timestamp"] = datetime.now().isoformat()
        self.file.write(json.dumps(data, default=str) + "\n")
        self.file.flush()

    def close(self) -> None:
        """Flush and close the log file."""
        self.file.close()


# ============================================================================
# ============= DEBUGGING CLASS    ===========================================
# ============================================================================

class DebugManager:
    """Manages per-category debug levels.

    Call map:
      set_level() → is_enabled() / should_log()
      get_status() → get_level()
    """

    CATEGORIES = {
        'network':    "HTTP requests/responses to LLM server",
        'payload':    "Full JSON payloads sent to server",
        'response':   "Response content and chunks from server",  # ← ADD THIS
        'stream':     "Streaming chunks received from server",
        'context':    "Token estimation and message context details",
        'thinking':   "Thinking/reasoning block extraction",
        'commands':   "Command processing internals",
        'urlfetch':   "URL fetching and HTML conversion",
        'all':        "Master toggle for everything",
    }


    VALID_LEVELS = {
        'off': 0,
        'basic': 1,
        'verbose': 2,
        'trace': 3
    }

    def __init__(self) -> None:
        """Initialize per-category debug levels to off."""
        # Each category stores its own level
        self._levels: Dict[str, int] = {cat: 0 for cat in self.CATEGORIES}

    def is_enabled(self, category: str) -> bool:
        """Quick check if any debugging is active for this category."""
        return self.get_level(category) > 0


    def set_level(self, category: str, level: str) -> bool:
        """Set debug level for a category. Returns True if valid.

        Args:
            category: Debug category name (or 'all').
            level: One of off/basic/verbose/trace.

        Returns:
            True if the level was applied.
        """
        if category not in self.CATEGORIES:
            return False
        if level.lower() not in self.VALID_LEVELS:
            return False

        level_int = self.VALID_LEVELS[level.lower()]

        if category == 'all':
            for cat in self.CATEGORIES:
                self._levels[cat] = level_int
        else:
            self._levels[category] = level_int
        return True

    def get_level(self, category: str) -> int:
        """Get current level. All respects the 'all' category master."""
        master = self._levels.get('all', 0)
        specific = self._levels.get(category, 0)
        return max(master, specific)

    def should_log(self, category: str, min_level: int = 1) -> bool:
        """Check if a debug message should be emitted.

        Args:
            category: Debug category name.
            min_level: Minimum level required to emit.
        """
        return self.get_level(category) >= min_level


    def get_status(self) -> dict:
        """Return current state for status display."""
        return {
            cat: level
            for cat, level in self._levels.items()
            if level > 0 or cat == 'all'
        }

# ============= DEBUG LOG function ============================================



def debug_log(debug_mgr: 'DebugManager', category: str, level: int, message: str,
              data: object = None, prefix: str = "DEBUG") -> None:
    """
    Central debug logging function.

    Args:
        debug_mgr: The DebugManager instance
        category: Which subsystem this belongs to
        level: Minimum level required (1=basic, 2=verbose, 3=trace)
        message: Human-readable description
        data: Optional structured data to format
        prefix: Label for the output line
    """
    if not debug_mgr.should_log(category, level):
        return

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # Build output
    output = f"[{prefix}:{category}:{timestamp}] {message}"

    if data is not None and debug_mgr.get_level(category) >= 2:
        if isinstance(data, (dict, list)):
            formatted = json.dumps(data, indent=2, default=str)
            output += f"\n{formatted}"
        elif isinstance(data, bytes) and debug_mgr.get_level(category) >= 3:
            # Trace level: show hex dump for binary data
            output += f"\n{data[:500]!r}"
        else:
            output += f" | {data}"

    sys.stderr.write(colorize(f"{output}\n", 'muted'))


# ============================================================================
# ============= MODEL QUERY CLASS ============================================
# ============================================================================

class ModelQuery:
    """Unified query handler for both Ollama and Llama.cpp backends.

    Call map:
      query_stream() → _build_stream_request() → build_request_payload()
                      → _parse_chunk() → _normalize_llamacpp_usage()
                      → _iter_stream_lines()
                      → _update_context_tokens()
      query_sync() → build_request_payload() → _inject_images_into_messages()
                   → _get_chat_url()
      build_request_payload() / _build_stream_request() → _get_chat_url()
      calculate_stats() → print_stats_display()
    """

    def __init__(self, base_url: Optional[str] = None, backend: Optional[str] = None,
                 context: Optional['CommandContext'] = None) -> None:
        """Initialize the query handler with a context (or legacy connection args).

        Args:
            base_url: Legacy: backend base URL (when context is None).
            backend: Legacy: backend name (when context is None).
            context: Shared CommandContext; preferred over base_url/backend.
        """
        if context is not None:
            self.ctx = context
        elif base_url is not None:
            # Legacy compatibility: create context from args if needed
            self.ctx = CommandContext()
            self.ctx.base_url = base_url
            self.ctx.backend = backend or self.ctx.backend
        else:
            # Fallback: use global CommandContext
            self.ctx = CommandContext()

    def _debug_request(self, url: str, payload: dict, headers: dict = None) -> None:
        """Log outgoing request to LLM server."""
        debug_mgr = self.ctx.debug_manager

        if not debug_mgr or not payload:
            return

        if headers is None:
            headers = {'Content-Type': 'application/json'}

        if debug_mgr.should_log('network', 1):
            try:
                payload_size = len(json.dumps(payload))
                debug_log(debug_mgr, 'network', 1,
                         f"POST {url} ({payload_size} bytes)",
                         prefix="HTTP→")
            except Exception:
                pass  # Silently fail debug logging

        if debug_mgr.should_log('payload', 1):
            try:
                # Create a safe copy for logging (mask large base64 data)
                safe_payload = self._mask_payload(payload)
                model_name = payload.get('model', '?') if isinstance(payload, dict) else '?'
                msg_count = len(payload.get('messages', [])) if isinstance(payload, dict) else 0

                debug_log(debug_mgr, 'payload', 1,
                         f"Sending to model '{model_name}' | {msg_count} messages",
                         safe_payload,
                         prefix="PAYLOAD")
            except Exception:
                pass  # Silently fail debug logging

    def _debug_response_chunk(self, chunk: dict, is_first: bool = False, is_final: bool = False, *args: tuple, **kwargs: dict) -> None:
        """Log incoming streaming chunk.

        Args:
            chunk: The raw chunk dict.
            is_first: Whether this is the first chunk of the stream.
            is_final: Whether this is the final chunk.
            *args, **kwargs: Ignored (API compatibility).
        """
        debug_mgr = self.ctx.debug_manager

        if not debug_mgr or chunk is None:
            return

        try:
            if is_first and debug_mgr.should_log('network', 1):
                debug_log(debug_mgr, 'network', 1, "Stream started", prefix="HTTP←")

            if debug_mgr.should_log('stream', 1) and is_final:
                debug_log(debug_mgr, 'stream', 1, "Stream completed", prefix="HTTP←")

            # --- ADD: Response content debugging ---
            if debug_mgr.should_log('response', 1):
                message = chunk.get('message', {}) if isinstance(chunk, dict) else {}
                content = message.get('content', '') if isinstance(message, dict) else ''
                thought = message.get('thought', '') or message.get('thinking', '') if isinstance(message, dict) else ''

                if content or thought:
                    debug_log(debug_mgr, 'response', 1,
                             f"Content: '{content[:50]}...' " if content else "Thinking block",
                             prefix="RESP")
            # --------------------------------------

            if debug_mgr.should_log('stream', 2) and not is_final:
                # Show chunk structure without flooding the terminal
                message = chunk.get('message', {}) if isinstance(chunk, dict) else {}
                content = message.get('content', '') if isinstance(message, dict) else ''

                if content and isinstance(content, str):
                    preview = content[:100] + ('...' if len(content) > 100 else '')
                    debug_log(debug_mgr, 'stream', 2,
                             f"Content chunk: '{preview}'", prefix="CHUNK")
        except Exception:
            pass  # Silently fail debug logging

    def _debug_final_stats(self, usage_stats: dict) -> None:
        """Log final usage statistics from server."""
        debug_mgr = self.ctx.debug_manager

        if not debug_mgr or usage_stats is None:
            return

        try:
            if debug_mgr.should_log('network', 1):
                debug_log(debug_mgr, 'network', 1,
                         "Response complete with usage stats",
                         usage_stats, prefix="HTTP←")
        except Exception:
            pass  # Silently fail debug logging

    def _mask_payload(self, payload: dict) -> dict:
        """Replace large binary data with size indicators for logging.

        Args:
            payload: The request payload to sanitize.

        Returns:
            A deep copy of the payload with images replaced by size indicators.
        """
        if payload is None:
            return {}

        try:
            # Deep copy to avoid modifying the original
            safe = json.loads(json.dumps(payload))

            if isinstance(safe, dict):
                for msg in safe.get('messages', []):
                    if isinstance(msg, dict) and 'images' in msg and msg['images']:
                        msg['images'] = [
                            f"<base64_image: {len(img)} bytes>" if isinstance(img, str) else "<binary_image>"
                            for img in msg['images']
                        ]

            return safe
        except Exception:
            return {"error": "Could not mask payload for logging"}




    @property
    def base_url(self) -> str:
        """The backend base URL from the shared context."""
        return self.ctx.base_url

    @property
    def backend(self) -> str:
        """The active backend name from the shared context."""
        return self.ctx.backend

    def _get_headers(self) -> dict:
        """Build request headers, injecting API key for cloud backends.

        Returns:
            Dict of HTTP headers for the request.
        """
        headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
        if self.backend in ("gemini", "opencodezen", "opencodego", "mistral", "deepseek") and self.ctx.api_key:
            headers['Authorization'] = f'Bearer {self.ctx.api_key}'
        return headers

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count. Delegates to CommandContext."""
        return self.ctx.estimate_tokens(text)

    def calculate_context_tokens(self, messages: list) -> int:
        """Calculate estimated total tokens in conversation context. Delegates to CommandContext."""
        return self.ctx.calculate_context_tokens(messages)

    @staticmethod
    def _extract_usage_timings(usage: dict) -> tuple:
        """Extract server-reported prefill/decode timings and rates.

        Handles Ollama nanosecond durations (prompt_eval_duration /
        eval_duration) and llama.cpp millisecond timings plus precomputed
        rates (prompt_ms / predicted_ms / prompt_per_second /
        predicted_per_second).

        Args:
            usage: Server usage dict.

        Returns:
            (prefill_sec, gen_sec, prefill_tps, gen_tps), each None if absent.
        """
        prefill_sec = gen_sec = prefill_tps = gen_tps = None
        pe_dur = usage.get("prompt_eval_duration") or 0   # Ollama nanoseconds
        ev_dur = usage.get("eval_duration") or 0
        if pe_dur > 0:
            prefill_sec = pe_dur / 1e9
        if ev_dur > 0:
            gen_sec = ev_dur / 1e9
        if usage.get("prompt_ms"):
            prefill_sec = float(usage["prompt_ms"]) / 1000.0
        if usage.get("predicted_ms"):
            gen_sec = float(usage["predicted_ms"]) / 1000.0
        if usage.get("prompt_per_second"):
            prefill_tps = float(usage["prompt_per_second"])
        if usage.get("predicted_per_second"):
            gen_tps = float(usage["predicted_per_second"])
        return prefill_sec, gen_sec, prefill_tps, gen_tps

    @staticmethod
    def _client_phase_seconds(phase_times: Optional[dict]) -> tuple:
        """Derive wall-clock (prefill, thinking, answer) seconds from timestamps.

        Args:
            phase_times: Optional dict with start/first_token/first_content/end.

        Returns:
            (prefill_sec, thinking_sec, answer_sec), each None if underivable.
        """
        if not phase_times:
            return None, None, None
        start = phase_times.get("start")
        first = phase_times.get("first_token")
        first_content = phase_times.get("first_content")
        end = phase_times.get("end")
        prefill = max(first - start, 0.0) if (start is not None and first is not None) else None
        thinking = max(first_content - first, 0.0) if (first is not None and first_content is not None) else None
        if first_content is not None and end is not None:
            answer = max(end - first_content, 0.0)
        elif first is not None and end is not None:
            answer = max(end - first, 0.0)
        else:
            answer = None
        return prefill, thinking, answer

    def _split_generation_tokens(self, content: str, thought: str, eval_count: int) -> tuple:
        """Split generated tokens into (thinking, answer) via a length ratio.

        The server reports one combined generated-token count (thinking +
        answer). The ratio is estimated from text length and scaled to the
        server total so the two parts always sum to `eval_count`.

        Args:
            content: Visible answer text.
            thought: Thinking/reasoning text.
            eval_count: Combined server-reported generated tokens.

        Returns:
            (think_tokens, answer_tokens).
        """
        if not thought:
            return 0, eval_count
        think_est = self.estimate_tokens(thought)
        answer_est = self.estimate_tokens(content)
        total_est = think_est + answer_est
        if eval_count > 0 and total_est > 0:
            think_tokens = min(int(round(eval_count * think_est / total_est)), eval_count)
            answer_tokens = eval_count - think_tokens
            # Never round a present answer down to zero.
            if answer_est > 0 and answer_tokens == 0:
                answer_tokens = 1
                think_tokens = max(eval_count - 1, 0)
            return think_tokens, answer_tokens
        return think_est, answer_est

    def calculate_stats(self, total_time: float, content: str, usage: Optional[dict] = None,
                        messages: Optional[list] = None, thought: str = "",
                        phase_times: Optional[dict] = None) -> dict:
        """Calculate per-query stats with prefill, thinking and answer rates.

        Timing is hybrid: server-reported durations (Ollama/llama.cpp) are
        preferred, with client wall-clock phase timestamps (time-to-first-token,
        first-content, stream end) as fallback for backends that report none.

        Args:
            total_time: Wall-clock seconds for the whole query.
            content: The visible response text.
            usage: Optional server usage dict.
            messages: Optional message list for context-token fallback.
            thought: Accumulated thinking/reasoning text.
            phase_times: Optional client phase timestamps dict.

        Returns:
            dict of eval_count/prompt_eval_count/total_context_tokens/total_time/
            tps/prefill_tps/think_tokens/think_tps/answer_tokens/answer_tps/
            gen_sec/content_length.
        """
        usage = usage or {}
        eval_count = usage.get("completion_tokens", 0) or usage.get("eval_count", 0)
        prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("prompt_eval_count", 0)
        total_context_tokens = usage.get("total_tokens", 0) or (prompt_tokens + eval_count)

        if not total_context_tokens and messages:
            total_context_tokens = self.calculate_context_tokens(messages)

        if not eval_count and (content or thought):
            eval_count = self.estimate_tokens(content) + self.estimate_tokens(thought)

        srv_prefill_sec, srv_gen_sec, srv_prefill_tps, srv_gen_tps = self._extract_usage_timings(usage)
        cli_prefill, cli_think, cli_answer = self._client_phase_seconds(phase_times)

        prefill_sec = srv_prefill_sec if srv_prefill_sec is not None else cli_prefill
        prefill_tps = srv_prefill_tps
        if prefill_tps is None and prefill_sec and prompt_tokens > 0:
            prefill_tps = prompt_tokens / prefill_sec

        think_tokens, answer_tokens = self._split_generation_tokens(content, thought, eval_count)

        # Decode window: server-reported total (accurate) is apportioned between
        # thinking and answer by token ratio. Thinking and answer tokens share the
        # same decode loop, so their rate equals the overall decode rate; the
        # useful split is the token counts. Without server timing, fall back to
        # the client-measured answer/thinking phase windows.
        if srv_gen_sec is not None and eval_count > 0:
            gen_sec = srv_gen_sec
            think_sec = srv_gen_sec * (think_tokens / eval_count)
            answer_sec = srv_gen_sec * (answer_tokens / eval_count)
        elif cli_answer is not None and cli_answer >= 0.05:
            # A near-zero client window means a single non-streamed chunk, not
            # a real decode phase — avoid fabricating an infinite rate.
            think_sec = cli_think if cli_think is not None else 0.0
            answer_sec = cli_answer
            gen_sec = think_sec + answer_sec
        else:
            gen_sec = think_sec = answer_sec = None

        think_tps = think_tokens / think_sec if think_sec and think_tokens > 0 else 0.0
        answer_tps = answer_tokens / answer_sec if answer_sec and answer_tokens > 0 else 0.0

        # Overall decode rate excluding prefill (server rate > server window > client).
        if srv_gen_tps is not None:
            tps = srv_gen_tps
        elif gen_sec and eval_count > 0:
            tps = eval_count / gen_sec
        else:
            tps = eval_count / total_time if eval_count > 0 and total_time > 0 else 0.0

        # With no thinking, the answer rate equals the overall decode rate.
        if answer_tps == 0.0 and think_tokens == 0 and answer_tokens > 0 and tps > 0:
            answer_tps = tps

        return {
            "eval_count": eval_count,
            "prompt_eval_count": prompt_tokens,
            "total_context_tokens": total_context_tokens,
            "total_time": total_time,
            "tps": tps,
            "prefill_tps": prefill_tps or 0.0,
            "prefill_sec": prefill_sec or 0.0,
            "gen_sec": gen_sec or 0.0,
            "think_tokens": think_tokens,
            "think_tps": think_tps,
            "answer_tokens": answer_tokens,
            "answer_tps": answer_tps,
            "content_length": len(content)
        }


    def print_stats_display(self, stats: dict) -> None:
        """Print formatted stats to stderr.

        Shows prefill (prompt-eval) throughput, thinking throughput and
        answer-only throughput so thinking tokens no longer dilute the
        visible generation rate.

        Args:
            stats: Stats dict from calculate_stats().
        """
        if not stats:
            return

        parts = [f"{stats['total_time']:.2f}s total"]

        prompt_tokens = stats.get("prompt_eval_count", 0)
        if prompt_tokens > 0 and stats.get("prefill_tps", 0.0) > 0:
            parts.append(f"prefill {prompt_tokens} tok @ {stats['prefill_tps']:.1f} t/s")
        if stats.get("think_tokens", 0) > 0:
            parts.append(f"think {stats['think_tokens']} tok @ {stats.get('think_tps', 0.0):.1f} t/s")

        if stats.get("eval_count", 0) > 0:
            answer_tokens = stats.get("answer_tokens", stats["eval_count"])
            if answer_tokens > 0:
                parts.append(f"answer {answer_tokens} tok @ {stats.get('answer_tps', 0.0):.1f} t/s")
            else:
                parts.append(f"gen {stats['eval_count']} tok @ {stats.get('tps', 0.0):.1f} t/s")
            ctx = self.ctx.current_context_tokens
            if not ctx:
                ctx = stats.get("total_context_tokens", 0)
            if not ctx:
                ctx = stats.get("prompt_eval_count", 0) + stats['eval_count']
            parts.append(f"Context: {ctx} tokens")
        else:
            parts.append(f"Content: {stats.get('content_length', 0)} chars")

        sys.stderr.write(colorize(f"\n--- Stats: {' | '.join(parts)} ---\n", 'muted'))


    @staticmethod
    def _inject_images_into_messages(messages: list, images: Optional[list], backend: str) -> None:
        """Inject image data into the last user message, mutating in-place.

        Args:
            messages: Message list (mutated in place).
            images: List of base64 image strings.
            backend: Backend name for shape selection.

        Ollama backend: sets messages[-1]["images"] = images list.
        OpenAI-compatible backends (llamacpp, lmstudio): embeds images as
        content parts with data:image URIs.
        """
        if not images or not messages or messages[-1].get("role") != "user":
            return
        if backend == "ollama":
            messages[-1]["images"] = images
        else:
            text = messages[-1].get("content", "")
            content_parts = [{"type": "text", "text": text or "Describe this image"}]
            for img in images:
                mime = guess_image_mime(img)
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{img}"}})
            messages[-1]["content"] = content_parts

    @staticmethod
    def _apply_backend_inference_params(payload: dict, backend: str, kwargs: dict) -> None:
        """Copy the backend-supported inference params from kwargs into the payload.

        Args:
            payload: The payload dict (mutated in place).
            backend: Backend name (controls which params are supported).
            kwargs: The kwargs dict (images/tools/params/etc.).
        """
        if backend == "llamacpp":
            params = ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty"]
        elif backend == "lmstudio":
            params = ["temperature", "top_p", "presence_penalty", "repeat_penalty"]
        elif backend in ("gemini", "opencodezen", "opencodego"):
            params = ["temperature", "top_p", "top_k"]
        elif backend in ("mistral", "deepseek"):
            params = ["temperature", "top_p"]
        else:
            params = []
        for param in params:
            if param in kwargs:
                payload[param] = kwargs[param]

    @staticmethod
    def _apply_no_thinking(payload: dict, backend: str) -> None:
        """Disable the model's reasoning phase via the backend template flag.

        Reasoning models ignore prompt-level instructions, so the thinking phase
        must be turned off through the chat-template flag. Each local backend and
        model family exposes it differently:
          - ollama: /api/chat `options.think = false`
          - llama.cpp / lmstudio Qwen-style templates: `chat_template_kwargs`
            (llama.cpp uses `thinking`, vLLM uses `enable_thinking`; both keys
            are set so either is honored, unknown keys are ignored by templates)
          - gpt-oss: has NO reasoning off-switch — the model is trained to always
            reason (llama.cpp: "incapable of not reasoning"). Only
            `reasoning_effort` (low/medium/high) is supported, so `/thinkingoff`
            downgrades it to "low" to minimize the analysis phase.

        Args:
            payload: The payload dict (mutated in place).
            backend: Backend name.
        """
        if backend == "ollama":
            payload["options"] = {**payload.get("options", {}), "think": False}
        elif backend in ("llamacpp", "lmstudio"):
            kwargs = dict(payload.get("chat_template_kwargs") or {})
            model = str(payload.get("model", "")).lower()
            if "gpt-oss" in model or "gpt_oss" in model:
                kwargs["reasoning_effort"] = "low"
            else:
                kwargs["enable_thinking"] = False
                kwargs["thinking"] = False
            payload["chat_template_kwargs"] = kwargs

    @staticmethod
    def _apply_reasoning_effort(payload: dict, backend: str, effort: str) -> None:
        """Set the model's reasoning effort via the backend-specific field.

        Templates that support reasoning levels (gpt-oss, newer Qwen3) read
        `reasoning_effort` (low/medium/high); templates that don't silently
        ignore it. Ollama exposes it through `options`, the OpenAI-compatible
        backends through `chat_template_kwargs`.

        Args:
            payload: The payload dict (mutated in place).
            backend: Backend name.
            effort: Reasoning effort level ("low", "medium" or "high").
        """
        if backend == "ollama":
            payload["options"] = {**payload.get("options", {}), "reasoning_effort": effort}
        elif backend in ("llamacpp", "lmstudio"):
            kwargs = dict(payload.get("chat_template_kwargs") or {})
            kwargs["reasoning_effort"] = effort
            payload["chat_template_kwargs"] = kwargs

    def reasoning_flags(self, model: str, backend: str, force_no_thinking: bool,
                        reasoning_effort: Optional[str] = None) -> dict:
        """Return the payload fields that would be sent for a reasoning state.

        Used by `/reasoning status` so the user can see the exact backend flag
        the next query will carry (e.g. `options.think` vs
        `chat_template_kwargs.reasoning_effort`), rather than only a label.

        Args:
            model: Model name (gpt-oss vs Qwen-style select the flag shape).
            backend: Backend name.
            force_no_thinking: Whether reasoning suppression is requested.
            reasoning_effort: Explicit effort level ("low"/"medium"/"high").
                Takes precedence over `force_no_thinking`.

        Returns:
            dict of payload fields (without "model"), empty when no flag applies.
        """
        payload = {"model": model}
        if reasoning_effort:
            self._apply_reasoning_effort(payload, backend, reasoning_effort)
        elif force_no_thinking:
            self._apply_no_thinking(payload, backend)
        return {k: v for k, v in payload.items() if k != "model"}

    def _apply_size_options(self, payload: dict, backend: str, kwargs: dict) -> None:
        """Apply warmup/context-size options to the payload.

        Args:
            payload: The payload dict (mutated in place).
            backend: Backend name.
            kwargs: The kwargs dict.
        """
        if kwargs.get('is_warmup'):
            if backend == "ollama":
                payload["options"] = {"num_predict": 1}
            elif backend in ("llamacpp", "lmstudio", "gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
                payload["max_tokens"] = 1
        elif context_size := kwargs.get('context_size'):
            if backend == "ollama":
                payload["options"] = {"num_ctx": context_size}
            elif backend in ("llamacpp", "lmstudio", "gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
                payload["max_tokens"] = context_size

    def build_request_payload(self, messages: list, model: str, stream_enabled: bool = False, **kwargs: dict) -> dict:
        """Build request payload for the backend.

        Args:
            messages: Message list to send.
            model: Model name.
            stream_enabled: Whether to request streaming.
            **kwargs: images, tools, context_size, is_warmup, inference params.

        Returns:
            The JSON payload dict.
        """
        self._inject_images_into_messages(messages, kwargs.get('images'), self.backend)

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream_enabled
        }

        self._apply_size_options(payload, self.backend, kwargs)

        if tools := kwargs.get('tools'):
            payload["tools"] = tools

        self._apply_backend_inference_params(payload, self.backend, kwargs)

        if kwargs.get('no_thinking'):
            self._apply_no_thinking(payload, self.backend)

        if effort := kwargs.get('reasoning_effort'):
            self._apply_reasoning_effort(payload, self.backend, str(effort))

        return payload

    def _get_chat_url(self, backend: str) -> str:
        """Return the chat API URL for the given backend.

        Args:
            backend: Backend name ("ollama" uses /api/chat, others use /v1/chat/completions).
        """
        return f"{self.base_url}/api/chat" if backend == "ollama" else f"{self.base_url}/v1/chat/completions"

    def query_sync(self, messages: list, model: str, stream_enabled: bool = False, **kwargs: dict) -> dict:
        """Non-streaming sync query wrapper.

        Args:
            messages: Message list to send.
            model: Model name.
            stream_enabled: Unused (always False for sync).
            **kwargs: Passed to build_request_payload.

        Returns:
            Parsed JSON response dict.
        """
        payload = self.build_request_payload(messages, model, stream_enabled=False, **kwargs)

        try:
            url = self._get_chat_url(self.backend)

            data = json.dumps(payload).encode('utf-8')
            req = Request(url, data=data, headers=self._get_headers())
            self._debug_request(url, payload)

            with _request_with_retry(req) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            msg = f"[ERROR] Sync query failed: {e}"
            if isinstance(e, HTTPError) and e.code == 403:
                if self.backend in ("gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
                    env_hint = {"gemini": "GEMINI_API_KEY", "opencodezen": "OPENCODEZEN_API_KEY", "opencodego": "OPENCODEGO_API_KEY", "mistral": "MISTRAL_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
                    msg += f"\n[HINT] {self.backend} authentication failed. Check your --api-key or {env_hint.get(self.backend, 'API_KEY')} env var."
                elif ":cloud" in model:
                    msg += "\n[HINT] Cloud models require authentication. Check your Ollama cloud API key or pull a local model instead."
            sys.stderr.write(colorize(msg, 'error'))
            return {"error": {"message": str(e), "type": type(e).__name__}}

    @staticmethod
    def _accumulate_stream_tool_calls(tool_calls: list, tool_call_index: dict, new_calls: list) -> None:
        """Merge incremental tool-call chunks into `tool_calls` and `tool_call_index`.

        Args:
            tool_calls: List to append full-object tool calls to.
            tool_call_index: Dict mapping index → partial tool call dict.
            new_calls: The new tool calls from a chunk.

        Handles both OpenAI-compatible deltas (arguments accumulate as strings, keyed by
        `index`) and Ollama full-object tool calls (no `index`, arguments already a dict).
        """
        for tc in new_calls:
            if "index" not in tc:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    }
                })
                continue
            idx = tc.get("index", 0)
            if idx not in tool_call_index:
                tool_call_index[idx] = {
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    }
                }
            else:
                existing = tool_call_index[idx]
                if tc.get("id"):
                    existing["id"] = tc["id"]
                fn_delta = tc.get("function") or {}
                if fn_delta.get("name"):
                    existing["function"]["name"] = fn_delta["name"]
                if "arguments" in fn_delta and fn_delta["arguments"]:
                    existing["function"]["arguments"] += fn_delta["arguments"]

    def _synthesize_sync_response(self, content: str, thinking: str, tool_calls: list, usage: dict) -> dict:
        """Build a non-streaming-shaped response dict from aggregated stream chunks.

        Args:
            content: Aggregated assistant text.
            thinking: Aggregated reasoning/thinking text.
            tool_calls: Accumulated native tool calls.
            usage: Aggregated usage dict.

        Returns:
            A sync-shaped response dict for the current backend.

        Mirrors the shape returned by `query_sync` for the current backend so callers
        (e.g. the ReAct loop) can parse it identically.
        """
        backend = self.backend
        if backend == "ollama":
            resp = {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
            }
            if thinking:
                resp["message"]["reasoning_content"] = thinking
            resp["prompt_eval_count"] = usage.get("prompt_eval_count", 0)
            resp["eval_count"] = usage.get("eval_count", 0)
            return resp
        resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "stop",
            }],
        }
        if thinking:
            resp["choices"][0]["message"]["reasoning_content"] = thinking
        if usage:
            resp["usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        return resp

    def _aggregate_sync_stream(self, req: Request, backend: str, socket_timeout: float,
                               on_chunk: object, cancel: Optional[dict] = None) -> tuple:
        """Read a streaming response and aggregate content/thinking/tools/usage.

        Args:
            req: The prepared urllib Request.
            backend: Backend name.
            socket_timeout: Socket timeout for the request.
            on_chunk: Optional per-chunk callback (thought, content, is_final).
            cancel: Optional {"event", "close"} cancel token dict.

        Returns:
            (full_content, full_thinking, tool_calls, tool_call_index, usage),
            or None when the request was cancelled.
        """
        full_content = ""
        full_thinking = ""
        tool_calls = []          # ordered, deduped accumulated tool calls
        tool_call_index = {}     # index -> partial tool call dict
        usage = {}
        try:
            with _request_with_retry(req, timeout=socket_timeout) as response:
                if cancel is not None and hasattr(response, 'close'):
                    # Expose a close callback so a ^C (or timeout retry) on the
                    # calling thread can abort this blocked stream read promptly.
                    cancel["close"] = response.close
                for raw_line in self._iter_stream_lines(response, backend):
                    if cancel is not None and cancel["event"].is_set():
                        return None
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    thought, content, is_final, u, tcs = self._parse_chunk(chunk, backend)
                    if thought:
                        full_thinking += thought
                    if content:
                        full_content += content
                    if tcs:
                        self._accumulate_stream_tool_calls(tool_calls, tool_call_index, tcs)
                    if u:
                        usage.update(u)
                    if on_chunk:
                        on_chunk(thought, content, is_final)
        except Exception as e:
            if cancel is not None and cancel["event"].is_set():
                # User aborted (^C) or a timeout retry superseded this request —
                # return quietly instead of spamming an error line.
                return None
            sys.stderr.write(colorize(f"[ERROR] {backend} sync stream failed: {e}\n", 'error'))
            return {"error": {"message": str(e), "type": type(e).__name__}}
        return full_content, full_thinking, tool_calls, tool_call_index, usage

    def query_sync_stream(self, messages: list, model: str, stream_enabled: bool = True, **kwargs: dict) -> dict:
        """Streaming variant of `query_sync` that aggregates chunks into a single dict.

        Performs a streaming request and synthesizes a non-streaming-shaped response
        dict (same as `query_sync`) so callers can parse it identically. Optionally
        invokes `kwargs['on_chunk'](thought, content, is_final)` per chunk for live
        feedback (e.g. real-time thinking/heartbeat display). Unlike `query_stream`,
        it does NOT write anything to stdout — display is the caller's responsibility.

        Args:
            messages: Conversation list.
            model: Model name.
            on_chunk: Optional callback invoked per parsed chunk.
            context_size, images, tools, inference params: passed through to payload.

        Returns:
            dict shaped like `query_sync` (or `{"error": {...}}` on failure).
        """
        on_chunk = kwargs.pop("on_chunk", None)
        images = kwargs.pop("images", None)
        context_size = kwargs.pop("context_size", None)
        socket_timeout = kwargs.pop("timeout", 120)
        cancel = kwargs.pop("cancel", None)
        backend = self.backend

        if images and messages and messages[-1].get("role") == "user":
            messages[-1] = dict(messages[-1])
            self._inject_images_into_messages(messages, images, backend)

        api_url, payload, headers = self._build_stream_request(
            backend, messages, model, True, context_size, **kwargs)

        data = json.dumps(payload).encode('utf-8')
        req = Request(api_url, data=data, headers=headers)
        self._debug_request(api_url, payload)

        result = self._aggregate_sync_stream(req, backend, socket_timeout, on_chunk, cancel)
        if isinstance(result, dict):  # error dict
            return result
        if result is None:  # cancelled
            return {"error": {"message": "cancelled", "type": "KeyboardInterrupt"}}
        full_content, full_thinking, tool_calls, tool_call_index, usage = result

        for idx in sorted(tool_call_index):
            tc = tool_call_index[idx]
            tool_calls.append({
                "id": tc["id"],
                "type": tc["type"],
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
            })

        return self._synthesize_sync_response(full_content, full_thinking, tool_calls, usage)


    def _normalize_llamacpp_usage(self, chunk: dict) -> dict:
        """
        Extract token counts from any Llama.cpp chunk format.
        Returns dict with standardized keys: total_tokens, prompt_tokens, completion_tokens
        """
        usage = {}

        # Format 1: Standard OpenAI-style usage block
        if "usage" in chunk and chunk["usage"]:
            u = chunk["usage"]
            usage["prompt_tokens"] = u.get("prompt_tokens", 0)
            usage["completion_tokens"] = u.get("completion_tokens", 0)
            usage["total_tokens"] = u.get("total_tokens",
                                           usage["prompt_tokens"] + usage["completion_tokens"])

        # Format 2: Llama.cpp timings block (your server uses this)
        if "timings" in chunk:
            t = chunk["timings"]
            usage["prompt_tokens"] = t.get("prompt_n", 0)
            usage["completion_tokens"] = t.get("predicted_n", 0)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            # Server-measured phase timings/rates (exclude prefill from decode rate)
            if t.get("prompt_ms"):
                usage["prompt_ms"] = t["prompt_ms"]
            if t.get("predicted_ms"):
                usage["predicted_ms"] = t["predicted_ms"]
            if t.get("prompt_per_second"):
                usage["prompt_per_second"] = t["prompt_per_second"]
            if t.get("predicted_per_second"):
                usage["predicted_per_second"] = t["predicted_per_second"]

        # Format 4: LM Studio stats block (tokens_per_second / time_to_first_token)
        if "stats" in chunk and chunk["stats"]:
            s = chunk["stats"]
            if s.get("tokens_per_second"):
                usage["predicted_per_second"] = s["tokens_per_second"]
            if s.get("time_to_first_token"):
                usage.setdefault("prompt_ms", s["time_to_first_token"] * 1000.0)

        # Format 3: Root-level fields (some versions)
        if "prompt_eval_count" in chunk:
            usage["prompt_tokens"] = usage.get("prompt_tokens", 0) or chunk.get("prompt_eval_count", 0)
        if "eval_count" in chunk:
            usage["completion_tokens"] = usage.get("completion_tokens", 0) or chunk.get("eval_count", 0)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return usage if usage else None

    def _build_stream_request(self, backend: str, messages: list, model: str,
                              stream_enabled: bool, context_size: Optional[int], **kwargs: dict) -> tuple:
        """Build URL, payload and headers for the backend.

        Args:
            backend: Backend name.
            messages: Message list to send.
            model: Model name.
            stream_enabled: Whether streaming was requested.
            context_size: Context window size (or None).
            **kwargs: Extra payload args.

        Returns:
            (api_url, payload, headers) tuple.
        """
        payload = self.build_request_payload(messages, model, stream_enabled=stream_enabled, context_size=context_size, **kwargs)
        return self._get_chat_url(backend), payload, self._get_headers()

    def _parse_chunk(self, chunk: dict, backend: str) -> tuple:
        """Extract (thought, content, is_final, usage, tool_calls) from a chunk for any backend.

        Args:
            chunk: A parsed JSON chunk dict.
            backend: Backend name for shape selection.

        Returns:
            (thought, content, is_final, usage, tool_calls) tuple.
        """
        if backend == "ollama":
            thought = (chunk.get("message", {}).get("thought") or
                       chunk.get("message", {}).get("thinking")) or ""
            content = chunk.get("message", {}).get("content", "")
            is_final = chunk.get("done", False)
            usage = None
            tool_calls = chunk.get("message", {}).get("tool_calls", [])
            if is_final:
                if "usage" in chunk:
                    usage = chunk["usage"]
                else:
                    usage = {
                        "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                        "eval_count": chunk.get("eval_count", 0),
                        "prompt_eval_duration": chunk.get("prompt_eval_duration", 0),
                        "eval_duration": chunk.get("eval_duration", 0),
                        "total_duration": chunk.get("total_duration", 0),
                        "load_duration": chunk.get("load_duration", 0),
                    }
            return thought, content, is_final, usage, tool_calls
        elif backend in ("llamacpp", "gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
            choices = chunk.get("choices", [])
            is_final = bool(choices and choices[0].get("finish_reason") is not None)
            delta = choices[0].get("delta", {}) if choices else {}
            thought = delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            tool_calls = delta.get("tool_calls", [])
            usage = self._normalize_llamacpp_usage(chunk) if is_final else None
            return thought, content, is_final, usage, tool_calls
        elif backend == "lmstudio":
            choices = chunk.get("choices", [])
            is_final = bool(choices and choices[0].get("finish_reason") is not None)
            delta = choices[0].get("delta", {}) if choices else {}
            thought = delta.get("reasoning") or ""
            content = delta.get("content") or ""
            tool_calls = delta.get("tool_calls", [])
            usage = self._normalize_llamacpp_usage(chunk) if is_final else None
            return thought, content, is_final, usage, tool_calls
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _iter_stream_lines(self, response: object, backend: str) -> tuple:
        """Yield decoded JSON lines from streaming response, stripping SSE prefixes.

        Args:
            response: The streaming HTTP response object.
            backend: Backend name (SSE stripping applies to non-ollama backends).
        """
        for line in response:
            decoded = line.decode('utf-8').strip()
            if not decoded:
                continue
            if backend in ("llamacpp", "lmstudio", "gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
                if decoded.startswith('data: '):
                    decoded = decoded[6:].strip()
            if decoded == '[DONE]':
                continue
            yield decoded

    def _update_context_tokens(self, backend: str, aggregated_usage: dict, messages: list) -> None:
        """Update context token tracking after stream completes.

        Args:
            backend: Backend name.
            aggregated_usage: Accumulated usage dict from chunks.
            messages: The message list (for recalc on KV-cache backends).
        """
        if backend == "ollama":
            total_tokens = (aggregated_usage.get("total_tokens", 0) or
                           aggregated_usage.get("prompt_eval_count", 0) + aggregated_usage.get("eval_count", 0))
            if total_tokens > 0:
                self.ctx.current_context_tokens = total_tokens
                if self.ctx.debug_manager.is_enabled('context'):
                    debug_log(self.ctx.debug_manager, 'context', 1,
                             f"Updated context tokens: {total_tokens}", prefix="CTX")
            refresh_ollama_context_window_size_from_ps(self.ctx)
        elif backend in ("llamacpp", "lmstudio", "gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
            # llama.cpp with KV cache reports only newly-evaluated tokens as
            # prompt_tokens, NOT the full context size. Always recalculate from
            # the actual messages array for an accurate context bar.
            if messages:
                self.ctx.current_context_tokens = self.ctx.calculate_context_tokens(messages)

    def _finalize_stream_tool_calls(self, stream_tool_call_index: dict) -> list:
        """Consolidate incremental tool-call fragments into a list of complete calls.

        Args:
            stream_tool_call_index: Maps tool-call index -> partial tool call dict.

        Returns:
            list of {"id", "type", "function": {"name", "arguments"}} dicts.
        """
        stream_tool_calls = []
        for idx in sorted(stream_tool_call_index):
            tc = stream_tool_call_index[idx]
            stream_tool_calls.append({
                "id": tc["id"],
                "type": tc["type"],
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
            })
        return stream_tool_calls

    @staticmethod
    def _close_thinking_block(start_thinking: bool, started_content: bool) -> bool:
        """Close an open `<thinking>` block, returning whether it was closed."""
        if start_thinking and not started_content:
            sys.stderr.write("\n</thinking>\n")
            return True
        return False

    def _stream_error_detail(self, e: Exception, backend: str, model: str) -> str:
        """Build a user-facing error message for a failed stream.

        Adds backend-specific authentication hints for 403 responses.
        """
        msg = f"\n[ERROR] {backend} streaming failed: {e}"
        if isinstance(e, HTTPError) and e.code == 403:
            if backend in ("gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
                env_hint = {"gemini": "GEMINI_API_KEY", "opencodezen": "OPENCODEZEN_API_KEY", "opencodego": "OPENCODEGO_API_KEY", "mistral": "MISTRAL_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
                msg += f"\n[HINT] {backend} authentication failed. Check your --api-key or {env_hint.get(backend, 'API_KEY')} env var."
            elif ":cloud" in model:
                msg += "\n[HINT] Cloud models require authentication. Check your Ollama cloud API key or pull a local model instead."
        return msg

    def _render_stream_chunk(self, chunk: dict, thought: str, content: str, is_final: bool,
                             debug: bool, show_thinking: bool, display_state: dict) -> str:
        """Render a parsed streaming chunk to the terminal (thinking + content).

        Args:
            chunk: The raw parsed chunk (for debug output).
            thought: Reasoning text for this chunk.
            content: Assistant text for this chunk.
            is_final: Whether this is the final chunk.
            debug: Print raw final JSON to stderr.
            show_thinking: Display the thinking/reasoning block.
            display_state: Mutable dict tracking start_thinking/started_content.

        Returns:
            The content text appended to the accumulator (may be "").
        """
        if debug and is_final:
            formatted_json = json.dumps(chunk, indent=4)
            sys.stderr.write(colorize(f"\n[DEBUG] Final JSON chunk from server:\n{formatted_json}\n", 'muted'))

        # Thinking display
        if thought and show_thinking:
            if not display_state["start_thinking"]:
                display_state["start_thinking"] = True
                sys.stderr.write("\n<thinking>\n")
            sys.stderr.write(thought)
            sys.stderr.flush()

        # Content display
        if content:
            if display_state["start_thinking"] and not display_state["started_content"]:
                sys.stderr.write("\n</thinking>\n")

            if not display_state["started_content"]:
                print("\n[--- Response ---]", file=sys.stdout)
                display_state["started_content"] = True

            sys.stdout.write(content)
            sys.stdout.flush()
        return content

    def query_stream(
        self,
        messages: list, model: str, stream_enabled: bool = True, debug: bool = False,
        show_thinking: bool = True, context_size: Optional[int] = None, images: Optional[list] = None,
        tool_calls_out: Optional[list] = None, **kwargs: dict
    ) -> str:
        """Stream response from any backend. Dispatches to backend-specific chunk parsing.

        Args:
            messages: Message list to send.
            model: Model name.
            stream_enabled: Whether to request streaming.
            debug: Print raw JSON chunks to stderr.
            show_thinking: Display the thinking/reasoning block.
            context_size: Optional context window size.
            images: Optional list of base64 images.
            tool_calls_out: Optional list to populate with accumulated tool_calls from streaming.
                For OpenAI-compatible streaming, incremental tool_calls are merged by index.
            **kwargs: Extra payload args.

        Returns:
            The accumulated full response content.
        """
        full_content = ""
        full_thought = ""
        start_time = time.time()
        first_token_time = None
        first_content_time = None
        backend = self.backend

        if images and messages and messages[-1].get("role") == "user":
            messages[-1] = dict(messages[-1])
            self._inject_images_into_messages(messages, images, self.backend)

        api_url, payload, headers = self._build_stream_request(
            backend, messages, model, stream_enabled, context_size, **kwargs)

        data = json.dumps(payload).encode('utf-8')
        req = Request(api_url, data=data, headers=headers)
        self._debug_request(api_url, payload)

        display_state = {"start_thinking": False, "started_content": False}
        first_chunk = True
        aggregated_usage = {}
        stream_tool_calls = []  # Accumulated tool calls from streaming (merged by index)
        stream_tool_call_index = {}  # Maps index -> partial tool call dict

        try:
            with _request_with_retry(req) as response:
                for raw_line in self._iter_stream_lines(response, backend):
                    try:
                        chunk = json.loads(raw_line)
                        thought, content, is_final, usage, tool_calls = self._parse_chunk(chunk, backend)

                        # Client-side phase timing: first token ends prefill,
                        # first content ends thinking, stream end ends the answer.
                        if thought:
                            full_thought += thought
                        if first_token_time is None and (thought or content or tool_calls):
                            first_token_time = time.time()
                        if first_content_time is None and content:
                            first_content_time = time.time()

                        self._debug_response_chunk(chunk, first_chunk, is_final)
                        first_chunk = False

                        # Accumulate tool_calls from streaming chunks
                        if tool_calls:
                            self._accumulate_stream_tool_calls(stream_tool_calls, stream_tool_call_index, tool_calls)

                        if is_final and usage:
                            self._debug_final_stats(usage)

                        if usage:
                            aggregated_usage.update(usage)

                        full_content += self._render_stream_chunk(
                            chunk, thought, content, is_final, debug, show_thinking, display_state)

                    except json.JSONDecodeError:
                        continue

            # Finalize accumulated tool_calls
            if stream_tool_call_index:
                stream_tool_calls = self._finalize_stream_tool_calls(stream_tool_call_index)
            if tool_calls_out is not None and stream_tool_calls:
                tool_calls_out.extend(stream_tool_calls)

            self._close_thinking_block(display_state["start_thinking"], display_state["started_content"])

            end_time = time.time()
            total_time = end_time - start_time
            self._update_context_tokens(backend, aggregated_usage, messages)
            phase_times = {
                "start": start_time,
                "first_token": first_token_time,
                "first_content": first_content_time,
                "end": end_time,
            }
            usage = self.calculate_stats(total_time, full_content, aggregated_usage, messages,
                                         thought=full_thought, phase_times=phase_times)
            # Track decode time (excl. prefill) so /stats avg throughput is not diluted.
            self.ctx.update_stats(usage["eval_count"], usage["prompt_eval_count"],
                                  usage.get("gen_sec") or total_time, usage["content_length"])
            self.print_stats_display(usage)

            return full_content

        except Exception as e:
            sys.stderr.write(colorize(f"{self._stream_error_detail(e, backend, model)}\n", 'error'))
            return full_content


# ============================================================================
# ============= COMMAND HANDLING CLASS ======================================
# ============================================================================

class ChatCompleter:
    """
    Provides tab-completion for interactive chat sessions.

    Supports completion for:
    - Commands (/listmodel, /switchmodel, etc.)
    - Paths (for /cwd, /ls)
    - Model names

    Call map:
      complete() → fetch_models() → fetch_models_ollama / fetch_models_llamacpp
                 → get_command_aliases() for command completion
    """

    def __init__(self, base_url: str, backend: str, api_key: Optional[str] = None) -> None:
        """Initialize the completer with connection info and command aliases.

        Args:
            base_url: Backend API base URL.
            backend: Backend name.
            api_key: Optional API key for cloud backends.
        """
        self.base_url = base_url
        self.backend = backend
        self.api_key = api_key
        self.commands = get_command_aliases()
        self.models = []

    def fetch_models(self) -> None:
        """Fetch available models from the backend."""
        if self.backend == "llamacpp":
            self.models = [m['name'] for m in fetch_models_llamacpp(self.base_url)]
        elif self.backend in ("gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
            self.models = [m['name'] for m in fetch_models_llamacpp(self.base_url, api_key=self.api_key)]
        else:
            self.models = [m['name'] for m in fetch_models_ollama(self.base_url)]


    def complete(self, text: str, state: int) -> Optional[str]:
        """The core readline autocompletion hook.

        Args:
            text: The token being completed.
            state: Readline completion state (0-based).

        Returns:
            The completion candidate, or None when exhausted.
        """
        buffer = readline.get_line_buffer()

        # 1. Model Autocompletion
        if buffer.startswith('/switchmodel '):
            matches = [m for m in self.models if m.startswith(text)]

        # 2. File Path Autocompletion (/cwd, /ls, /image and inline @)
        # buffer is the full line, text could be in the middle of the line
        elif buffer.startswith('/cwd ') or buffer.startswith('/ls ') or buffer.startswith('/image ') or text.startswith('@'):
            # Determine how much of the text is the actual path
            if text.startswith('@'):
                path_input = text[1:]
            else:
                path_input = text

            path = os.path.expanduser(path_input)
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            if not dirname:
                dirname = '.'

            matches = []
            try:
                if os.path.exists(dirname) and os.path.isdir(dirname):
                    for item in os.listdir(dirname):
                        if item.startswith(basename):
                            full_path = os.path.join(dirname, item)
                            prefix = os.path.dirname(path_input)

                            if os.path.isdir(full_path):
                                matches.append(os.path.join(prefix, item) + '/' if prefix else item + '/')
                            else:
                                matches.append(os.path.join(prefix, item) if prefix else item)
            except PermissionError:
                pass

            # Re-attach the @ symbol if we are completing an @ file inclusion
            if text.startswith('@'):
                matches = ['@' + m for m in matches]

        # 3. Command Autocompletion
        elif not text or text.startswith('/') or text in ['e', 'ex', 'exi', 'q', 'qu', 'qui']:
            matches = [c for c in self.commands if c.startswith(text)]

        else:
            matches = []

        return matches[state] if state < len(matches) else None


# ============================================================================
# ============= INPUT HANDLING CLASS =========================================
# ============================================================================


def _make_input_prompts(prompt_prefix: str) -> tuple:
    """Build (prompt, continuation) prompt strings for the current TTY style.

    readline-aware terminals get a `> ` prompt; plain terminals a `: ` prompt.
    """
    if READLINE_AVAILABLE:
        return (colorize(f"{prompt_prefix} > ", 'warning', is_prompt=True),
                colorize("... > ", 'warning', is_prompt=True))
    return (colorize(f"{prompt_prefix}: ", 'warning'),
            colorize("... : ", 'warning'))


def _remove_last_history_item() -> None:
    """Drop the just-read line from readline history (used during multiline entry)."""
    if READLINE_AVAILABLE:
        try:
            readline.remove_history_item(readline.get_current_history_length() - 1)
        except Exception:
            pass


def _add_history(text: str) -> None:
    """Add a completed multiline entry to readline history (skips empty text)."""
    if READLINE_AVAILABLE and text:
        readline.add_history(text)


_HISTORY_ESC = '\x1e'  # record separator: escape marker inside one history entry


def _encode_history_entry(text: str) -> str:
    """Encode embedded newlines so a multi-line entry survives the history file.

    Readline persists one entry per physical line, so a multi-line block would be
    split on save and come back as separate one-line entries. Escaping is
    reversible: a literal marker becomes `ESCESC` and a newline becomes `ESC n`.

    Args:
        text: A single history entry (may contain newlines).

    Returns:
        A one-line encoding suitable for `_save_history_file`.
    """
    return text.replace(_HISTORY_ESC, _HISTORY_ESC * 2).replace('\n', _HISTORY_ESC + 'n')


def _decode_history_entry(text: str) -> str:
    """Reverse `_encode_history_entry` (`ESCESC` -> marker, `ESC n` -> newline)."""
    out = []
    index = 0
    length = len(text)
    while index < length:
        if text[index] == _HISTORY_ESC:
            nxt = text[index + 1] if index + 1 < length else ''
            if nxt == _HISTORY_ESC:
                out.append(_HISTORY_ESC)
                index += 2
                continue
            if nxt == 'n':
                out.append('\n')
                index += 2
                continue
        out.append(text[index])
        index += 1
    return ''.join(out)


def _save_history_file(path: str) -> None:
    """Persist readline history, preserving multi-line entries as single items."""
    if not READLINE_AVAILABLE:
        return
    current = readline.get_current_history_length()
    limit = readline.get_history_length()
    start = current - limit + 1 if limit and limit > 0 and current > limit else 1
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(start, current + 1):
            item = readline.get_history_item(i)
            if item is None:
                continue
            f.write(_encode_history_entry(item) + '\n')


def _load_history_file(path: str) -> None:
    """Load readline history, restoring entries encoded by `_save_history_file`."""
    if not READLINE_AVAILABLE:
        return
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            raw = raw.rstrip('\n')
            if raw:
                readline.add_history(_decode_history_entry(raw))


def _read_multiline_lines(cont_prompt: str, is_terminator: object, transform: object = lambda ln: ln,
                          include_terminator: bool = False) -> Optional[list]:
    """Read continuation lines until the terminator predicate fires.

    Args:
        cont_prompt: Prompt string shown for each continuation line.
        is_terminator: Callable(line) -> bool; stops reading when True.
        transform: Applied to each appended line (e.g. strip a trailing backslash).
        include_terminator: If True, the terminating line is appended (transformed).

    Returns:
        list of str, or None if the user pressed Ctrl+C (cancelled).
    """
    lines = []
    while True:
        try:
            m_line = input(cont_prompt)
            _remove_last_history_item()
            if is_terminator(m_line):
                if include_terminator:
                    lines.append(transform(m_line))
                return lines
            lines.append(transform(m_line))
        except KeyboardInterrupt:
            print(colorize("\n[Multiline entry cancelled]", 'warning'), file=sys.stderr)
            return None


class _ExitRequested(Exception):
    """Raised when the user asks to leave the chat loop (double Ctrl+C at the prompt)."""


def gather_user_input(prompt_prefix: str, show_multiline: bool = True) -> Optional[str]:
    """
    Gather user input with multiline support.

    Supports triple-quote (`\"\"\"`) block entry and backslash (`\\`) line
    continuation. Recalling a previously-submitted multiline block with the
    up/down keys restores it as a native readline multi-line buffer: Left/Right
    move across the whole block, editing keys work as in bash, Ctrl+J (or
    Alt+Enter) inserts a new line, and Enter sends it. Returns the entered text,
    or None if a multiline entry was cancelled (single Ctrl+C inside a `\"\"\"` /
    `\\` block). A double Ctrl+C at the main prompt raises `_ExitRequested` so
    the chat loop can exit cleanly; a double Ctrl+D / EOF exits directly.

    Args:
        prompt_prefix: String shown before the input prompt.
        show_multiline: Whether to honor `\"\"\"` / `\\` multiline entry.

    Raises:
        _ExitRequested: When the user presses Ctrl+C twice at the main prompt.
    """
    ctrl_c_count = 0
    ctrl_d_count = 0

    while True:
        try:
            prompt_str, cont_prompt_str = _make_input_prompts(prompt_prefix)
            line = input(prompt_str)
            ctrl_c_count = 0

            if not line.strip():
                return line

            # 1. Handle """ Block Multiline
            if show_multiline and line.strip() == '"""':
                lines = _read_multiline_lines(cont_prompt_str,
                                              lambda ln: ln.strip() == '"""')
                if lines is None:
                    return None  # Escape out of multiline without quitting
                result = "\n".join(lines)
                _remove_last_history_item()  # drop the opening """ entry
                _add_history(result)
                return result

            # 2. Handle \ Line Continuation
            if line.endswith('\\'):
                lines = [line[:-1]]  # Strip the trailing backslash
                tail = _read_multiline_lines(
                    cont_prompt_str,
                    lambda ln: not ln.endswith('\\'),
                    transform=lambda ln: ln[:-1] if ln.endswith('\\') else ln,
                    include_terminator=True)
                if tail is None:
                    return None
                result = "\n".join(lines + tail)
                _remove_last_history_item()  # drop the opening "\" entry
                _add_history(result)
                return result

            # 3. Standard Single Line
            return line

        except KeyboardInterrupt:
            ctrl_c_count += 1
            if ctrl_c_count >= 2:
                raise _ExitRequested()
            print("\n(Press Ctrl+C again to exit)", file=sys.stderr)

        except EOFError:
            print("\n[EOF received, one more and it exits]", file=sys.stderr)
            ctrl_d_count += 1
            if ctrl_d_count >= 2:
                print("\n[Exiting]", file=sys.stderr)
                sys.exit(1)

def _confirm_sensitive_inclusion(filepath: str) -> bool:
    """Ask the user before including a sensitive file (sensitive path check + confirm).

    Args:
        filepath: The file path being included.

    Returns:
        True when the user confirmed inclusion.
    """
    print(colorize(f"[WARNING] Attempting to load sensitive file: {filepath}", 'warning'), file=sys.stderr)
    print(colorize("  This could leak private data (keys, tokens, passwords) to the LLM.", 'warning'), file=sys.stderr)
    try:
        confirm = input(colorize("  Confirm file inclusion? [y/N] ", 'warning', is_prompt=True)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        confirm = 'n'
    if confirm != 'y':
        print(colorize(f"[Skipped: {filepath}]", 'warning'), file=sys.stderr)
        return False
    return True


def _load_file_inclusion(expanded_path: str, filepath: str) -> Optional[str]:
    """Read a file and return its inclusion block (with token-count info).

    Args:
        expanded_path: ~-expanded absolute path to read.
        filepath: Display path (from the user's @ mention).

    Returns:
        The formatted inclusion text, an error message, or None when skipped.
    """
    print(colorize(f"[--- Loading file: {filepath} ---]", 'muted'), file=sys.stderr)
    try:
        with open(expanded_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        lines_count = len(file_content.splitlines())
        char_count = len(file_content)
        word_count = len(file_content.split())

        token_count = 0
        token_method = "api"
        if CommandContext._initialized:
            ctx = CommandContext()
            if ctx.base_url and ctx.model:
                if ctx.backend == "ollama":
                    token_count = get_message_token_count_ollama(ctx.base_url, file_content, ctx.model)
                elif ctx.backend == "llamacpp":
                    token_count = get_message_token_count_llamacpp(ctx.base_url, file_content)

        if token_count == 0:
            token_count = estimate_token_count(file_content)
            token_method = "est"

        if token_method == "api":
            print(colorize(f"Successfully loaded {lines_count} lines ({char_count} chars, {word_count} words, ~{token_count} tokens).", 'info'), file=sys.stderr)
        else:
            print(colorize(f"Successfully loaded {lines_count} lines ({char_count} chars, {word_count} words, ~{token_count} tokens est).", 'warning'), file=sys.stderr)

        return f"\n[Content of local file `{filepath}`]:\n```text\n{file_content}\n```\n"
    except UnicodeDecodeError:
        err_msg = f"[Failed to load `{filepath}`: Appears to be a binary or non-UTF-8 file]"
        print(colorize(err_msg, 'error'), file=sys.stderr)
        return err_msg + "\n"
    except Exception as e:
        err_msg = f"[Failed to load `{filepath}`: {e}]"
        print(colorize(err_msg, 'error'), file=sys.stderr)
        return err_msg + "\n"


def _process_file_inclusions(text: str) -> str:
    """Scan text for @filepath mentions and load referenced files.

    Args:
        text: The input line to scan for @-inclusions.

    Returns:
        The text with file contents appended after the @ mentions.
    """
    inclusions = []
    words = text.split()
    i = 0
    while i < len(words):
        word = words[i]
        if word.startswith('@') and len(word) > 1:
            candidate = word[1:]
            j = i
            # Reconstruct paths containing spaces: extend the candidate with
            # following words until it resolves to an existing file.
            while j < len(words):
                if os.path.isfile(os.path.expanduser(candidate.rstrip('.,?!;:)"\''))):
                    break
                j += 1
                if j < len(words):
                    candidate += " " + words[j]
            raw_path = candidate
            filepath = raw_path.rstrip('.,?!;:)"\'')
            expanded_path = os.path.expanduser(filepath)
            if os.path.isfile(expanded_path):
                i = j + 1
                file_size = os.path.getsize(expanded_path)
                if file_size > MAX_FILE_INCLUSION_SIZE:
                    err_msg = f"[Failed to load `{filepath}`: File too large ({file_size / 1024 / 1024:.1f} MB, max 5 MB)]"
                    print(colorize(err_msg, 'error'), file=sys.stderr)
                    continue
                if expanded_path.startswith('/etc/') or '/.' in expanded_path:
                    if not _confirm_sensitive_inclusion(filepath):
                        continue
                inclusion = _load_file_inclusion(expanded_path, filepath)
                if inclusion:
                    inclusions.append(inclusion)
                continue
            i += 1
        else:
            i += 1
    return inclusions


def _process_command_lines(text: str) -> list:
    """Process lines starting with ! (shell) or /curl (URL fetch).

    Args:
        text: The input text (may contain multiple lines).

    Returns:
        List of processed lines (shell output / fetched content substituted).
    """
    processed = []
    in_literal_block = False
    for line in text.split('\n'):
        stripped = line.lstrip()

        if stripped == '"""':
            in_literal_block = not in_literal_block
            processed.append(line)
            continue

        if not in_literal_block and stripped.startswith("!"):
            command = stripped[1:].strip()
            if command:
                output_str = execute_os_command(command)
            else:
                output_str = "[Command rejected: Invalid characters]"
            processed.append(output_str)

        elif stripped.startswith("/curl "):
            url = stripped[6:].strip()
            if not url:
                processed.append(line)
                continue
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            try:
                print(colorize(f"[--- Fetching URL: {url} ---]", 'muted'), file=sys.stderr)
                text_content, used_tool = fetch_and_convert_url(url)
                word_count = len(text_content.split()) if text_content else 0
                if word_count > 0:
                    print(colorize(f"[Successfully fetched {word_count} words from {url}]" + (f" (via {used_tool})" if used_tool and used_tool != "None" else ""), 'muted'), file=sys.stderr)
                    processed.append(text_content)
                    preview = text_content[:500].strip()
                    header = "[Output truncated for terminal display. LLM received full text.]" if len(text_content) > 500 else ""
                    print(colorize(f"{header}", 'muted'), file=sys.stderr)
                    print(colorize(f"```text\n{preview}\n```", 'info'), file=sys.stderr)
                else:
                    print(colorize(f"[Warning: No text content could be extracted from {url}]", 'warning'), file=sys.stderr)
            except Exception as e:
                print(colorize(f"[Failed to fetch URL: {e}]", 'error'), file=sys.stderr)
        else:
            processed.append(line)
    return processed


def process_inline_commands(full_input: str) -> str:
    """Process inline commands (!, /curl, @) within user input.

    Args:
        full_input: The raw user input line.

    Returns:
        The input with shell outputs, fetched content, and @-file content appended.
    """
    file_inclusions = _process_file_inclusions(full_input)
    processed_lines = _process_command_lines(full_input)
    final_output = "\n".join(processed_lines)
    if file_inclusions:
        final_output += "\n" + "".join(file_inclusions)
    return final_output



def execute_os_command(command: str, timeout: Optional[int] = None) -> str:
    """
    Execute OS command with safety checks and timeout.

    Args:
        command (str): Command to execute
        timeout (int): Maximum execution time in seconds (None = use default)

    Returns:
        str: Command output or error message
    """
    if timeout is None and CommandContext._initialized:
        timeout = CommandContext().shell_timeout
    if timeout is None:
        timeout = 5
    # Gate: opencode breakdown + home guard + CWD leniency (pipes/redirs allowed)
    if command and len(command) > 500:
        msg = "[Command rejected: too long]"
        print(colorize(msg, 'error'), file=sys.stderr)
        return "[Command rejected: too long]"
    ctx = CommandContext() if CommandContext._initialized else None
    # Use host mode for inline ! commands (never container)
    gate = check_shell_approval(command, ctx=ctx, executor_mode="host")
    if not gate["approved"]:
        msg = gate.get("message") or "[Command rejected by shell approval gate]"
        print(colorize(msg, 'error'), file=sys.stderr)
        return f"[Command rejected: {msg}]"

    print(f"[--- Executing (max {timeout}s): {command} ---]", file=sys.stderr)

    try:
        process = subprocess.run(
            command, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            timeout=timeout,
            executable="/bin/bash"
        )

        raw_output = process.stdout if process.stdout else ""

        # --- THE FIX: Print the output to the user's terminal! ---
        if raw_output.strip():
            print(colorize(raw_output, 'info'))
        else:
            print(colorize("[Command executed successfully with no output]", 'muted'))
        # ---------------------------------------------------------


        output = raw_output or "[Command executed successfully with no output]"

    except subprocess.TimeoutExpired:
        print(f"[Command timed out after {timeout} seconds!]", file=sys.stderr)
        output = f"[Command execution interrupted: Time limit exceeded ({timeout}s)]"

    except Exception as e:
        print(f"[Failed to execute command: {e}]", file=sys.stderr)
        return f"[Execution error: {e}]"

    return f"\n[Command executed: `{command}`]\n```text\n{output.strip()}\n```\n"


def fetch_and_convert_url(url: str) -> tuple:
    """Fetch URL and extract clean text using core standard libraries only.

    Args:
        url: The URL to fetch.

    Returns:
        (text, tool) tuple where tool is "htmlstrip" or "None".
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        req = Request(url, headers=headers)
        with _request_with_retry(req, timeout=15) as response:
            charset = response.info().get_content_charset() or 'utf-8'
            html_content = response.read().decode(charset, errors='ignore')
        if not html_content.strip():
            return "", "None"
        stripper = CoreHTMLStripper()
        stripper.feed(html_content)
        return stripper.get_text(), "htmlstrip"
    except Exception as e:
        return f"[Failed to fetch URL: {e}]", "None"


class CoreHTMLStripper(HTMLParser):
    """Zero-dependency HTML text extractor using a robust nesting-depth counter.

    Tracks skip-depth instead of a single tag name, so nested or sequential
    skipped tags (script, style, etc.) properly resume text capture.

    Call map:
      feed(text) → handle_starttag / handle_endtag / handle_data
      get_text() → returns accumulated text
    """
    skip_tags = {'script', 'style', 'head', 'meta', 'noscript', 'link', 'title'}

    def __init__(self) -> None:
        """Initialize the stripper with empty text parts and skip depth 0."""
        super().__init__()
        self.text_parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """Increment skip depth for excluded block tags.

        Args:
            tag: The start tag name.
            attrs: Tag attributes (ignored).
        """
        if tag.lower() in self.skip_tags:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        """Decrement skip depth for excluded block tags.

        Args:
            tag: The end tag name.
        """
        if tag.lower() in self.skip_tags:
            self.skip_depth = max(0, self.skip_depth - 1)

    def handle_data(self, data: str) -> None:
        """Capture non-skipped text, trimmed of surrounding whitespace.

        Args:
            data: Raw text content between tags.
        """
        if self.skip_depth == 0:
            cleaned = data.strip()
            if cleaned:
                self.text_parts.append(cleaned)

    def get_text(self) -> str:
        """Return the accumulated text joined by newlines."""
        return '\n'.join(self.text_parts)


HTMLStripper = CoreHTMLStripper  # Backward-compat alias for tests


# ============================================================================
# ============= CHAT LOOP CLASS ============================================
# ============================================================================

class ChatLoop:
    """
    Unified chat loop that handles both Ollama and Llama.cpp backends.

    State is managed through a shared CommandContext singleton instead of
    individual self.* attributes.

    Call map:
      run() → run_init_session() → run_update_ollama_context()
            → dispatches to run_handle_*() by command
            → run_process_query() on non-command input

      run_process_query() → query_handler.query_stream()
      run_agentic_query() → _init_agentic_query()
                          → query_handler.query_sync() (ReAct loop)
                          → parse_tool_call() / parse_tool_calls()
                          → _execute_tool_calls() → tool_registry.execute()
                          → _finalize_agentic_query() → query_handler.query_stream()

      parse_tool_call() → _normalize_tool_json() / _extract_json_balanced()
      parse_tool_calls() → _find_tool_call_brace() / _extract_json_balanced()
      _execute_tool_calls() → tool_registry.execute()
      _finalize_agentic_query() → parse_tool_call() / parse_tool_calls()
    """

    def __init__(self, context: CommandContext) -> None:
        """Initialize chat loop with shared context, completer, and query handler.

        Args:
            context: The shared CommandContext singleton.
        """
        self.ctx = context
        self.completer = context.create_completer()
        self.query_handler = context.create_query_handler()
        self.executor = context.create_executor()
        self.tool_registry = context.create_tool_registry()
        self.commands = get_command_aliases()

    def dump_context_to_file(self, filepath: str) -> None:
        """Dump current conversation history to a JSON file for browsing."""
        if not hasattr(self, 'messages') or not self.messages:
            raise ValueError("No conversation history to dump")

        history = []
        for msg in self.messages:
            msg_copy = dict(msg)

            if 'images' in msg_copy:
                msg_copy['images'] = [
                    f"[image: {len(img)} bytes]" for img in msg_copy['images']
                ]
            history.append(msg_copy)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


    def fetch_models(self) -> None:
        """Fetch available models from the backend."""
        if self.ctx.backend in ("llamacpp", "lmstudio"):
            self.ctx.models = [m['name'] for m in fetch_models_llamacpp(self.ctx.base_url)]
        elif self.ctx.backend in ("gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
            self.ctx.models = [m['name'] for m in fetch_models_llamacpp(self.ctx.base_url, api_key=self.ctx.api_key)]
        else:
            self.ctx.models = [m['name'] for m in fetch_models_ollama(self.ctx.base_url)]

        # Only auto-select if a model WAS explicitly set but not found on server
        # Don't auto-select if model is empty (user must pick manually)
        if self.ctx.model and self.ctx.models and self.ctx.model not in self.ctx.models:
            old_model = self.ctx.model
            self.ctx.model = self.ctx.models[0]
            print(colorize(f"[INFO] Model '{old_model}' not found on server. Auto-selecting '{self.ctx.model}'", 'warning'), file=sys.stderr)

    def run(self, stream_enabled: bool = True, debug: bool = False, images: Optional[List[str]] = None) -> None:
        """Main chat loop - handles interactive session."""
        host_clean = urlparse(self.ctx.base_url).netloc

        self.run_init_session(stream_enabled, debug, images)

        while True:
            try:
                self.run_update_ollama_context(host_clean)
                self.run_display_context_bar()

                prompt_prefix = self.run_build_prompt(host_clean)
                full_input = gather_user_input(prompt_prefix)
                if full_input is None or not full_input.strip():
                    continue

                if not self.ctx.model and not (full_input.startswith('/') or full_input.lower() in ('exit', 'quit')):
                    print(colorize("\n[ERROR] No model selected. Use /listmodel to see available models, then /switchmodel <name> to select one.", 'error'), file=sys.stderr)
                    continue

                dispatched = self._dispatch_command(full_input)
                if dispatched is True:
                    break
                if dispatched is False:
                    continue

                self.run_process_query(full_input)

            except _ExitRequested:
                self.run_handle_exit('exit')
                break

            except KeyboardInterrupt:
                print("\n[Interrupted]", file=sys.stderr)
                continue

            except Exception as e:
                if isinstance(e, EOFError):
                    print("[EOF - Goodbye!]", file=sys.stderr)
                    break
                else:
                    print(colorize(f"[ERROR] ChatLoop->run {e}", 'error'), file=sys.stderr)
                    if self.ctx.debug_mode or self.ctx.debug_manager.get_level('all') > 0:
                        traceback.print_exc(file=sys.stderr)

        return

    def _dispatch_command(self, full_input: str) -> Optional[bool]:
        """Route a line to the matching /command handler.

        Args:
            full_input: The raw input line from the user.

        Returns:
            True if the loop should break (exit requested), False when a
            handler consumed the line (continue), or None if it is a plain
            query to be processed normally.
        """
        if self.run_handle_exit(full_input) is True:
            return True

        handlers = (
            self.run_handle_help,
            self.run_handle_stats,
            self.run_handle_listmodel,
            self.run_handle_context_size,
            self.run_handle_clear,
            self.run_handle_image,
            self.run_handle_dumpcontext,
            self.run_handle_debug,
            self.run_handle_thinking,
            self.run_handle_reasoning,
            self.run_handle_cwd,
            self.run_handle_ls,
            self.run_handle_switchmodel,
            self.run_handle_spawnshell,
            self.run_handle_agentic,
            self.run_handle_listtool,
            self.run_handle_compact,
            self.run_handle_drop,
            self.run_handle_tokencount,
            self.run_handle_sessions,
            self.run_handle_resume,
            self.run_handle_save,
        )
        for handler in handlers:
            if handler(full_input) is False:
                return False
        return None

    def list_models(self, filter_arg: Optional[str] = None) -> None:
        """List available models from the backend, with optional name filter."""
        if self.ctx.backend == "ollama":
            list_models_ollama(self.ctx.base_url, filter_arg, file=sys.stdout)
            return
        elif self.ctx.backend in ("gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
            models = fetch_models_llamacpp(self.ctx.base_url, api_key=self.ctx.api_key)
        else:
            models = fetch_models_llamacpp(self.ctx.base_url)
        if not models:
            print(colorize(f"\n[No models found via {self.ctx.backend} API]", 'warning'), file=sys.stderr)
            return
        models.sort(key=lambda x: x.get('name', ''))
        header = f"{'NAME':<50} | {'OWNED BY'}"
        print(colorize(header, 'muted'))
        print(colorize("-" * len(header), 'muted'))
        for m in models:
            owned_by = m.get('owned_by', 'N/A')
            print(f"{m['name']:<50} | {owned_by}")
        print()

    def set_context_size(self, full_input: str) -> None:
        """Parse and apply /contextsizeset command argument."""
        parts = full_input.split()
        if len(parts) > 1 and parts[1].isdigit():
            val = int(parts[1])
            if val == 0:
                self.ctx.context_size = None
                print("[Context size reset to default]", file=sys.stderr)
            else:
                if val > MAX_CONTEXT_SIZE:
                    print(colorize(f"[ERROR] Context size {val} exceeds maximum {MAX_CONTEXT_SIZE}", 'error'), file=sys.stderr)
                    return
                if self.ctx.backend in ("llamacpp", "lmstudio"):
                    print(colorize(f"[contextsizeset is not supported on {self.ctx.backend} backend]", 'warning'), file=sys.stderr)
                    return
                self.ctx.context_size = val
                self.ctx.context_window_size = val
                print(f"[Context size set to {val}]", file=sys.stderr)
        else:
            print("[Usage: /contextsizeset <integer> (use 0 for default)]", file=sys.stderr)

    def handle_spawnshell(self) -> Optional[str]:
        """Spawn an interactive shell session."""
        if not PTY_AVAILABLE:
            print("[spawnshell requires Unix-like system with pty]", file=sys.stderr)
            return None

        print("[Spawning interactive shell. Type 'exit' to return.]", file=sys.stderr)

        try:
            shell_cmd = os.environ.get('SHELL', '/bin/bash')
            output_lines = []

            def read_output(fd: int) -> bytes:
                """Read up to 4096 bytes from the pty fd, capturing decoded text.

                Args:
                    fd: The pty file descriptor.

                Returns:
                    The raw bytes read (b'' on OSError).
                """
                try:
                    data = os.read(fd, 4096)
                    if data:
                        output_lines.append(data.decode('utf-8', errors='replace'))
                    return data
                except OSError:
                    return b''

            pty.spawn(shell_cmd, read_output)

        except Exception as e:
            print(f"[ERROR] Shell exited: {e}", file=sys.stderr)

        return "".join(output_lines).strip()

    def _parse_shell_into_blocks(self, text: str) -> list:
        """Split shell session into command blocks by detecting prompt lines.

        Args:
            text: The captured shell session text.

        Returns:
            List of non-empty command block strings.
        """
        text = strip_ansi(text)
        import re
        parts = re.split(r'(?m)^.*[\$#] ', text)
        return [p.strip() for p in parts if p.strip()]

    def _filter_smart_blocks(self, blocks: list) -> list:
        """Filter out trivial commands (cd, ls, pwd, clear, echo, exit).

        Args:
            blocks: List of parsed command blocks.

        Returns:
            Subset of blocks worth sending (non-trivial, >20 chars).
        """
        boring_commands = {'cd', 'ls', 'pwd', 'clear', 'exit', 'echo'}
        return [b for b in blocks if b.split('\n')[0].strip().split()[0] not in boring_commands if b.split() and len(b.strip()) > 20]

    def _edit_session(self, content: str) -> Optional[str]:
        """Open content in editor (VISUAL > EDITOR > vim) and return the edited result.

        Args:
            content: The text to open in the editor.

        Returns:
            Edited text, or None if the edit was cancelled.
        """
        import tempfile
        editor = os.environ.get('VISUAL') or os.environ.get('EDITOR') or 'vim'
        tmpfile = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                tmpfile = f.name
            subprocess.run([editor, tmpfile], check=True)
            with open(tmpfile, 'r') as f:
                return f.read()
        except (subprocess.CalledProcessError, Exception) as e:
            print(colorize(f"[Edit cancelled: {e}]", 'error'), file=sys.stderr)
            return None
        finally:
            if tmpfile:
                try:
                    os.unlink(tmpfile)
                except OSError:
                    pass

    def _handle_shell_session(self, session_output: str) -> None:
        """Parse captured shell session and let user choose what to send.

        Args:
            session_output: The raw shell session text.
        """
        clean = strip_ansi(session_output)
        if len(clean) > MAX_WRITE_FILE_SIZE:
            print(colorize(f"\n[Shell session output too large ({len(clean)} bytes), discarding — will not be sent to LLM]", 'error'), file=sys.stderr)
            return
        token_count = estimate_token_count(clean)
        blocks = self._parse_shell_into_blocks(clean)

        print(colorize(f"\n[Shell session: {len(blocks)} command(s), ~{token_count} tokens]", 'info'), file=sys.stderr)
        for i, block in enumerate(blocks, 1):
            block_tokens = estimate_token_count(block)
            first = block.split('\n')[0].strip()[:80]
            print(colorize(f"  {i}. ~{block_tokens:>5}  {first}", 'muted'), file=sys.stderr)

        try:
            choice = input(colorize("\n[S]end all / Sma[r]t / [E]dit / S[k]ip / [1,3-5] by #? ", 'warning')).strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = 'k'

        if choice in ('s', 'send'):
            self.run_process_query(f"Shell session transcript:\n{clean}")
        elif choice in ('r', 'smart'):
            filtered = self._filter_smart_blocks(blocks)
            if filtered:
                content = "\n---\n".join(f"$ {b.split(chr(10))[0].strip()}\n{chr(10).join(b.split(chr(10))[1:]).strip()}" for b in filtered)
                print(colorize(f"[Smart filter: {len(filtered)}/{len(blocks)} commands]", 'info'), file=sys.stderr)
                self.run_process_query(f"Shell session transcript (filtered):\n{content}")
            else:
                print(colorize("[No interesting commands found, sending all]", 'warning'), file=sys.stderr)
                self.run_process_query(f"Shell session transcript:\n{clean}")
        elif choice in ('e', 'edit'):
            edited = self._edit_session(clean)
            if edited and edited.strip():
                edited_clean = strip_ansi(edited)
                print(colorize(f"[Editing accepted: ~{estimate_token_count(edited_clean)} tokens]", 'info'), file=sys.stderr)
                self.run_process_query(f"Shell session transcript:\n{edited_clean}")
            else:
                print(colorize("[Edit cancelled or empty, discarding]", 'muted'), file=sys.stderr)
        else:
            # Try numeric selection (e.g. "1", "1,3,7", "1-5", "1,3-5,7")
            indices = self._parse_number_ranges(choice, len(blocks))
            if indices:
                content = "\n---\n".join(f"$ {blocks[i - 1].split(chr(10))[0].strip()}\n{chr(10).join(blocks[i - 1].split(chr(10))[1:]).strip()}" for i in indices)
                print(colorize(f"[Selected {len(indices)}/{len(blocks)} commands]", 'info'), file=sys.stderr)
                self.run_process_query(f"Shell session transcript (selected commands):\n{content}")
            else:
                print(colorize("[Shell session discarded]", 'muted'), file=sys.stderr)

    @staticmethod
    def _parse_number_ranges(text: str, max_val: int) -> Optional[list]:
        """Parse '1,3-5,7' into [1, 3, 4, 5, 7]. Returns None on invalid input.

        Args:
            text: The selection string.
            max_val: Maximum valid index (1-based).

        Returns:
            Sorted list of unique 1-based indices, or None when invalid/out of range.
        """
        valid_chars = set('0123456789,- ')
        if not text or not all(c in valid_chars for c in text):
            return None
        result = set()
        for part in text.replace(',', ' ').split():
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                bounds = part.split('-')
                if len(bounds) != 2:
                    return None
                try:
                    start, end = int(bounds[0]), int(bounds[1])
                    if start < 1 or end > max_val or start > end:
                        return None
                    result.update(range(start, end + 1))
                except ValueError:
                    return None
            else:
                try:
                    n = int(part)
                    if n < 1 or n > max_val:
                        return None
                    result.add(n)
                except ValueError:
                    return None
        return sorted(result)

    # ============================================================================
    # ============= REFACTORED RUN HANDLERS ====================================
    # ============================================================================

    def run_init_session(self, stream_enabled: bool = True, debug: bool = False, images: Optional[List[str]] = None) -> None:
        """Initialize session state from args."""
        self.fetch_models()
        self.ctx.debug_mode = debug
        self.ctx.stream_enabled = stream_enabled

        print(colorize(f"\n[ollamaquery2 v{__version__} - {self.ctx.backend.upper()} Chat Mode]", 'info'), file=sys.stderr)
        print(format_help_text(compact=True), file=sys.stderr)
        print(colorize("Type /help for details", 'muted'), file=sys.stderr)
        if READLINE_AVAILABLE:
            print(colorize('Multiline: open/close with """ ; recall with Up/Down, '
                           'edit with Left/Right, Ctrl+J adds a line\n', 'muted'), file=sys.stderr)
        else:
            print(file=sys.stderr)

        if images:
            self.ctx.current_images = images

        if READLINE_AVAILABLE:
            try:
                readline.set_completer_delims(' \t\n')
                self.completer.fetch_models()
                readline.set_completer(self.completer.complete)
                readline.parse_and_bind('tab: complete')

                # Let a recalled multi-line buffer grow a line: Ctrl+J / Alt+Enter
                # insert a literal newline instead of submitting (Enter still sends).
                try:
                    readline.parse_and_bind(r'"\C-j": "\C-v\C-j"')
                    readline.parse_and_bind(r'"\e\r": "\C-v\C-j"')
                except Exception:
                    pass

                histfile = os.path.expanduser("~/.ollamaquery.d/session")
                histdir = os.path.dirname(histfile)
                if not os.path.exists(histdir):
                    os.makedirs(histdir, exist_ok=True)

                try:
                    _load_history_file(histfile)
                except Exception:
                    pass

                readline.set_history_length(1000)

                def save_history() -> None:
                    """Persist the readline history to disk at exit."""
                    if READLINE_AVAILABLE:
                        try:
                            _save_history_file(histfile)
                        except Exception:
                            pass

                atexit.register(save_history)
            except Exception as e:
                print(colorize(f"[ERROR] Readline setup failed: {e}", 'error'), file=sys.stderr)

        if images:
            print(colorize("[Image loaded for session]", 'success'), file=sys.stderr)

        if self.ctx.context_window_size == 0:
            refresh_context_window_size(self.ctx)

    def run_update_ollama_context(self, host_clean: str) -> None:
        """Update context window size for Ollama backend."""
        if self.ctx.backend == "ollama" and self.ctx.context_window_size == 0:
            if not self.ctx.model:
                return
            model_ctx_list = fetch_loaded_models_context_ollama(self.ctx.base_url)
            for m, c in model_ctx_list:
                if m.startswith(self.ctx.model.split(':')[0]):
                    if c > 0:
                        self.ctx.context_window_size = c

    def run_display_context_bar(self) -> None:
        """Display context bar and warnings."""
        if self.ctx.context_window_size > 0:
            bar = context_bar(self.ctx.current_context_tokens, self.ctx.context_window_size)
            print(bar, file=sys.stderr)

            pct = self.ctx.current_context_tokens / self.ctx.context_window_size
            if pct >= 0.80:
                print(colorize("Context almost full! /clear recommended", 'error'), file=sys.stderr)
            elif pct >= 0.60:
                print(colorize("Context getting full", 'warning'), file=sys.stderr)

    def run_build_prompt(self, host_clean: str) -> str:
        """Build the dynamic prompt prefix."""
        if self.ctx.model:
            prompt_prefix = f"{self.ctx.backend}@{host_clean}/{self.ctx.model}"
        else:
            prompt_prefix = f"{self.ctx.backend}@{host_clean}/(no model selected)"
        if self.ctx.current_images:
            prompt_prefix += "[img]"
        return prompt_prefix

    def run_handle_exit(self, full_input: str) -> Optional[bool]:
        """Handle exit/quit commands. Returns True to break loop."""
        if full_input.lower() in ['exit', 'quit', '/exit', '/quit']:
            # Auto-save session on exit if there's meaningful conversation
            if hasattr(self, 'messages') and len(self.messages) > 2:
                try:
                    sm = SessionManager()
                    session_id = sm.save_session(self.messages, self.ctx)
                    print(colorize(f"[Auto-save] Session saved as {session_id}", 'muted'), file=sys.stderr)
                except Exception:
                    pass
            print(colorize("\n[Goodbye!]", 'info'), file=sys.stderr)
            return True
        return None

    def run_handle_help(self, full_input: str) -> Optional[bool]:
        """Handle /help command. Returns False to continue loop."""
        if full_input in ['/?', '/help']:
            print(format_help_text(compact=False), file=sys.stderr)
            return False
        return None

    def run_handle_stats(self, full_input: str) -> Optional[bool]:
        """Handle /stats and /usage commands."""
        if full_input.lower() in ['/stats', '/usage']:
            cum = self.ctx.get_cumulative_stats()
            print(colorize("\n[Usage Summary]", 'info'), file=sys.stderr)
            print(f"  Queries: {cum['total_queries']}", file=sys.stderr)
            print(f"  Tokens (completion): {cum['total_completion_tokens']:,}", file=sys.stderr)
            print(f"  Tokens (prompt): {cum['total_prompt_tokens']:,}", file=sys.stderr)
            print(f"  Total tokens: {cum['total_tokens']:,}", file=sys.stderr)
            print(f"  Avg throughput: {cum['avg_tps']:.2f} t/s", file=sys.stderr)
            print(f"  Avg tokens/query: {cum['avg_tokens_per_query']:.1f}", file=sys.stderr)
            print()
            if self.ctx.backend == "ollama":
                model_ctx_list = fetch_loaded_models_context_ollama(self.ctx.base_url)
                for m, c in model_ctx_list:
                    print(f"    model : {m} context : {c}", file=sys.stderr)
            return False
        return None

    def run_handle_listmodel(self, full_input: str) -> Optional[bool]:
        """Handle /listmodel and /listmodelall commands."""
        if full_input.startswith('/listmodel'):
            parts = full_input.split(maxsplit=1)
            if full_input.strip() == '/listmodelall':
                if self.ctx.backend == "ollama":
                    list_models_ollama(self.ctx.base_url, include_capabilities=True, file=sys.stderr)
                else:
                    self.list_models()
            else:
                self.list_models(parts[1] if len(parts) > 1 else None)
            return False
        return None

    def run_handle_context_size(self, full_input: str) -> Optional[bool]:
        """Handle /contextsizeset command."""
        if full_input.startswith('/contextsizeset'):
            self.set_context_size(full_input)
            return False
        return None

    def run_handle_clear(self, full_input: str) -> Optional[bool]:
        """Handle /clear command."""
        if full_input == '/clear':
            print(colorize("[Context memory wiped clean]", 'success'), file=sys.stderr)
            self.ctx.reset()
            if hasattr(self, 'messages'):
                del self.messages
            refresh_context_window_size(self.ctx)
            return False
        return None

    def run_handle_image(self, full_input: str) -> Optional[bool]:
        """Handle /image command for attaching or clearing images."""
        if full_input.startswith('/image'):
            parts = full_input.split(maxsplit=1)
            if len(parts) < 2 or parts[1].strip() in ('', 'clear', 'none'):
                self.ctx.current_images = []
                print(colorize("[Image cleared]", 'info'), file=sys.stderr)
            else:
                paths = shlex.split(parts[1].strip(), posix=(os.name != 'nt'))
                new_images = []
                loaded = 0
                for p in paths:
                    img_path = os.path.expanduser(p)
                    if os.path.isfile(img_path):
                        img_data = prepare_image_data(img_path)
                        if img_data:
                            new_images.append(img_data)
                            loaded += 1
                        else:
                            print(colorize(f"[Error: Could not encode {os.path.basename(img_path)}]", 'error'), file=sys.stderr)
                    else:
                        print(colorize(f"[Error: File not found: {img_path}]", 'error'), file=sys.stderr)
                if loaded:
                    self.ctx.current_images = new_images
                    names = ", ".join(os.path.basename(p) for p in paths if os.path.isfile(os.path.expanduser(p)))
                    print(colorize(f"[{loaded} image(s) attached: {names}]", 'success'), file=sys.stderr)
            return False
        return None

    def run_handle_dumpcontext(self, full_input: str) -> Optional[bool]:
        """Handle /dumpcontext command."""
        if full_input.startswith('/dumpcontext'):
            parts = full_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print(colorize("[Usage: /dumpcontext <filepath>]", 'warning'), file=sys.stderr)
                return False

            dump_path = os.path.expanduser(parts[1].strip())
            try:
                self.dump_context_to_file(dump_path)
                print(colorize(f"[Context dumped to {dump_path}]", 'success'), file=sys.stderr)
            except ValueError as e:
                print(colorize(f"[ERROR] {e}", 'error'), file=sys.stderr)
            except Exception as e:
                print(colorize(f"[ERROR] Failed to dump context: {e}", 'error'), file=sys.stderr)
            return False
        return None

    def _debug_level_name(self, level: int) -> str:
        """Map a numeric debug level to its name.

        Args:
            level: 0-3 debug level.

        Returns:
            "off"/"basic"/"verbose"/"trace" (or the raw number).
        """
        return {0: 'off', 1: 'basic', 2: 'verbose', 3: 'trace'}.get(level, str(level))

    def _debug_show_categories(self) -> None:
        """Print all debug categories with their current levels."""
        print(colorize("\n[Debug Categories - Use /debug <category> <level>]", 'info'), file=sys.stderr)
        print("  Levels: off (0), basic (1), verbose (2), trace (3)", file=sys.stderr)
        print()
        for cat, desc in DebugManager.CATEGORIES.items():
            current_level = self.ctx.debug_manager.get_level(cat)
            level_name = self._debug_level_name(current_level)
            marker = '>' if current_level > 0 else ' '
            print(f"  {marker} {cat:<12} [{level_name:<7}] {desc}", file=sys.stderr)
        print()

    def _debug_handle_single(self, arg: str) -> None:
        """Handle a single /debug argument (list/status/on/off/level/category).

        Args:
            arg: The first argument after /debug.
        """
        if arg == 'list':
            print(colorize("\n[Debug Categories]", 'info'), file=sys.stderr)
            for cat, desc in DebugManager.CATEGORIES.items():
                current_level = self.ctx.debug_manager.get_level(cat)
                level_name = self._debug_level_name(current_level)
                marker = '>' if current_level > 0 else ' '
                print(f"  {marker} {cat:<12} [{level_name}]", file=sys.stderr)
            print()

        elif arg == 'status':
            status = self.ctx.debug_manager.get_status()
            if any(level > 0 for level in status.values() if isinstance(level, int)):
                print(colorize("\n[Active Debug Categories]", 'info'), file=sys.stderr)
                for cat, level in status.items():
                    if isinstance(level, int) and level > 0:
                        level_name = self._debug_level_name(level)
                        print(f"  > {cat}: {level_name}", file=sys.stderr)
                print()
            else:
                print(colorize("\n[Debug: No categories active]\n", 'muted'), file=sys.stderr)

        elif arg in ('on', 'off', '0', '1', '2', '3'):
            if arg == 'on':
                level = 'verbose'
            elif arg == 'off':
                level = 'off'
            elif arg in ('0', '1', '2', '3'):
                level_map = {'0': 'off', '1': 'basic', '2': 'verbose', '3': 'trace'}
                level = level_map[arg]
            else:
                level = arg

            self.ctx.debug_manager.set_level('all', level)
            print(colorize(f"\n[Debug: ALL categories -> {level}]\n", 'success'), file=sys.stderr)

        else:
            if arg in DebugManager.CATEGORIES:
                current = self.ctx.debug_manager.get_level(arg)
                level_name = self._debug_level_name(current)
                desc = DebugManager.CATEGORIES[arg]
                print(colorize(f"\n[Debug: {arg} = {level_name}] - {desc}", 'info'), file=sys.stderr)
                print("Usage: /debug {} [off|basic|verbose|trace]\n".format(arg), file=sys.stderr)
            else:
                print(colorize(f"\n[Unknown category: '{arg}']", 'error'), file=sys.stderr)
                print(colorize("Use '/debug' to see available categories\n", 'muted'), file=sys.stderr)

    def run_handle_debug(self, full_input: str) -> Optional[bool]:
        """Handle /debug command."""
        if full_input.startswith('/debug'):
            parts = full_input.split(maxsplit=2)

            if len(parts) == 1 or (len(parts) == 2 and not parts[1].strip()):
                self._debug_show_categories()

            elif len(parts) == 2:
                self._debug_handle_single(parts[1].lower())

            elif len(parts) == 3:
                category, level = parts[1].lower(), parts[2].lower()

                if not self.ctx.debug_manager.set_level(category, level):
                    print(colorize(f"\n[Invalid category '{category}' or level '{level}']", 'error'), file=sys.stderr)
                    print(colorize("Use '/debug' to see available categories\n", 'muted'), file=sys.stderr)
                else:
                    print(colorize(f"\n[Debug: {category} -> {level}]\n", 'success'), file=sys.stderr)

            return False
        return None

    def run_handle_thinking(self, full_input: str) -> Optional[bool]:
        """Handle /thinkingon and /thinkingoff commands."""
        if full_input == '/thinkingoff':
            self.ctx.force_no_thinking = True
            print(colorize("[Model will skip reasoning phase]", 'warning'), file=sys.stderr)
            return False
        elif full_input == '/thinkingon':
            self.ctx.force_no_thinking = False
            print(colorize("[Reasoning phase enabled]", 'success'), file=sys.stderr)
            return False
        return None

    def run_handle_reasoning(self, full_input: str) -> Optional[bool]:
        """Handle /reasoning [off|on|low|medium|high] command."""
        if not full_input.startswith('/reasoning'):
            return None
        parts = full_input.split()
        if len(parts) == 1 or parts[1] in ('status', 'show'):
            if self.ctx.reasoning_effort:
                state_str = f"Reasoning effort: {self.ctx.reasoning_effort}"
            elif self.ctx.force_no_thinking:
                state_str = "Reasoning: off"
            else:
                state_str = "Reasoning: on (model default effort)"
            print(colorize(f"[{state_str}]", 'info'), file=sys.stderr)
            self._print_reasoning_flag()
            return False
        val = parts[1].lower()
        if val == 'off':
            self.ctx.force_no_thinking = True
            self.ctx.reasoning_effort = None
            print(colorize("[Reasoning: off — model will skip reasoning phase]", 'warning'), file=sys.stderr)
            self._print_reasoning_flag()
            return False
        if val == 'on':
            self.ctx.force_no_thinking = False
            self.ctx.reasoning_effort = None
            print(colorize("[Reasoning: on — model default effort]", 'success'), file=sys.stderr)
            self._print_reasoning_flag()
            return False
        if val in ('low', 'medium', 'high'):
            self.ctx.force_no_thinking = False
            self.ctx.reasoning_effort = val
            print(colorize(f"[Reasoning effort: {val}]", 'info'), file=sys.stderr)
            self._print_reasoning_flag()
            return False
        print(colorize("[Usage: /reasoning [off|on|low|medium|high]]", 'warning'), file=sys.stderr)
        return False

    def _print_reasoning_flag(self) -> None:
        """Print the exact backend flag the next query will carry for reasoning."""
        flags = self.query_handler.reasoning_flags(
            self.ctx.model, self.ctx.backend,
            self.ctx.force_no_thinking, self.ctx.reasoning_effort)
        flag_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in flags.items()) or "none"
        print(colorize(f"[Backend flag: {flag_str}]", 'muted'), file=sys.stderr)

    def run_handle_cwd(self, full_input: str) -> Optional[bool]:
        """Handle /cwd command."""
        if full_input.startswith('/cwd'):
            parts = full_input.split(maxsplit=1)
            if len(parts) > 1:
                try:
                    os.chdir(os.path.expanduser(parts[1]))
                except Exception as e:
                    print(colorize(f"[ERROR] {e}", 'error'), file=sys.stderr)
            print(colorize(f"[Current directory: {os.getcwd()}]", 'info'), file=sys.stderr)
            return False
        return None

    def run_handle_ls(self, full_input: str) -> Optional[bool]:
        """Handle /ls command."""
        if full_input.startswith('/ls'):
            try:
                args_part = full_input[3:].strip()
                if args_part:
                    ls_args = shlex.split(args_part)
                    subprocess.run(['ls'] + ls_args, check=False)
                else:
                    subprocess.run(['ls'], check=False)
            except Exception as e:
                print(colorize(f"[ERROR] {e}", 'error'), file=sys.stderr)
            return False
        return None

    def run_handle_switchmodel(self, full_input: str) -> Optional[bool]:
        """Handle /switchmodel command."""
        if full_input.startswith('/switchmodel'):
            parts = full_input.split()
            if len(parts) > 1 and parts[1].strip():
                new_model = parts[1].strip()
                if self.ctx.backend == "ollama":
                    model_exists = is_available_ollama_model(self.ctx.base_url, new_model)
                elif self.ctx.backend in ("gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
                    model_exists = is_available_llamacpp_model(self.ctx.base_url, new_model, api_key=self.ctx.api_key)
                else:
                    model_exists = is_available_llamacpp_model(self.ctx.base_url, new_model)

                if model_exists:
                    self.ctx.model = new_model
                    self.ctx.context_window_size = 0

                    if hasattr(self, 'messages') and self.messages:
                        self.ctx.current_context_tokens = self.ctx.calculate_context_tokens(self.messages)
                    else:
                        self.ctx.current_context_tokens = 0

                    if self.ctx.backend == "ollama":
                        all_models = fetch_models_ollama(self.ctx.base_url)
                        model_size = 0
                        for m in all_models:
                            if m.get('name') == new_model:
                                model_size = m.get('size', 0)
                                break

                        if model_size:
                            size_str = parse_size(model_size)
                            print(colorize(f"[Loading   '{new_model}' ({size_str})]", 'muted'), file=sys.stderr)
                        else:
                            print(colorize(f"[Loading   '{new_model}']", 'muted'), file=sys.stderr)
                    else:
                        refresh_context_window_size(self.ctx)

                    if self.ctx.backend not in ("gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
                        # Warmup ping to trigger model loading (not needed for cloud APIs)
                        if hasattr(self, 'messages') and self.messages:
                            ping_messages = list(self.messages)
                        else:
                            ping_messages = [{"role": "system", "content": self.ctx.system_prompt}]
                        res_ping = self.query_handler.query_sync(ping_messages, new_model, stream_enabled=False, is_warmup=True)

                        if self.ctx.backend == "ollama":
                            ptokens = res_ping.get("prompt_eval_count", 0)
                        else:
                            usage_block = res_ping.get("usage", {})
                            ptokens = usage_block.get("prompt_tokens", 0)
                        if ptokens > 0:
                            self.ctx.current_context_tokens = ptokens

                    print(colorize(f"[Switched to '{new_model}']", 'success'), file=sys.stderr)
                    return False
                else:
                    print(colorize(f"[ERROR] Model '{new_model}' not found on server", 'error'), file=sys.stderr)
                    print(colorize("Use /listmodel to see available models", 'muted'), file=sys.stderr)
                    return False
            else:
                print(colorize("[ERROR] Usage: /switchmodel <model_name>", 'error'), file=sys.stderr)
                return False
        return None

    def run_handle_agentic(self, full_input: str) -> Optional[bool]:
        """Handle /agentic command and subcommands (like /debug)."""
        if not full_input.startswith('/agentic'):
            return None
        parts = full_input.split()

        # Bare /agentic or /agentic status -> show status
        if len(parts) == 1 or (len(parts) >= 2 and parts[1] == 'status'):
            self._print_agentic_status()
            return False

        subcmd = parts[1]

        if subcmd in ("on", "off"):
            return self._agentic_toggle_on_off(subcmd == "on")
        if subcmd == "full":
            return self._agentic_set_full()
        if subcmd == "sandbox":
            return self._agentic_toggle_sandbox()
        if subcmd in ("iterations", "timeout", "heartbeattokens"):
            return self._agentic_set_number(subcmd, parts)
        if subcmd == "compactthreshold":
            return self._agentic_set_threshold(parts)
        if subcmd in self._AGENTIC_TOGGLE_MAP:
            return self._agentic_toggle_named(subcmd)
        if subcmd == "acl":
            return self._handle_agentic_acl(parts)

        print(colorize("[Usage: /agentic [on|off|full|auto|sandbox|verbose|thinking|trace|log|lazytool|acl|iterations <N>|timeout <N>|heartbeattokens <N>|compactthreshold <v>|status]]", 'warning'), file=sys.stderr)
        return False

    # Named toggles (always toggle between on/off)
    _AGENTIC_TOGGLE_MAP = {
        "auto":     ("auto_confirm",        "Auto-confirm"),
        "verbose":  ("agentic_verbose",     "Verbose"),
        "thinking": ("agentic_show_thinking", "Show thinking"),
        "trace":    ("agentic_trace",        "Trace"),
        "log":      ("agentic_logging",      "Logging"),
        "lazytool": ("lazy_tool",            "Lazy tool extraction"),
    }

    def _agentic_toggle_on_off(self, target: bool) -> bool:
        """Turn agentic mode explicitly on or off, swapping the system prompt."""
        if self.ctx.agentic_mode == target:
            print(colorize(f"[Agentic mode already {'ON' if target else 'OFF'}]", 'muted'), file=sys.stderr)
            return False
        self.ctx.agentic_mode = target
        state = "ON" if self.ctx.agentic_mode else "OFF"
        if self.ctx.agentic_mode:
            self.ctx._saved_system_prompt = self.ctx.system_prompt
            self.ctx.system_prompt = get_agentic_prompt(self.ctx.model)
        else:
            if hasattr(self.ctx, '_saved_system_prompt'):
                self.ctx.system_prompt = self.ctx._saved_system_prompt
        print(colorize(f"[Agentic mode: {state}]", 'success' if self.ctx.agentic_mode else 'warning'), file=sys.stderr)
        return False

    def _agentic_set_full(self) -> bool:
        """Enable everything: agentic mode, verbose, thinking, trace, auto-confirm, lazy."""
        self.ctx._saved_system_prompt = self.ctx.system_prompt
        self.ctx.system_prompt = get_agentic_prompt(self.ctx.model)
        self.ctx.agentic_mode = True
        self.ctx.agentic_verbose = True
        self.ctx.agentic_show_thinking = True
        self.ctx.agentic_trace = True
        self.ctx.auto_confirm = True
        self.ctx.lazy_tool = True
        print(colorize("[Agentic mode: ON]", 'success'), file=sys.stderr)
        print(colorize("[System prompt switched to agentic mode]", 'muted'), file=sys.stderr)
        print(colorize("[Verbose: ON]", 'info'), file=sys.stderr)
        print(colorize("[Show thinking: ON]", 'info'), file=sys.stderr)
        print(colorize("[Trace: ON]", 'info'), file=sys.stderr)
        print(colorize("[Auto-confirm: ON]", 'info'), file=sys.stderr)
        print(colorize("[Lazy tool extraction: ON]", 'info'), file=sys.stderr)
        return False

    def _agentic_toggle_sandbox(self) -> bool:
        """Toggle the executor between container and host mode."""
        new_mode = "container" if self.executor.mode != "container" else "host"
        self.executor.mode = new_mode
        self.tool_registry.executor.mode = new_mode
        print(colorize(f"[Executor mode: {self.executor.mode}]", 'info'), file=sys.stderr)
        return False

    def _agentic_set_number(self, subcmd: str, parts: list) -> bool:
        """Set a numeric agentic setting (iterations, step timeout, heartbeat tokens)."""
        if len(parts) < 3 or not parts[2].isdigit():
            print(colorize(f"[Usage: /agentic {subcmd} <number>]", 'warning'), file=sys.stderr)
            return False
        val = max(1, int(parts[2]))
        attr_map = {
            "iterations": ("agentic_max_iterations", "Max iterations"),
            "timeout": ("agentic_step_timeout", "Step timeout"),
            "heartbeattokens": ("agentic_heartbeat_tokens", "Tokens per heartbeat dot"),
        }
        attr, label = attr_map[subcmd]
        setattr(self.ctx, attr, val)
        print(colorize(f"[{label}: {val}]", 'info'), file=sys.stderr)
        return False

    def _agentic_set_threshold(self, parts: list) -> bool:
        """Set the agentic auto-compact threshold (fraction or percent).

        Args:
            parts: Split /agentic args (parts[2] is the threshold value, e.g.
                `0.85` or `85`).

        Returns:
            False to keep the chat loop running.
        """
        if len(parts) < 3:
            print(colorize("[Usage: /agentic compactthreshold <value>]  (e.g. 0.85 or 85)", 'warning'), file=sys.stderr)
            return False
        try:
            val = float(parts[2])
        except ValueError:
            print(colorize(f"[Agentic] '{parts[2]}' is not a number.", 'error'), file=sys.stderr)
            return False
        if val > 1:
            val = val / 100.0
        if not (0.0 < val < 1.0):
            print(colorize("[Agentic] Threshold must be between 0 and 1 (or 1-99 as percent).", 'error'), file=sys.stderr)
            return False
        self.ctx.agentic_compact_threshold = val
        print(colorize(f"[Agentic] Auto-compact threshold set to {val:.0%}.", 'success'), file=sys.stderr)
        return False

    def _agentic_toggle_named(self, subcmd: str) -> bool:
        """Toggle one of the named boolean agentic settings."""
        attr, label = self._AGENTIC_TOGGLE_MAP[subcmd]
        new_val = not getattr(self.ctx, attr)
        setattr(self.ctx, attr, new_val)
        state = "ON" if new_val else "OFF"
        print(colorize(f"[{label}: {state}]", 'info'), file=sys.stderr)
        return False

    def _handle_agentic_acl(self, parts: list) -> bool:
        """Handle /agentic acl [list|status|log|allow|ask|deny <path> [read|write]|remove <path>|reset]."""
        acl = self.ctx.path_acl
        action = parts[2] if len(parts) > 2 else "list"
        if action in ("list", "status"):
            self._print_acl(acl)
            return False
        if action == "log":
            self._print_acl_log(acl)
            return False
        if action == "reset":
            acl.reset()
            print(colorize("[Path ACL reset to defaults]", 'info'), file=sys.stderr)
            self._print_acl(acl)
            return False
        if action in ("allow", "ask", "deny"):
            if len(parts) < 4:
                print(colorize(f"[Usage: /agentic acl {action} <path> [read|write|any]]", 'warning'), file=sys.stderr)
                return False
            path = parts[3]
            op = "any" if len(parts) < 5 else parts[4]
            if op not in ("read", "write", "any"):
                print(colorize(f"[Invalid op '{op}'. Use read, write or any]", 'warning'), file=sys.stderr)
                return False
            rp = os.path.realpath(os.path.expanduser(path))
            acl.add(op, action, path, source="session")
            print(colorize(f"[Path ACL] {op} {action} {rp}", 'success'), file=sys.stderr)
            return False
        if action == "remove":
            if len(parts) < 4:
                print(colorize("[Usage: /agentic acl remove <path>]", 'warning'), file=sys.stderr)
                return False
            rp = os.path.realpath(os.path.expanduser(parts[3]))
            acl.remove(parts[3])
            print(colorize(f"[Path ACL] removed rules for {rp}", 'info'), file=sys.stderr)
            return False
        print(colorize("[Usage: /agentic acl [list|status|log|allow|ask|deny <path> [read|write|any]|remove <path>|reset]]", 'warning'), file=sys.stderr)
        return False

    def _print_acl(self, acl: 'PathAcl') -> None:
        """Print the current path ACL rules table.

        Args:
            acl: The PathAcl instance to display.
        """
        print(colorize("\n[Path ACL - Use /agentic acl <allow|ask|deny|remove|reset|log>]", 'info'), file=sys.stderr)
        print("  Defaults: CWD always allowed; reads inside ~ allowed.", file=sys.stderr)
        print("  Everything else outside CWD/~ is denied unless a rule matches.", file=sys.stderr)
        print("  'ask' rules prompt per access and are never auto-approved.", file=sys.stderr)
        print(file=sys.stderr)
        print(f"  {'OP':<6} {'ACTION':<7} {'PATH':<44} SOURCE", file=sys.stderr)
        print("  " + "-" * 74, file=sys.stderr)
        for r in acl.rules:
            shown = r["path"] if r.get("kind") == "prefix" else "~/.* (home dotfile)"
            print(f"  {r['op']:<6} {r['action']:<7} {shown:<44} {r['source']}", file=sys.stderr)
        print(file=sys.stderr)
        print("  Recent access decisions: /agentic acl log", file=sys.stderr)
        print(file=sys.stderr)

    def _print_acl_log(self, acl: 'PathAcl') -> None:
        """Print recent path ACL access decisions.

        Args:
            acl: The PathAcl instance whose log to display.
        """
        if not acl.log:
            print(colorize("[Path ACL] No access decisions recorded yet.", 'muted'), file=sys.stderr)
            return
        print(colorize("\n[Path ACL - recent access decisions]", 'info'), file=sys.stderr)
        for entry in reversed(acl.log[-20:]):
            print(f"  {entry['ts']} {entry['decision']:<6} {entry['tool']:<14} {entry['path']}  ({entry['rule'] or 'default'})", file=sys.stderr)
        print(file=sys.stderr)

    def _print_agentic_status(self) -> None:
        """Display current agentic settings like /debug output."""
        c = self.ctx
        print(colorize("\n[Agentic Settings - Use /agentic <option> [value]]", 'info'), file=sys.stderr)
        print("  Subcommands: on, off, full, auto, sandbox, verbose, thinking, trace, log, lazytool,", file=sys.stderr)
        print("               iterations <N>, timeout <N>, heartbeattokens <N>, compactthreshold <v>, acl, status", file=sys.stderr)
        print(file=sys.stderr)

        settings = [
            ("agentic",    "Master toggle",             str(c.agentic_mode).lower()),
            ("auto",       "Skip destructive tool confirmation", str(c.auto_confirm).lower()),
            ("sandbox",    "Run tool subprocesses in container", self.executor.mode),
            ("verbose",    "Show raw model responses during ReAct", str(c.agentic_verbose).lower()),
            ("thinking",   "Show model reasoning during ReAct", str(c.agentic_show_thinking).lower()),
            ("trace",      "Show full tool args and results", str(c.agentic_trace).lower()),
            ("log",        "Write structured JSONL logs", str(c.agentic_logging).lower()),
            ("lazytool",   "Extract tool calls from anywhere in reply", str(c.lazy_tool).lower()),
            ("iterations", "Max ReAct loop iterations", str(c.agentic_max_iterations)),
            ("timeout",    "Per-step timeout (seconds)", f"{c.agentic_step_timeout}s"),
            ("heartbeattokens", "Tokens per heartbeat dot", str(c.agentic_heartbeat_tokens)),
            ("compactthreshold", "Agentic auto-compact trigger", f"{c.agentic_compact_threshold:.0%}"),
        ]
        for name, desc, value in settings:
            marker = ">" if (value not in ("off", "false", "host", "0") and "off" not in value) else " "
            print(f"  {marker} {name:<12} [{value:<8}] {desc}", file=sys.stderr)
        print()

    def run_handle_listtool(self, full_input: str) -> Optional[bool]:
        """Handle /listtool command."""
        if full_input.strip() != '/listtool':
            return None
        header = f"Available Tools (agentic: {'ON' if self.ctx.agentic_mode else 'OFF'})"
        print(colorize(f"\n{header}", 'info'), file=sys.stderr)
        print(colorize("\u2500" * 70, 'muted'), file=sys.stderr)
        print(self.tool_registry.list_tools_str(), file=sys.stderr)
        print(file=sys.stderr)
        return False

    @staticmethod
    def _normalize_tool_json(json_text: str) -> Optional[dict]:
        """Parse JSON and normalize any tool call format to {"tool": ..., "arguments": ...}.

        Supports:
        - Internal:  {"tool": "name", "arguments": {...}}
        - OpenAI:    {"type": "function", "function": {"name": "name", "arguments": {...}}}
        - Compact:   {"function": {"name": "name", "arguments": {...}}}
        Arguments can be a dict or a JSON string.
        """
        try:
            obj = json.loads(json_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        if "tool" in obj:
            return obj
        tool_name = None
        tool_args = {}
        if "function" in obj and isinstance(obj["function"], dict):
            fn = obj["function"]
            if "name" in fn:
                tool_name = fn["name"]
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        tool_args = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    tool_args = raw_args
        # OpenAI format without function wrapper: {"type": "function", "name": "...", "arguments": {...}}
        if not tool_name and obj.get("type") == "function" and isinstance(obj.get("name"), str):
            tool_name = obj["name"]
            raw_args = obj.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    tool_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    tool_args = {"raw": raw_args}
            elif isinstance(raw_args, dict):
                tool_args = raw_args
        if tool_name:
            return {"tool": tool_name, "arguments": tool_args}
        return None

    @staticmethod
    def _find_tool_call_brace(text: str, pos: int = 0) -> int:
        """Find the next { that introduces a tool call JSON (with tool/function/type key).

        Args:
            text: The response text to search.
            pos: Starting offset.

        Returns:
            Index of the opening brace, or -1 if none found.
        """
        idx = text.find('{', pos)
        while idx != -1:
            rest = text[idx+1:].lstrip()
            if rest.startswith(('"tool"', '"function"', '"type"')):
                return idx
            idx = text.find('{', idx + 1)
        return -1

    @staticmethod
    def _rfind_tool_call_brace(text: str) -> int:
        """Find the last { that introduces a tool call JSON, scanning right-to-left.

        Args:
            text: The response text to search.

        Returns:
            Index of the last tool-call opening brace, or -1 if none found.
        """
        idx = text.rfind('{')
        while idx != -1:
            rest = text[idx+1:].lstrip()
            if rest.startswith(('"tool"', '"function"', '"type"')):
                return idx
            if idx == 0:
                break
            idx = text.rfind('{', 0, idx)
        return -1

    @staticmethod
    def _extract_json_balanced(text: str, start: int) -> Optional[str]:
        """Extract balanced JSON from text starting at an opening brace.

        Args:
            text: The response text.
            start: Index of the opening brace.

        Returns:
            The balanced JSON substring, or None if unbalanced.

        Handles braces inside JSON string values correctly by tracking
        string boundaries and escape sequences.
        """
        depth = 0
        in_string = False
        escaped = False
        for i, ch in enumerate(text[start:], start):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == '\\':
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        return None

    @staticmethod
    def _tool_call_json_acceptable(result: Optional[dict]) -> bool:
        """Reject tool-call JSON that looks like an XML/HTML hybrid fragment.

        Args:
            result: Normalized tool call dict (or None).

        Returns:
            True when the result should be accepted.
        """
        if not result:
            return False
        args_str = json.dumps(result.get("arguments", {}))
        return not ('<' in args_str and '>' in args_str)

    def _parse_fenced_tool_call(self, text: str, lazy: bool) -> Optional[dict]:
        """Pass 2: extract a tool call from a fenced ```json code block.

        Args:
            text: The response text.
            lazy: Whether lazy extraction is enabled.

        Returns:
            A normalized tool call dict, or None.
        """
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            if block.startswith("json"):
                block = block[4:].strip()
            elif block.startswith("text"):
                block = block[4:].strip()
            brace_idx = ChatLoop._find_tool_call_brace(block)
            if brace_idx != -1:
                prefix = "```".join(parts[:i])
                suffix = "```".join(parts[i+1:])
                accept = lazy
                if not accept:
                    accept = ((not prefix or len(prefix) <= 100) and not suffix)
                if accept:
                    json_str = self._extract_json_balanced(block, brace_idx)
                    if json_str:
                        result = self._normalize_tool_json(json_str)
                        if self._tool_call_json_acceptable(result):
                            return result
        return None

    def _parse_bare_tool_call(self, text: str, lazy: bool) -> Optional[dict]:
        """Pass 3: extract a bare JSON tool call (anywhere in lazy mode).

        Args:
            text: The response text.
            lazy: Whether lazy extraction is enabled.

        Returns:
            A normalized tool call dict, or None.
        """
        if lazy:
            brace_idx = ChatLoop._find_tool_call_brace(text)
            if brace_idx != -1:
                json_str = self._extract_json_balanced(text, brace_idx)
                if json_str:
                    result = self._normalize_tool_json(json_str)
                    if self._tool_call_json_acceptable(result):
                        return result
        else:
            # Strict mode: only at START of the text
            stripped = text.lstrip()
            if stripped.startswith('{') and stripped[1:].lstrip().startswith(('"tool"', '"function"', '"type"')):
                start = text.index('{')
                json_str = self._extract_json_balanced(text, start)
                if json_str:
                    result = self._normalize_tool_json(json_str)
                    if result:
                        return result
            # Pass 4: bare JSON tool call at the END of the text (strict mode only).
            # The JSON must be the final content (no trailing text), but a short
            # prose preamble is allowed — models often narrate intent ("Let me
            # check...") then emit the JSON tool call as the last thing.
            idx = ChatLoop._rfind_tool_call_brace(text)
            if idx >= 0:
                json_str = self._extract_json_balanced(text, idx)
                if json_str:
                    end = idx + len(json_str)
                    if not text[end:].strip():
                        result = self._normalize_tool_json(json_str)
                        if self._tool_call_json_acceptable(result):
                            return result
        return None

    def parse_tool_call(self, text: str) -> Optional[dict]:
        """Extract tool call JSON from LLM response. Returns {"tool": ..., "arguments": ...} or None.

        Args:
            text: The response text to parse.
        """
        lazy = getattr(self.ctx, 'lazy_tool', False)
        text = text.strip()
        # Pass 1: full JSON parse
        result = self._normalize_tool_json(text)
        if result:
            return result
        # Pass 2: fenced JSON code block
        result = self._parse_fenced_tool_call(text, lazy)
        if result:
            return result
        # Pass 3: bare JSON object with "tool" key
        return self._parse_bare_tool_call(text, lazy)

    def _extract_tool_calls_lazy(self, text: str) -> list:
        """Lazy mode: find ALL tool call JSONs anywhere in the text (deduped).

        Args:
            text: The response text.

        Returns:
            List of normalized tool call dicts.
        """
        results = []
        seen_keys = set()
        pos = 0
        while pos < len(text):
            idx = ChatLoop._find_tool_call_brace(text, pos)
            if idx == -1:
                break
            json_str = self._extract_json_balanced(text, idx)
            if not json_str:
                pos = idx + 1
                continue
            result = self._normalize_tool_json(json_str)
            if result:
                key = (result["tool"], json.dumps(result.get("arguments", {}), sort_keys=True))
                if key not in seen_keys:
                    seen_keys.add(key)
                    results.append(result)
            pos = idx + len(json_str) if json_str else idx + 1
        return results

    def _extract_tool_calls_strict(self, text: str) -> list:
        """Strict mode: only consecutive tool calls from the start.

        Args:
            text: The response text.

        Returns:
            List of normalized tool call dicts (empty when the text has preamble).
        """
        results = []
        # Strict mode: only consecutive tool calls from the start
        stripped = text.lstrip()
        if not (stripped.startswith('{') and stripped[1:].lstrip().startswith(('"tool"', '"function"', '"type"'))):
            return []
        pos = 0
        while pos < len(text):
            # Skip whitespace between consecutive tool calls
            while pos < len(text) and text[pos] in (' ', '\t', '\n', '\r'):
                pos += 1
            if pos >= len(text):
                break
            # Check if next non-whitespace is a tool call
            if not (text[pos] == '{' and text[pos+1:].lstrip().startswith(('"tool"', '"function"', '"type"'))):
                break
            json_str = self._extract_json_balanced(text, pos)
            if not json_str:
                break
            result = self._normalize_tool_json(json_str)
            if result:
                results.append(result)
            else:
                break
            pos += len(json_str)
        return results

    def parse_tool_calls(self, text: str) -> list[dict]:
        """Extract ALL tool call JSONs from the response.

        First tries direct JSON parse (works for clean tool calls at start of text).
        Falls back to strict/lazy extraction for embedded or malformed JSON.

        In strict mode (default): only matches consecutive tool calls starting
        from the beginning of the response (no preamble).
        In lazy mode (/agentic lazytool): finds tool calls anywhere in the text.
        Returns list of {"tool": ..., "arguments": ...} dicts.

        Args:
            text: The response text to parse.
        """
        lazy = getattr(self.ctx, 'lazy_tool', False)
        text = text.strip()

        # Pass 0: Quick check — if text starts with '{', try direct JSON parse.
        # A proper JSON parser handles braces inside string values correctly,
        # unlike brace-counting approaches.
        if text.startswith('{'):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and 'tool' in obj:
                    return [obj]
            except (json.JSONDecodeError, ValueError):
                pass

        if lazy:
            return self._extract_tool_calls_lazy(text)
        return self._extract_tool_calls_strict(text)

    @staticmethod
    def _is_stuck(text: str, threshold: float = 0.8) -> bool:
        """Detect if the model is repeating itself using sliding-window frequency check.

        Checks if the last ~400 chars contain a phrase that appears 3+ times.
        Uses multiple window sizes to catch variable-length repetition periods.
        """
        if len(text) < 200:
            return False
        tail = text[-400:]
        half_len = len(tail) // 2
        for window_size in (30, 50, 80):
            if window_size > half_len:
                continue
            match_tail = tail[-window_size:]
            if not match_tail.strip():
                continue  # whitespace/indentation window — not a stuck loop
            if tail.count(match_tail) >= 3:
                return True
        return False

    @staticmethod
    def _call_with_timeout(func: object, timeout_sec: int, *args: tuple, **kwargs: dict) -> object:
        """Call a function with a wall-clock timeout using a daemon thread.

        Args:
            func: The callable to invoke.
            timeout_sec: Timeout in seconds.
            *args, **kwargs: Passed to func.

        Returns:
            The function's return value, or None on timeout.
        """
        result = [None]
        exception = [None]

        def worker() -> None:
            """Run the target function in the daemon thread, capturing result/exception."""
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout_sec)
        if thread.is_alive():
            return None  # timed out
        if exception[0]:
            raise exception[0]
        return result[0]

    def _accumulate_step_feedback(self, state: dict, thought: str, content: str) -> None:
        """Accumulate thought/content into the step buffer with F5 cap tracking.

        Args:
            state: The step feedback buffer dict (mutated in place).
            thought: Reasoning text for this chunk.
            content: Assistant text for this chunk.
        """
        if thought:
            state["thought"] += thought
            state["last_activity"] = time.time()
            # F5: track an estimate of total thinking tokens so a single step
            # can't silently burn the whole budget on reasoning undetected.
            # Detection + flag only — we keep accumulating (F4 tail-caps the
            # nudge) and keep streaming live thinking; the flag is surfaced to
            # the timeout nudge so a verbose thinker is told to be concise.
            cap = int(getattr(self.ctx, 'agentic_max_thinking_tokens', 0) or 0)
            if cap > 0:
                state["thinking_tokens"] += max(1, len(thought) // 4)
                if state["thinking_tokens"] >= cap:
                    state["thinking_capped"] = True
        if content:
            state["content"] += content
            state["last_activity"] = time.time()

    def _render_step_feedback(self, state: dict, thought: str, content: str,
                              show_thinking: bool) -> None:
        """Render live feedback for a chunk (thinking stream or heartbeat dot).

        Args:
            state: The step feedback buffer dict (mutated in place).
            thought: Reasoning text for this chunk.
            content: Assistant text for this chunk.
            show_thinking: Whether to stream the reasoning live.

        Note: when thinking is not streamed, liveness dots come from the
        watchdog thread started in `_make_agentic_step_feedback`, not from
        chunk arrival — long tool-call generations may deliver no chunks at
        all, so a chunk-driven heartbeat would leave the terminal silent.
        """
        if show_thinking:
            if thought:
                with state["write_lock"]:
                    if not state["thinking_open"]:
                        state["thinking_open"] = True
                        sys.stderr.write("\n<thinking>\n")
                    sys.stderr.write(thought)
                    sys.stderr.flush()
            if content and state["thinking_open"]:
                with state["write_lock"]:
                    sys.stderr.write("\n</thinking>\n")
                    sys.stderr.flush()
                    state["thinking_open"] = False

    def _make_agentic_step_feedback(self) -> tuple:
        """Build live feedback callbacks for an agentic ReAct step.

        Returns (on_chunk, finalize, buffer). `on_chunk(thought, content, is_final)`
        streams model thinking to stderr in real time when `agentic_show_thinking`
        is enabled. Heartbeat dots are token-driven: one dot per
        `ctx.agentic_heartbeat_tokens` (default 10) streamed tokens, so the
        cadence reflects real generation progress rather than wall-clock. Because
        some generations deliver no chunks at all (e.g. llama.cpp buffering a
        tool-call response for tens of seconds), a daemon watchdog thread also
        emits a dot (~1/s) once the step has been silent for ~2s, resuming as
        soon as tokens flow again. While thinking is being streamed live both
        mechanisms hold off (the user already sees progress). `buffer` is a dict
        with accumulated `content`/`thought` — used after a timeout to decide
        abort-vs-continue. `finalize()` closes any open `<thinking>` block and
        stops the watchdog + lingering worker thread (on timeout) from writing
        further feedback.

        Returns:
            (on_chunk, finalize, buffer) tuple of callables + state dict.
        """
        state = {
            "thinking_open": False,
            "last_activity": time.time(),
            "cancelled": False,
            "heartbeat_shown": False,
            "content": "",
            "thought": "",
            "thinking_tokens": 0,
            "thinking_capped": False,
            "tok_acc": 0,
            "write_lock": threading.Lock(),
        }
        show_thinking = self.ctx.agentic_show_thinking
        heartbeat_tokens = max(1, int(getattr(self.ctx, 'agentic_heartbeat_tokens', 10) or 10))

        def watchdog() -> None:
            """Idle-fallback liveness: dot ~1/s while the step is silent.

            Token-driven dots (see on_chunk) cover active streaming; generations
            that deliver no chunks at all would otherwise leave the terminal
            silent, so after ~2s without token activity the watchdog ticks once
            per second until tokens resume. This fires even while the
            `<thinking>` block is open: a silent thinking phase usually means the
            model has switched to a tool call (native tool-call deltas carry no
            content, so the block never closes on its own) — the block is closed
            and the dots keep the user informed. If thinking later resumes, the
            block reopens.
            """
            idle_grace = 2.0
            while not state["cancelled"]:
                time.sleep(1.0)
                if state["cancelled"]:
                    return
                if time.time() - state["last_activity"] < idle_grace:
                    continue
                with state["write_lock"]:
                    if state["cancelled"]:
                        return
                    if state["thinking_open"]:
                        sys.stderr.write("\n</thinking>\n")
                        state["thinking_open"] = False
                    state["heartbeat_shown"] = True
                    sys.stderr.write(".")
                    sys.stderr.flush()

        threading.Thread(target=watchdog, daemon=True).start()

        def on_chunk(thought: str, content: str, is_final: bool) -> None:
            """Handle a parsed stream chunk: accumulate + render feedback.

            Args:
                thought: Reasoning text for this chunk.
                content: Assistant text for this chunk.
                is_final: Whether this is the final chunk.
            """
            if state["cancelled"]:
                return
            self._accumulate_step_feedback(state, thought, content)
            self._render_step_feedback(state, thought, content, show_thinking)
            if (thought or content) and not state["thinking_open"]:
                state["tok_acc"] += max(1, len(thought) // 4) + max(1, len(content) // 4)
                while state["tok_acc"] >= heartbeat_tokens:
                    state["tok_acc"] -= heartbeat_tokens
                    state["heartbeat_shown"] = True
                    with state["write_lock"]:
                        sys.stderr.write(".")
                        sys.stderr.flush()

        def finalize() -> None:
            """Close any open thinking block and stop further feedback output."""
            with state["write_lock"]:
                if state["thinking_open"]:
                    sys.stderr.write("\n</thinking>\n")
                    sys.stderr.flush()
                    state["thinking_open"] = False
                elif state["heartbeat_shown"]:
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                    state["heartbeat_shown"] = False
                state["cancelled"] = True

        return on_chunk, finalize, state

    def _partial_tool_call_pending(self, partial_text: str) -> bool:
        """Check whether a timed-out generation contains a parseable tool call.

        Distinguishes "model was narrating prose (safe to retry)" from "model was
        about to emit a tool call that a retry could re-execute".
        """
        if not partial_text or not partial_text.strip():
            return False
        try:
            calls = self.parse_tool_calls(partial_text)
            if not calls:
                single = self.parse_tool_call(partial_text)
                if single:
                    calls = [single]
        except Exception:
            return False
        return bool(calls)

    @staticmethod
    def _looks_like_truncated_tool_call(text: str) -> bool:
        """Detect an in-flight (unbalanced) tool-call JSON opening in partial text.

        `_partial_tool_call_pending` only sees *parseable* tool calls. A model cut
        off mid-emission leaves an unbalanced `{"tool": ...` opening that would
        parse as nothing — but echoing that prefix back invites the model to
        continue the JSON from the middle and emit a malformed call. This catches
        the unbalanced opening so the nudge can omit partial content.
        """
        if not text or not text.strip():
            return False
        try:
            idx = ChatLoop._find_tool_call_brace(text)
            if idx == -1:
                return False
            return ChatLoop._extract_json_balanced(text, idx) is None
        except Exception:
            return False

    @staticmethod
    def _timeout_nudge(partial_content: str, partial_thought: str, pending_tool: bool,
                       metrics: Optional[dict] = None) -> str:
        """Build the timeout-continuation nudge message for the model.

        Feeds the model's partial reasoning/content back so it continues from
        where it left off instead of restarting its analysis from scratch
        (re-thinking the same problem burns a full generation budget). Partial
        *content* is omitted when a tool call was pending — feeding a cut-off
        JSON prefix back invites a malformed JSON continuation. The reasoning is
        always included; it is the model's internal monologue, not the output
        channel, so echoing it is safe and preserves the in-flight train of
        thought.

        Two refinements:
          - F4: the echoed reasoning is capped from the **tail** (`[-4000:]`,
            where the model was cut off) rather than the head, so a resumed step
            continues instead of re-deriving the earlier analysis.
          - F3: an optional `metrics` dict (`tok_per_sec`, `expired_sec`,
            `budget_sec`) tells the model its actual generation speed, the timeout
            it hit, and the new budget in seconds/tokens so it can self-throttle
            and finish within budget instead of getting killed again.

        Args:
            partial_content: Partial text content from the step buffer.
            partial_thought: Partial reasoning (`reasoning_content`) from the step buffer.
            pending_tool: True if the partial content contained a parseable tool call.
            metrics: Optional dict with tok_per_sec / expired_sec / budget_sec.

        Returns:
            str: The nudge message to append as a user turn.
        """
        def _cap_tail(text: str, cap: int = 4000) -> str:
            """Cap text from the tail, marking omitted leading content."""
            text = text.strip()
            if len(text) <= cap:
                return text
            return "…[earlier reasoning omitted]\n" + text[-cap:]

        lines = []
        thought = partial_thought.strip()
        if thought:
            lines.append("Your reasoning before the interruption:\n" + _cap_tail(thought))
        if not pending_tool:
            content = partial_content.strip()
            if content:
                lines.append("Your response before the interruption:\n" + _cap_tail(content))
        base = ("You were interrupted by a timeout while generating your response. "
                "Please continue your previous response.")

        if metrics and metrics.get("budget_sec"):
            tps = metrics.get("tok_per_sec", 0.0) or 0.0
            expired = metrics.get("expired_sec", 0) or 0
            budget = metrics.get("budget_sec", 0) or 0
            est_tokens = int(budget * tps)
            base += ("\n\nGeneration metrics:\n"
                     f"- ~{tps:.1f} tokens/second\n"
                     f"- cut off after {expired}s (step timeout)\n"
                     f"- new budget: {budget}s (~{est_tokens} tokens at your current speed)\n")
            if metrics.get("thinking_capped"):
                cap = metrics.get("thinking_cap") or 0
                if cap:
                    base += (f"- note: your reasoning exceeded the {cap}-token cap — be more "
                             "concise and move to a tool call or final answer\n")
                else:
                    base += ("- note: your reasoning is very long — be more concise and move to "
                             "a tool call or final answer\n")
            base += ("Keep your total thinking + output within that budget — finish as soon "
                     "as you have enough to act, and prefer a tool call or final answer over "
                     "re-deriving your full analysis.")
        if lines:
            return base + "\n\n" + "\n\n".join(lines)
        return base

    def _agentic_timeout_state_c(self, iteration: int, expired: int, last_tool: str,
                                 pending_tool: bool, step_timeout: int, messages: list,
                                 partial_text: str, partial_thought: str,
                                 content_is_tool: bool, _nudge_metrics: object) -> tuple:
        """State C policy: last tool was destructive.

        Args:
            iteration: Current iteration.
            expired: The timeout that just expired (seconds).
            last_tool: Name of the last executed tool.
            pending_tool: Whether a tool call was in flight in the partial output.
            step_timeout: The current step budget.
            messages: Conversation list (nudge appended on continue).
            partial_text / partial_thought / content_is_tool: Nudge inputs.
            _nudge_metrics: Callable building the metrics dict for the nudge.

        Returns:
            (should_break, step_timeout).
        """
        if pending_tool:
            # State C: destructive last tool AND a tool call was in flight —
            # retrying risks re-executing it.
            print(colorize(
                f"\n[Agentic] Step {iteration} timed out after {expired}s. "
                f"Last tool was '{last_tool}' (destructive) and a tool call was "
                "pending. Aborting to avoid re-execution.",
                'error'), file=sys.stderr)
            return True, step_timeout
        # Destructive tool already completed; the model was only narrating.
        # Continue with State B backoff.
        print(colorize(
            f"\n[Agentic] Step {iteration} timed out after {expired}s. "
            f"Last tool was '{last_tool}' (destructive) but no tool call was "
            "pending — continuing with backoff.",
            'warning'), file=sys.stderr)
        return self._agentic_timeout_state_b(
            iteration, expired, messages, step_timeout,
            partial_text, partial_thought, content_is_tool, _nudge_metrics)

    def _agentic_timeout_state_a(self, iteration: int, expired: int, messages: list,
                                 step_timeout: int, making_progress: bool, tok_per_sec: float,
                                 partial_text: str, partial_thought: str,
                                 content_is_tool: bool, _nudge_metrics: object) -> tuple:
        """State A policy: no tool executed yet.

        Extends the budget while the model is actively generating (F3), doubling
        `agentic_timeout_max` up to `AGENTIC_TIMEOUT_MAX_CEILING` when the ceiling
        is reached; otherwise retries once and aborts after two consecutive
        timeouts.

        Args:
            iteration: Current iteration.
            expired: The timeout that just expired.
            messages: Conversation list (nudge appended on continue).
            step_timeout: The current step budget.
            making_progress: Whether tokens were still flowing when the timer fired.
            tok_per_sec: Estimated generation speed.
            partial_text / partial_thought / content_is_tool: Nudge inputs.
            _nudge_metrics: Callable building the metrics dict for the nudge.

        Returns:
            (should_break, step_timeout).
        """
        if making_progress and step_timeout < self.ctx.agentic_timeout_max:
            # F3: model is actively generating (tokens flowing) — extend the
            # budget instead of retrying at the same (insufficient) timeout,
            # which would just fail again. Only thinking-only steps benefit
            # from this; a stalled step falls through to the retry/abort.
            step_timeout = min(step_timeout * 2, self.ctx.agentic_timeout_max)
            print(colorize(
                f"\n[Agentic] Step {iteration} timed out after {expired}s but still "
                f"generating ({tok_per_sec:.1f} tok/s) — extending to {step_timeout}s.",
                'warning'), file=sys.stderr)
            messages.append({'role': 'user', 'content': self._timeout_nudge(
                partial_text, partial_thought, content_is_tool,
                metrics=_nudge_metrics(step_timeout)), '_system_nudge': True})
            return False, step_timeout
        if making_progress and self.ctx.agentic_timeout_max < AGENTIC_TIMEOUT_MAX_CEILING:
            # At the ceiling but still generating — raise the ceiling (capped at
            # the hard ceiling) so a genuinely long think can finish.
            self.ctx.agentic_timeout_max = min(
                self.ctx.agentic_timeout_max * 2, AGENTIC_TIMEOUT_MAX_CEILING)
            step_timeout = self.ctx.agentic_timeout_max
            print(colorize(
                f"\n[Agentic] Step {iteration} timed out after {expired}s but still "
                f"generating ({tok_per_sec:.1f} tok/s) — raising max timeout to "
                f"{self.ctx.agentic_timeout_max}s.", 'warning'), file=sys.stderr)
            messages.append({'role': 'user', 'content': self._timeout_nudge(
                partial_text, partial_thought, content_is_tool,
                metrics=_nudge_metrics(step_timeout)), '_system_nudge': True})
            return False, step_timeout
        # No recent progress, or the hard timeout ceiling is reached →
        # conservative; abort after two timeouts.
        if self.ctx.agentic_consecutive_timeouts >= 2:
            print(colorize(
                f"\n[Agentic] Step {iteration} timed out twice. Model may be stuck. Aborting.",
                'error'), file=sys.stderr)
            return True, step_timeout
        print(colorize(
            f"\n[Agentic] Step {iteration} timed out after {expired}s. "
            "Model may be thinking — retrying.",
            'warning'), file=sys.stderr)
        messages.append({'role': 'user', 'content': self._timeout_nudge(
            partial_text, partial_thought, content_is_tool,
            metrics=_nudge_metrics(step_timeout)), '_system_nudge': True})
        return False, step_timeout

    def _agentic_timeout_state_b(self, iteration: int, expired: int, messages: list,
                                 step_timeout: int, partial_text: str, partial_thought: str,
                                 content_is_tool: bool, _nudge_metrics: object) -> tuple:
        """State B policy: a tool already ran — exponential backoff, raising the
        max ceiling (doubling it, capped at `AGENTIC_TIMEOUT_MAX_CEILING`) when
        the current budget has reached it.

        Args:
            iteration: Current iteration.
            expired: The timeout that just expired.
            messages: Conversation list (nudge appended on continue).
            step_timeout: The current step budget.
            partial_text / partial_thought / content_is_tool: Nudge inputs.
            _nudge_metrics: Callable building the metrics dict for the nudge.

        Returns:
            (should_break, step_timeout).
        """
        if step_timeout >= self.ctx.agentic_timeout_max:
            if self.ctx.agentic_timeout_max >= AGENTIC_TIMEOUT_MAX_CEILING:
                # Hard ceiling reached — stop growing and abort.
                print(colorize(
                    f"\n[Agentic] Max timeout ceiling ({AGENTIC_TIMEOUT_MAX_CEILING}s) "
                    "reached. Aborting agentic query.", 'error'), file=sys.stderr)
                return True, step_timeout
            # Already at the ceiling — raise the ceiling by doubling it (capped
            # at the hard ceiling) rather than aborting, so a slow-but-progressing
            # model can finish.
            self.ctx.agentic_timeout_max = min(
                self.ctx.agentic_timeout_max * 2, AGENTIC_TIMEOUT_MAX_CEILING)
            step_timeout = self.ctx.agentic_timeout_max
            print(colorize(
                f"\n[Agentic] Step {iteration} timed out after {expired}s. "
                f"Model was executing tools — raising max timeout to {self.ctx.agentic_timeout_max}s.",
                'warning'), file=sys.stderr)
        else:
            step_timeout = min(step_timeout * 2, self.ctx.agentic_timeout_max)
            print(colorize(
                f"\n[Agentic] Step {iteration} timed out after {expired}s. "
                f"Model was executing tools — extending to {step_timeout}s.",
                'warning'), file=sys.stderr)

        messages.append({'role': 'user', 'content': self._timeout_nudge(
            partial_text, partial_thought, content_is_tool,
            metrics=_nudge_metrics(step_timeout)), '_system_nudge': True})
        return False, step_timeout

    def _handle_agentic_timeout(self, iteration: int, step_timeout: int, messages: list,
                                partial_text: str = "", partial_thought: str = "",
                                elapsed_sec: Optional[float] = None,
                                partial_tokens: Optional[int] = None,
                                idle_sec: Optional[float] = None,
                                thinking_capped: bool = False) -> tuple:
        """Handle a step timeout in the ReAct loop.

        Applies the escalation policy based on model state:
          State C — last tool was destructive → abort ONLY if a tool call was
                    pending in the timed-out generation (a retry could re-execute
                    it). If the model was mid-narration with no tool call pending,
                    the destructive tool already completed, so continue with backoff.
          State A — no tool executed yet       → extend if generating, else retry once / abort after two.
          State B — a tool already ran         → exponential backoff; doubles the max ceiling (up to AGENTIC_TIMEOUT_MAX_CEILING) when reached.

        On any continue, the nudge message includes the model's partial reasoning
        (and partial content when no tool call was pending) so it resumes rather
        than regenerating the same analysis — long-thinking models that time out
        mid-thought would otherwise restart from scratch and hit the same wall.
        It also reports the model's generation speed and the new budget (F3
        budget-awareness) so it can self-throttle within the extended timeout.
        Nudges are tagged `_system_nudge` so they steer the in-flight loop but are
        excluded from the persisted conversation history.

        Args:
            iteration: Current ReAct iteration number.
            step_timeout: The timeout (seconds) that just expired.
            messages: The conversation list (a nudge message is appended on continue).
            partial_text: Content generated before the timeout (from the step buffer).
            partial_thought: Reasoning generated before the timeout (from the step buffer).
            elapsed_sec: Wall-clock seconds the step actually ran (for tok/s).
            partial_tokens: Estimated tokens generated before the timeout (for tok/s).
            idle_sec: Seconds since the last chunk arrived (progress/stall signal).

        Returns:
            (should_break: bool, new_step_timeout: int)
        """
        self.ctx.agentic_consecutive_timeouts += 1
        last_tool = self.ctx.agentic_last_tool_name
        expired = step_timeout
        pending_tool = self._partial_tool_call_pending(partial_text)
        content_is_tool = pending_tool or self._looks_like_truncated_tool_call(partial_text)

        # F3: generation-rate + progress signal from the step buffer.
        tok_per_sec = 0.0
        if elapsed_sec and elapsed_sec > 0 and partial_tokens and partial_tokens > 0:
            tok_per_sec = partial_tokens / elapsed_sec
        grace = max(0, int(getattr(self.ctx, 'agentic_progress_grace', 15)))
        making_progress = (idle_sec is not None and idle_sec <= grace and tok_per_sec > 0)

        def _nudge_metrics(budget_sec: int) -> dict:
            """Build the metrics dict for the timeout nudge message.

            Args:
                budget_sec: The new extended step budget in seconds.
            """
            m = {"tok_per_sec": tok_per_sec, "expired_sec": expired, "budget_sec": budget_sec}
            if thinking_capped:
                m["thinking_capped"] = True
                m["thinking_cap"] = int(getattr(self.ctx, 'agentic_max_thinking_tokens', 0) or 0)
            return m

        if last_tool in AGENTIC_TIMEOUT_ABORT_TOOLS:
            return self._agentic_timeout_state_c(
                iteration, expired, last_tool, pending_tool, step_timeout, messages,
                partial_text, partial_thought, content_is_tool, _nudge_metrics)
        if not self.ctx.agentic_has_executed_tool:
            return self._agentic_timeout_state_a(
                iteration, expired, messages, step_timeout, making_progress, tok_per_sec,
                partial_text, partial_thought, content_is_tool, _nudge_metrics)
        return self._agentic_timeout_state_b(
            iteration, expired, messages, step_timeout,
            partial_text, partial_thought, content_is_tool, _nudge_metrics)

    def _init_agentic_query(self, final_content: str) -> Optional[tuple]:
        """Initialize messages, logger, and tool format for an agentic query.

        Args:
            final_content: The processed user query text.

        Returns:
            (messages, logger, openai_tools, send_tools_api) or None on early exit.
        """
        if not getattr(self, 'messages', None):
            self.messages = [{'role': 'system', 'content': self.ctx.system_prompt}]

        tool_format = get_tool_format(self.ctx.model)
        if tool_format == "openai":
            include_tool_defs = False
            send_tools_api = True
        else:
            include_tool_defs = True
            send_tools_api = False

        tool_defs_block = self.tool_registry.get_system_prompt_block() if include_tool_defs else ""
        messages = [{'role': 'system', 'content': get_agentic_prompt(self.ctx.model, tool_defs_block, include_tool_defs=include_tool_defs)}]
        if len(self.messages) > 1:
            messages.extend(self.messages[1:])

        logger = AgenticLogger() if self.ctx.agentic_logging else None
        if logger:
            logger.write(type="start", user_input=final_content)

        openai_tools = []
        for name, defn in AGENTIC_TOOL_DEFS.items():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": defn["description"],
                    "parameters": defn["parameters"]
                }
            })

        return messages, logger, openai_tools, send_tools_api

    @staticmethod
    def _cap_tool_observation(tool_name: str, observation: str, cap: int = 4000) -> str:
        """Truncate a tool observation to fit the context, read_file pages excepted.

        `read_file` already bounds and paginates its own output and its header
        carries the `next` offset, so a blanket cap would silently drop the
        continuation pointer and defeat chunked reads of large files.
        """
        if tool_name == "read_file":
            return observation
        if not observation:
            return observation
        if len(observation) > cap:
            return observation[:cap] + f"\n... [truncated to {cap} chars]"
        return observation

    def _observation_char_cap(self, messages: list) -> int:
        """Progressive observation truncation cap based on context usage.

        Args:
            messages: The conversation list (for context-usage measurement).

        Returns:
            Max observation length in chars (4000/3000/2000 by usage).
        """
        if self.ctx.context_window_size <= 0:
            return 4000
        msg_tokens = self.ctx.calculate_context_tokens(messages)
        usage_ratio = msg_tokens / self.ctx.context_window_size
        if usage_ratio > 0.7:
            return 2000
        if usage_ratio > 0.5:
            return 3000
        return 4000

    def _execute_single_tool(self, tool_name: str, tool_args: dict, iteration: int,
                             logger: Optional['AgenticLogger'], messages: list) -> dict:
        """Run one tool, returning a normalized observation dict.

        Args:
            tool_name: Tool to execute.
            tool_args: Tool arguments.
            iteration: Current iteration.
            logger: Optional AgenticLogger.
            messages: The conversation list (for truncation measurement).

        Returns:
            {"observation": str, "raw": str, "success": bool, "cancelled": bool}.
        """
        args_display = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
        print(colorize(f"\n[Tool] {tool_name}({args_display})", 'warning'), file=sys.stderr, end="")
        sys.stderr.flush()

        if self.ctx.agentic_trace:
            print(colorize(f"\n[Trace] Full args: {json.dumps(tool_args, default=str)[:2000]}", 'muted'), file=sys.stderr)

        t_start = time.time()
        result = self.tool_registry.execute(tool_name, tool_args)
        elapsed = time.time() - t_start
        status = "OK" if result["success"] else "ERROR"
        print(colorize(f" → {status} ({elapsed:.1f}s)", 'info' if result["success"] else 'error'), file=sys.stderr)

        if result["success"]:
            observation = result["output"]
        else:
            full = result["output"] or ""
            lines = full.split("\n")[:4]
            context = "\n".join(lines).strip()
            error_msg = result.get("error", "") or ""
            observation = f"{context}\nERROR: {error_msg}" if context else f"ERROR: {error_msg}"
        if not observation:
            observation = "[Tool returned no output]"

        # read_file pages are exempt — they self-bound and carry a `next`
        # offset that a blanket cap would silently drop.
        max_obs_chars = self._observation_char_cap(messages)
        observation = self._cap_tool_observation(tool_name, observation, max_obs_chars)

        timed_observation = json.dumps({"tool": tool_name, "duration_s": round(elapsed, 1), "success": result["success"], "output": observation})
        if self.ctx.agentic_trace:
            trace_out = result["output"] if len(result["output"]) < 2000 else result["output"][:2000] + "..."
            print(colorize(f"[Trace] Result: {json.dumps({'success': result['success'], 'output': trace_out, 'error': result['error']}, default=str)}", 'muted'), file=sys.stderr)

        if self.ctx.agentic_logging and logger:
            logger.write(type="result", iteration=iteration, tool_name=tool_name, tool_args=tool_args, result=result)

        cancelled = (not result["success"]) and "Cancelled" in (result.get("error") or "")
        return {"observation": f"[{tool_name}] {timed_observation}", "raw": observation,
                "success": result["success"], "cancelled": cancelled}

    def _execute_tool_calls(self, tool_calls: list, last_tool_call: Optional[dict], iteration: int, logger: Optional['AgenticLogger'],
                            messages: list, response_text: str, api_tool_calls: list) -> tuple:
        """Execute a list of tool calls, collecting observations.

        Args:
            tool_calls: Normalized tool call list.
            last_tool_call: Previous tool call (for same-tool-loop detection).
            iteration: Current ReAct iteration.
            logger: Optional AgenticLogger.
            messages: The conversation list.
            response_text: The assistant's text for this step.
            api_tool_calls: Native tool calls from the response.

        Returns:
            (observations, raw_observations, abort_loop, last_tool_call, final_answer).
        """
        observations = []
        raw_observations = []
        abort_loop = False
        final_answer = ""
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call["tool"]
            tool_args = tool_call.get("arguments", {})

            current_call = (tool_name, json.dumps(tool_args, sort_keys=True))
            if i == 0 and current_call == last_tool_call:
                print(colorize("[Agentic] Same tool call repeated, breaking loop.", 'warning'), file=sys.stderr)
                final_answer = "[Agentic: model stuck in tool loop]"
                abort_loop = True
                break
            if i == 0:
                last_tool_call = current_call

            exec = self._execute_single_tool(tool_name, tool_args, iteration, logger, messages)
            if exec["cancelled"]:
                print(colorize("[Agentic] Tool cancelled by user, aborting.", 'warning'), file=sys.stderr)
                final_answer = "[Agentic query cancelled]"
                abort_loop = True
                break

            observations.append(exec["observation"])
            raw_observations.append(exec["raw"])

        return observations, raw_observations, abort_loop, last_tool_call, final_answer

    def _finalize_agentic_query(self, messages: list, final_answer: str, final_content: str,
                                send_tools_api: bool, openai_tools: list,
                                logger: Optional['AgenticLogger'],
                                iteration: int, response_text: str) -> None:
        """Stream final answer or execute pending tool call, then update self.messages."""
        stream_tool_used = False
        if not final_answer:
            final_answer = response_text if response_text else "[Agentic: no answer produced]"

        if logger:
            logger.write(type="final", final_answer=final_answer)

        pending_tool = self.parse_tool_call(final_answer)
        if pending_tool:
            observation = self._finalize_pending_tool(pending_tool)
            print(colorize(f"\n{observation}", 'info'), file=sys.stdout)
            self.messages.append({'role': 'assistant', 'content': final_answer})
            print()
        else:
            if not getattr(self, 'messages', None):
                self.messages = [{'role': 'system', 'content': self.ctx.system_prompt}]
            final_messages = list(messages)
            if final_messages:
                final_messages[0] = {'role': 'system', 'content': self.ctx.system_prompt}
            if final_messages and final_messages[-1]['role'] != 'user':
                final_messages.append({'role': 'user', 'content': final_content})

            stream_kwargs = dict(get_inference_params(self.ctx.model))
            stream_tool_calls_out = []
            if send_tools_api:
                stream_kwargs["tools"] = openai_tools

            response = self.query_handler.query_stream(
                final_messages, self.ctx.model,
                stream_enabled=self.ctx.stream_enabled,
                debug=self.ctx.debug_mode,
                show_thinking=(not self.ctx.force_no_thinking),
                context_size=self.ctx.context_size,
                images=([] if self.ctx.supports_vision is False else self.ctx.current_images),
                tool_calls_out=stream_tool_calls_out,
                no_thinking=self.ctx.force_no_thinking,
                reasoning_effort=self.ctx.reasoning_effort,
                **stream_kwargs
            )
            if response or stream_tool_calls_out:
                stream_tool_calls = self._collect_stream_tool_calls(stream_tool_calls_out, response)
                if stream_tool_calls:
                    stream_tool_used = self._execute_stream_tool_calls(
                        stream_tool_calls, stream_tool_calls_out, response, send_tools_api)
                else:
                    self.messages.append({'role': 'assistant', 'content': response})
                    print()
            else:
                print(colorize(f"\n{final_answer}", 'success'), file=sys.stdout)

        stream_tool_used = self._agentic_streaming_reentry(stream_tool_used, send_tools_api, openai_tools)

    def _finalize_pending_tool(self, pending_tool: dict) -> str:
        """Execute a single pending tool call found in the final answer.

        Returns the (possibly truncated) observation text.
        """
        tool_name = pending_tool["tool"]
        tool_args = pending_tool.get("arguments", {})
        args_display = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
        print(colorize(f"\n[Tool] {tool_name}({args_display})", 'warning'), file=sys.stderr)
        result = self.tool_registry.execute(tool_name, tool_args)
        observation = result["output"] if result["success"] else f"ERROR: {result['error']}"
        if not observation:
            observation = "[Tool returned no output]"
        observation = self._cap_tool_observation(tool_name, observation)
        return observation

    def _collect_stream_tool_calls(self, stream_tool_calls_out: list, response: str) -> list:
        """Normalize tool calls from the final streaming response.

        Args:
            stream_tool_calls_out: Accumulated native tool calls from the stream.
            response: The streamed response text.

        Returns:
            List of {"tool", "arguments"} dicts.

        Prefers native `tool_calls` out-params; falls back to parsing the text
        response for inline JSON tool calls.
        """
        stream_tool_calls = []
        if stream_tool_calls_out:
            for tc in stream_tool_calls_out:
                func = tc.get("function", {})
                name = func.get("name", "")
                args_raw = func.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {"raw": args_raw}
                stream_tool_calls.append({"tool": name, "arguments": args})
        if not stream_tool_calls and response:
            stream_tool_calls = self.parse_tool_calls(response)
            if not stream_tool_calls:
                single = self.parse_tool_call(response)
                if single:
                    stream_tool_calls = [single]
        return stream_tool_calls

    def _append_stream_observations(self, response: str, stream_tool_calls_out: list,
                                    stream_observations: list, send_tools_api: bool) -> None:
        """Record assistant tool_call + tool results into self.messages.

        Args:
            response: The streamed response text.
            stream_tool_calls_out: Native tool calls from the stream.
            stream_observations: Per-tool observation strings.
            send_tools_api: Whether to use OpenAI `tool` role messages.
        """
        if stream_tool_calls_out:
            stream_content = response or ""
            if not stream_content:
                inline_parts = []
                for stc in stream_tool_calls_out:
                    fn = stc.get("function", {})
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                    inline_parts.append(json.dumps({
                        "tool": fn.get("name", ""),
                        "arguments": args
                    }))
                stream_content = "\n".join(inline_parts)
            assistant_msg = {'role': 'assistant', 'content': stream_content}
            assistant_msg['tool_calls'] = stream_tool_calls_out
            self.messages.append(assistant_msg)
        else:
            self.messages.append({'role': 'assistant', 'content': response})
        if stream_observations:
            if send_tools_api and stream_tool_calls_out:
                for idx, tc in enumerate(stream_tool_calls_out):
                    obs = stream_observations[idx] if idx < len(stream_observations) else "ERROR: Tool execution declined by the user."
                    clean_obs = obs.split("] ", 1)[1] if "] " in obs else obs
                    self.messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.get('id', ''),
                        'name': tc.get('function', {}).get('name', ''),
                        'content': clean_obs
                    })
            else:
                self.messages.append({'role': 'user', 'content': "Tool result:\n" + "\n---\n".join(stream_observations)})
        elif send_tools_api and stream_tool_calls_out:
            for tc in stream_tool_calls_out:
                self.messages.append({
                    'role': 'tool',
                    'tool_call_id': tc.get('id', ''),
                    'name': tc.get('function', {}).get('name', ''),
                    'content': "ERROR: Tool execution declined by the user."
                })

    def _execute_stream_tool_calls(self, stream_tool_calls: list, stream_tool_calls_out: list,
                                   response: str, send_tools_api: bool) -> bool:
        """Execute tools produced by the final streaming answer; record self.messages.

        Args:
            stream_tool_calls: Normalized tool call list.
            stream_tool_calls_out: Native tool calls from the stream.
            response: The streamed response text.
            send_tools_api: Whether to use OpenAI `tool` role messages.

        Returns:
            True if any tool ran (so the caller can re-enter the loop).

        Runs each tool, prints results, and appends the assistant tool_call and the
        observations to `self.messages` (OpenAI `tool` role when `send_tools_api`,
        otherwise a `Tool result:` user note).
        """
        stream_observations = []
        for stream_tc in stream_tool_calls:
            tool_name = stream_tc["tool"]
            tool_args = stream_tc.get("arguments", {})
            args_display = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
            print(colorize(f"\n[Tool] {tool_name}({args_display})", 'warning'), file=sys.stderr, end="")
            sys.stderr.flush()
            t_start = time.time()
            result = self.tool_registry.execute(tool_name, tool_args)
            if not result["success"] and "Cancelled" in (result.get("error") or ""):
                print(colorize("[Agentic] Streaming tool sequence cancelled by user, aborting.", 'warning'), file=sys.stderr)
                break
            elapsed = time.time() - t_start
            status = "OK" if result["success"] else "ERROR"
            print(colorize(f" → {status} ({elapsed:.1f}s)", 'info' if result["success"] else 'error'), file=sys.stderr)
            observation = result["output"] if result["success"] else f"ERROR: {result['error']}"
            if not observation:
                observation = "[Tool returned no output]"
            observation = self._cap_tool_observation(tool_name, observation)
            print(colorize(f"\n{observation}", 'info'), file=sys.stdout)
            stream_observations.append(f"[{tool_name}] {observation}")

        self._append_stream_observations(response, stream_tool_calls_out, stream_observations, send_tools_api)
        print()
        return bool(stream_observations)

    def _agentic_streaming_reentry(self, stream_tool_used: bool, send_tools_api: bool, openai_tools: list) -> bool:
        """Re-enter the loop if the final streaming answer itself carried a tool call.

        Args:
            stream_tool_used: Whether the final stream emitted a tool call.
            send_tools_api: Whether native tools are in use.
            openai_tools: Native tool definitions (for the tools API).

        Returns:
            The final `stream_tool_used` value.

        Some models narrate intent in prose then emit the JSON tool call last, so
        the streamed finalize can still carry a tool call. Run up to 3 bounded
        rounds: re-query with updated history, parse tool calls, execute them, feed
        observations back, until the model answers plainly.
        """
        reentry_round = 0
        while stream_tool_used and reentry_round < 3:
            reentry_round += 1
            self._agentic_streaming_reentry_count = getattr(self, '_agentic_streaming_reentry_count', 0) + 1
            tool_format = get_tool_format(self.ctx.model)
            include_tool_defs = tool_format != "openai"
            tool_defs_block = self.tool_registry.get_system_prompt_block() if include_tool_defs else ""
            reentry_messages = [{'role': 'system', 'content': get_agentic_prompt(self.ctx.model, tool_defs_block, include_tool_defs=include_tool_defs)}]
            if len(self.messages) > 1:
                reentry_messages.extend(self.messages[1:])
            sync_kwargs = dict(get_inference_params(self.ctx.model))
            if send_tools_api:
                sync_kwargs["tools"] = openai_tools
            reentry_response = self.query_handler.query_sync(
                reentry_messages, self.ctx.model,
                context_size=self.ctx.context_size,
                images=([] if self.ctx.supports_vision is False else self.ctx.current_images),
                no_thinking=self.ctx.force_no_thinking,
                reasoning_effort=self.ctx.reasoning_effort,
                **sync_kwargs
            )
            reentry_text = _extract_sync_content(self.ctx, reentry_response)
            if not reentry_text:
                break
            self.messages.append({'role': 'assistant', 'content': reentry_text})
            reentry_tools = self.parse_tool_calls(reentry_text)
            if not reentry_tools:
                single = self.parse_tool_call(reentry_text)
                if single:
                    reentry_tools = [single]
            if not reentry_tools:
                print(colorize(f"\n{reentry_text}", 'success'), file=sys.stdout)
                print()
                break
            reentry_observations = []
            for rt in reentry_tools:
                rt_name = rt["tool"]
                rt_args = rt.get("arguments", {})
                args_display = ", ".join(f"{k}={v!r}" for k, v in rt_args.items())
                print(colorize(f"\n[Tool] {rt_name}({args_display})", 'warning'), file=sys.stderr, end="")
                sys.stderr.flush()
                rt_start = time.time()
                rt_result = self.tool_registry.execute(rt_name, rt_args)
                rt_elapsed = time.time() - rt_start
                rt_status = "OK" if rt_result["success"] else "ERROR"
                print(colorize(f" → {rt_status} ({rt_elapsed:.1f}s)", 'info' if rt_result["success"] else 'error'), file=sys.stderr)
                rt_obs = rt_result["output"] if rt_result["success"] else f"ERROR: {rt_result['error']}"
                if not rt_obs:
                    rt_obs = "[Tool returned no output]"
                rt_obs = self._cap_tool_observation(rt_name, rt_obs)
                print(colorize(f"\n{rt_obs}", 'info'), file=sys.stdout)
                reentry_observations.append(f"[{rt_name}] {rt_obs}")
            if reentry_observations:
                self.messages.append({'role': 'user', 'content': "Tool result:\n" + "\n---\n".join(reentry_observations)})
                stream_tool_used = True
            else:
                stream_tool_used = False
        return stream_tool_used

    def _maybe_auto_compact_agentic(self, messages: list) -> list:
        """Auto-compact the ReAct loop's local messages near the context limit.

        Agentic steps can balloon the local `messages` array (tool observations,
        re-queries). When usage passes `ctx.agentic_compact_threshold` (default
        85%, later than the regular `ctx.compaction_threshold`), mechanically
        compact with a wider keep-recent window so the loop keeps working memory.

        Returns the (possibly compacted) message list.
        """
        if self.ctx.context_window_size <= 0:
            return messages
        msg_tokens = self.ctx.calculate_context_tokens(messages)
        if msg_tokens <= int(self.ctx.context_window_size * self.ctx.agentic_compact_threshold):
            return messages
        before_len = len(messages)
        messages = compact_messages(messages, self.ctx,
                                    keep_recent=max(8, COMPACTION_KEEP_RECENT))
        print(colorize(f"\n[Auto-compact] Agentic context: {before_len} → {len(messages)} msgs",
                       'warning'), file=sys.stderr)
        return messages

    def _persist_agentic_history(self, messages: list, agentic_seed_len: int, compact: bool = False) -> None:
        """Merge the ReAct loop's local turns back into `self.messages`.

        The local `messages` array is seeded as [agentic system] + self.messages[1:],
        so the genuinely new turns begin at index `agentic_seed_len`. They are
        inserted at that boundary — right after the current user query and BEFORE
        any messages the streaming finalize appended — so the conversation stays
        chronologically ordered (user query → tool work → final answer).

        `_system_nudge` messages (loop-internal steering, e.g. the empty-response
        "Please provide a tool call..." fallback) are excluded from persistent
        history so repeated empty responses don't pile into context forever.

        Also called from the `except KeyboardInterrupt` path so a ^C that aborts a
        step still preserves every completed tool turn instead of erasing the
        conversation (the interrupted step's partial output is never in `messages`
        yet, so it is correctly not persisted).

        Args:
            messages: The local ReAct message list.
            agentic_seed_len: Index in `self.messages` where new turns begin.
            compact: Whether to auto-compact the merged history if over threshold.
        """
        if len(messages) > agentic_seed_len:
            new_turns = [m for m in messages[agentic_seed_len:] if not m.get('_system_nudge')]
            if new_turns:
                self.messages[agentic_seed_len:agentic_seed_len] = new_turns
        sanitize_tool_pairing(self.messages)

        # Persist compaction savings: the ReAct loop may have auto-compacted its
        # local `messages`, but that compaction is discarded by the merge above.
        # Re-check the merged history and compact if it exceeds the threshold.
        if compact and self.ctx.context_window_size > 0:
            msg_tokens = self.ctx.calculate_context_tokens(self.messages)
            threshold = int(self.ctx.context_window_size * self.ctx.compaction_threshold)
            if msg_tokens > threshold:
                before_len = len(self.messages)
                self.messages = compact_messages(self.messages, self.ctx)
                print(colorize(
                    f"[Auto-compact] Agentic history persisted: {before_len} → {len(self.messages)} msgs",
                    'warning'), file=sys.stderr)

        # Recalculate context tokens after merging agentic history
        self.ctx.current_context_tokens = self.ctx.calculate_context_tokens(self.messages)

    def _parse_agentic_tool_calls(self, response_text: str, api_tool_calls: list) -> list:
        """Build normalized tool calls from either native API calls or inline JSON text.

        Args:
            response_text: Assistant text content (may embed JSON tool calls).
            api_tool_calls: Native `tool_calls` from the response (already
                backend-normalized), or an empty list.

        Returns:
            list of {"tool": name, "arguments": args} dicts (may be empty).
        """
        tool_calls = []
        if api_tool_calls:
            for tc in api_tool_calls:
                func = tc.get('function', {})
                name = func.get('name', '')
                args_raw = func.get('arguments', {})
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = args_raw
                tool_calls.append({"tool": name, "arguments": args})
        elif response_text:
            tool_calls = self.parse_tool_calls(response_text)
            if not tool_calls:
                single = self.parse_tool_call(response_text)
                if single:
                    tool_calls = [single]
        return tool_calls

    def _append_tool_messages(self, messages: list, response_text: str, api_tool_calls: list,
                              tool_calls: list, observations: list, raw_observations: list,
                              send_tools_api: bool) -> None:
        """Append the assistant turn and tool-result messages after executing tools.

        Native-tools backends get `tool`-role messages keyed by `tool_call_id`;
        inline-mode backends get a plain `Tool result:` user note.
        """
        combined = "\n---\n".join(observations) if observations else "[No tool output]"
        assistant_content = response_text
        if api_tool_calls and not response_text and tool_calls:
            tool_json = json.dumps(tool_calls[0])
            assistant_content = tool_json
        assistant_msg = {'role': 'assistant', 'content': assistant_content}
        if api_tool_calls:
            assistant_msg['tool_calls'] = api_tool_calls
        messages.append(assistant_msg)
        if send_tools_api and api_tool_calls and observations:
            for idx, tc in enumerate(api_tool_calls):
                obs = raw_observations[idx] if idx < len(raw_observations) else "ERROR: Cancelled or skipped due to preceding tool sequence abort."
                tool_msg = {'role': 'tool', 'tool_call_id': tc.get('id', ''), 'content': obs}
                tool_msg['name'] = tc.get('function', {}).get('name', '')
                messages.append(tool_msg)
        else:
            messages.append({'role': 'user', 'content': f"Tool result:\n{combined}"})

    def _agentic_step_header(self, iteration: int, max_iterations: int, logger: Optional['AgenticLogger'], messages: list) -> list:
        """Print the ReAct step header and auto-compact messages if needed.

        Args:
            iteration: Current loop iteration (1-based).
            max_iterations: Loop ceiling.
            logger: Optional AgenticLogger.
            messages: The conversation list (returned compacted).

        Returns:
            The (possibly compacted) message list.
        """
        if logger:
            logger.write(type="iteration", iteration=iteration)
        print(colorize(f"\r[Agentic] Step {iteration}/{max_iterations}…", 'muted'), file=sys.stderr, end="")
        sys.stderr.flush()

        messages = self._maybe_auto_compact_agentic(messages)

        # Verbose: show payload token count
        if self.ctx.agentic_verbose and self.ctx.context_window_size > 0:
            payload_tokens = self.ctx.calculate_context_tokens(messages)
            pct = payload_tokens / self.ctx.context_window_size
            print(colorize(f" [{payload_tokens} tokens, {pct:.0%} of ctx]", 'muted'), file=sys.stderr, end="")
        return messages

    def _agentic_call_step(self, messages: list, step_timeout: int, images_to_send: list,
                           sync_kwargs: dict, on_chunk: object, finalize_step: object, step_cancel: dict) -> tuple:
        """Run one ReAct model call under a wall-clock timeout.

        Args:
            messages: The conversation list to send.
            step_timeout: Timeout in seconds.
            images_to_send: Images for the request (empty when vision unsupported).
            sync_kwargs: Inference params / tools kwargs for the call.
            on_chunk: Feedback callback from _make_agentic_step_feedback.
            finalize_step: Closes feedback output on completion/interrupt.
            step_cancel: Cancel token dict for aborting the request.

        Returns:
            (response, step_start) where response is the sync-shaped dict, or
            None on timeout. Raises KeyboardInterrupt after aborting the
            in-flight request (so the [Interrupted] handler persists turns).
        """
        step_start = time.time()
        try:
            response = self._call_with_timeout(
                self.query_handler.query_sync_stream, step_timeout,
                messages, self.ctx.model,
                context_size=self.ctx.context_size,
                images=images_to_send,
                on_chunk=on_chunk,
                timeout=step_timeout + 30,
                cancel=step_cancel,
                **sync_kwargs
            )
        except KeyboardInterrupt:
            # F1: stop the zombie generation from streaming to the terminal AND
            # abort the request, then let the [Interrupted] handler take over
            # (F2 persists completed turns).
            _signal_abort(step_cancel)
            finalize_step()
            raise
        finalize_step()
        return response, step_start

    def _agentic_timeout_continue(self, step_buf: dict, step_cancel: dict, step_start: float,
                                  iteration: int, step_timeout: int, messages: list) -> tuple:
        """Handle a timed-out step: abort the request and escalate the budget.

        Args:
            step_buf: The step feedback buffer (accumulated thought/content).
            step_cancel: Cancel token dict for the in-flight request.
            step_start: Wall-clock timestamp when the step began.
            iteration: Current iteration.
            step_timeout: Timeout that just expired.
            messages: The conversation list (may receive a nudge message).

        Returns:
            (should_break, new_step_timeout).
        """
        # Abort the still-running request so the daemon thread dies promptly
        # and the server slot frees for a retry.
        _signal_abort(step_cancel)
        # F3: pass generation timing/rate so the timeout policy can distinguish
        # "still generating" (extend budget) from "stalled" (conservative
        # retry / abort), and so the nudge can report the model its speed and
        # remaining token budget.
        elapsed = time.time() - step_start
        partial_tokens = self.ctx.estimate_tokens(
            (step_buf.get("thought", "") + " " + step_buf.get("content", "")).strip())
        last_activity = step_buf.get("last_activity")
        idle_sec = (time.time() - last_activity) if last_activity else None
        return self._handle_agentic_timeout(
            iteration, step_timeout, messages,
            partial_text=step_buf.get("content", ""),
            partial_thought=step_buf.get("thought", ""),
            elapsed_sec=elapsed, partial_tokens=partial_tokens, idle_sec=idle_sec,
            thinking_capped=step_buf.get("thinking_capped", False))

    def _agentic_handle_response_flags(self, response: object, images_to_send: list,
                                       send_tools_api: bool, messages: list) -> tuple:
        """Track tokens and handle API/vision/tools errors for a step response.

        Args:
            response: The sync-shaped response dict.
            images_to_send: Images included in the request.
            send_tools_api: Whether native tools are currently enabled.
            messages: The conversation list (mutated on tools fallback).

        Returns:
            (status, api_error, send_tools_api) where status is one of
            "ok" / "api_error" / "vision" / "tools".
        """
        # Track tokens from sync response to keep context bar accurate
        if isinstance(response, dict):
            if self.ctx.backend == "ollama":
                _pt = response.get("prompt_eval_count", 0)
                _et = response.get("eval_count", 0)
                if _pt > 0:
                    self.ctx.current_context_tokens = _pt + _et
            else:
                # llama.cpp KV cache makes prompt_tokens unreliable;
                # recalculate from actual messages for accuracy
                self.ctx.current_context_tokens = self.ctx.calculate_context_tokens(messages)

        # Check for API-level errors from query_sync
        if isinstance(response, dict) and "error" in response and not response.get("choices") and not response.get("message", {}).get("content"):
            err_msg = response["error"].get("message", "Unknown error") if isinstance(response["error"], dict) else str(response["error"])
            print(colorize(f"\n[Agentic] API error: {err_msg}", 'error'), file=sys.stderr)
            return "api_error", err_msg, send_tools_api

        if images_to_send and self.ctx.supports_vision is not False:
            if check_vision_error(response):
                self.ctx.supports_vision = False
                print(colorize("\n[WARNING] Model does not support vision. Stripping images for subsequent queries.", 'warning'), file=sys.stderr)
                return "vision", "", send_tools_api

        if send_tools_api and check_tools_error(response):
            send_tools_api = False
            tool_defs_block = self.tool_registry.get_system_prompt_block()
            messages[0] = {'role': 'system', 'content': get_agentic_prompt(self.ctx.model, tool_defs_block, include_tool_defs=True)}
            print(colorize("\n[WARNING] Model does not support native tools API. Falling back to inline tool definitions.", 'warning'), file=sys.stderr)
            return "tools", "", send_tools_api
        return "ok", "", send_tools_api

    def _agentic_step_content(self, response: object) -> tuple:
        """Extract response text + native tool calls from a step response.

        Args:
            response: The sync-shaped response dict.

        Returns:
            (response_text, api_tool_calls) tuple (verbose display printed here).
        """
        response_text = _extract_sync_content(self.ctx, response)
        api_tool_calls = _extract_sync_tool_calls(self.ctx, response)

        if self.ctx.agentic_verbose and response_text:
            truncated = len(response_text) > 500
            display = response_text[:500] + ("..." if truncated else "")
            print(colorize(f"\n[Verbose] {display}", 'muted'), file=sys.stderr)
            if truncated:
                print(colorize(f"[Verbose] ({len(response_text)} total chars, showing first 500)", 'muted'), file=sys.stderr)
        return response_text, api_tool_calls

    def _agentic_finish_query(self, messages: list, final_answer: str, final_content: str,
                              send_tools_api: bool, openai_tools: list, logger: Optional['AgenticLogger'],
                              iteration: int, max_iterations: int, response_text: str,
                              api_error: str, agentic_seed_len: int) -> None:
        """Post-loop finalize: print the API-error banner or stream the final answer.

        Args:
            messages: The loop's message list.
            final_answer: The answer produced by the loop (may be "").
            final_content: The user's processed query text.
            send_tools_api: Whether native tools are in use.
            openai_tools: Native tool definitions.
            logger: Optional AgenticLogger.
            iteration: Final iteration count.
            max_iterations: Loop ceiling.
            response_text: The last response text.
            api_error: Set when the loop aborted on an API error.
            agentic_seed_len: Seed boundary for history persistence.
        """
        if api_error:
            # Backend unreachable (connection refused, 5xx, ...). Do NOT re-query
            # a dead endpoint for a final answer — the finalize path would launch
            # a second doomed streaming request (another 3 retries) and print a
            # misleading "[Agentic: no answer produced]". Nothing is merged into
            # the persistent history beyond the user's own message, so repeated
            # attempts while the server is down don't pollute the context.
            print(colorize(
                f"\n[Agentic] Backend unreachable ({api_error}). "
                "Aborted without a final answer; conversation history untouched.",
                'warning'), file=sys.stderr)
        else:
            if iteration >= max_iterations and not final_answer:
                final_answer = response_text

            self._finalize_agentic_query(messages, final_answer, final_content, send_tools_api, openai_tools, logger, iteration, response_text)

        # Merge the ReAct loop turns back into self.messages for cross-turn
        # memory (see `_persist_agentic_history` for ordering/nudge handling).
        self._persist_agentic_history(messages, agentic_seed_len, compact=True)

        if logger:
            logger.write(type="end", total_iterations=iteration)

    def run_agentic_query(self, full_input: str) -> None:
        """ReAct loop: query model, parse tool calls, execute tools, stream final answer."""
        self._agentic_streaming_reentry_count = 0
        logger = None
        try:
            final_content = process_inline_commands(full_input)
            if not final_content.strip():
                return

            if not getattr(self, 'messages', None):
                sys_msg = {'role': 'system', 'content': self.ctx.system_prompt}
                self.ctx.stamp_tokens(sys_msg)
                self.messages = [sys_msg]
            user_msg = {'role': 'user', 'content': final_content}
            self.ctx.stamp_tokens(user_msg)
            self.messages.append(user_msg)
            agentic_seed_len = len(self.messages)
            sanitize_tool_pairing(self.messages)

            inited = self._init_agentic_query(final_content)
            if inited is None:
                return
            messages, logger, openai_tools, send_tools_api = inited

            iteration = 0
            max_iterations = self.ctx.agentic_max_iterations
            final_answer = ""
            last_tool_call = None
            step_timeout = self.ctx.agentic_step_timeout
            response_text = ""
            api_error = ""

            # Reset timeout escalation state at the start of each agentic query.
            self.ctx.agentic_consecutive_timeouts = 0
            self.ctx.agentic_has_executed_tool = False
            self.ctx.agentic_last_tool_name = ""

            while iteration < max_iterations:
                iteration += 1
                messages = self._agentic_step_header(iteration, max_iterations, logger, messages)

                images_to_send = [] if self.ctx.supports_vision is False else self.ctx.current_images
                sync_kwargs = dict(get_inference_params(self.ctx.model))
                if send_tools_api:
                    sync_kwargs["tools"] = openai_tools
                if self.ctx.force_no_thinking:
                    sync_kwargs["no_thinking"] = True
                if self.ctx.reasoning_effort:
                    sync_kwargs["reasoning_effort"] = self.ctx.reasoning_effort
                on_chunk, finalize_step, step_buf = self._make_agentic_step_feedback()
                step_cancel = {"event": threading.Event(), "close": None}

                response, step_start = self._agentic_call_step(
                    messages, step_timeout, images_to_send, sync_kwargs,
                    on_chunk, finalize_step, step_cancel)

                if response is None:
                    # Timed out — escalate (extend while generating, else retry/abort).
                    should_break, step_timeout = self._agentic_timeout_continue(
                        step_buf, step_cancel, step_start, iteration, step_timeout, messages)
                    if should_break:
                        break
                    response_text = ""
                    continue

                status, api_error, send_tools_api = self._agentic_handle_response_flags(
                    response, images_to_send, send_tools_api, messages)
                if status == "api_error":
                    break
                if status in ("vision", "tools"):
                    continue

                response_text, api_tool_calls = self._agentic_step_content(response)

                if not response_text and not api_tool_calls:
                    messages.append({'role': 'user', 'content': 'Please provide a tool call or your final answer.',
                                     '_system_nudge': True})
                    continue

                if response_text and self._is_stuck(response_text):
                    print(colorize("\n[Agentic] Model appears stuck (repetitive output), aborting.", 'warning'), file=sys.stderr)
                    break

                tool_calls = self._parse_agentic_tool_calls(response_text, api_tool_calls)

                if logger:
                    first_call = tool_calls[0] if tool_calls else None
                    logger.write(type="turn", iteration=iteration, model_response=response_text, tool_call=first_call)

                if not tool_calls:
                    final_answer = response_text
                    self.ctx.agentic_consecutive_timeouts = 0
                    break

                observations, raw_observations, abort_loop, last_tool_call, tool_final = self._execute_tool_calls(
                    tool_calls, last_tool_call, iteration, logger, messages, response_text, api_tool_calls
                )
                if tool_calls:
                    self.ctx.agentic_has_executed_tool = True
                    self.ctx.agentic_last_tool_name = tool_calls[0]["tool"]
                    self.ctx.agentic_consecutive_timeouts = 0
                if tool_final:
                    final_answer = tool_final
                if abort_loop:
                    break

                self._append_tool_messages(
                    messages, response_text, api_tool_calls, tool_calls,
                    observations, raw_observations, send_tools_api)

            self._agentic_finish_query(
                messages, final_answer, final_content, send_tools_api, openai_tools,
                logger, iteration, max_iterations, response_text, api_error, agentic_seed_len)

        except KeyboardInterrupt:
            # F2: a ^C aborts the current step but must NOT erase the completed
            # tool work. Persist the finished turns before handing control back;
            # ChatLoop.run prints the [Interrupted] banner and resumes the prompt.
            if 'messages' in locals() and 'agentic_seed_len' in locals():
                self._persist_agentic_history(messages, agentic_seed_len)
            raise
        except Exception as e:
            print(colorize(f"\n[Agentic] Internal error: {e}", 'error'), file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            if logger:
                logger.close()

    def run_handle_spawnshell(self, full_input: str) -> Optional[bool]:
        """Handle /spawnshell command. Captures shell session and lets user choose what to send."""
        if full_input == '/spawnshell':
            session_output = self.handle_spawnshell()
            if session_output:
                self._handle_shell_session(session_output)
            return False
        return None

    def run_handle_compact(self, full_input: str) -> Optional[bool]:
        """Handle /compact command and subcommands (like /agentic)."""
        if not full_input.startswith('/compact'):
            return None
        parts = full_input.split()
        subcmd = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            return self._compact_list()
        if subcmd == "threshold":
            return self._compact_threshold(parts)
        if subcmd == "now":
            return self._compact_run()
        if subcmd == "force":
            if len(parts) > 2:
                return self._compact_force_message(parts[2])
            return self._compact_run(force=True)
        if subcmd == "llm":
            return self._compact_run(use_llm=True)
        if subcmd in ("", "status"):
            return self._print_compact_status()

        print(colorize("[Usage: /compact [now|force [index]|llm|threshold <v>|list|status]]", 'warning'), file=sys.stderr)
        return False

    def _compact_run(self, force: bool = False, use_llm: bool = False) -> bool:
        """Perform compaction. Returns False to keep the loop running."""
        if not hasattr(self, 'messages') or len(self.messages) <= 2:
            print(colorize("[Compact] Nothing to compact.", 'warning'), file=sys.stderr)
            return False
        before_len = len(self.messages)
        before_tokens = self.ctx.calculate_context_tokens(self.messages)

        # Repair any orphaned tool messages before summarizing/compacting so
        # strict chat templates (gpt-oss) don't reject the resulting history.
        sanitize_tool_pairing(self.messages)

        # Optional: LLM-summarize large tool results before mechanical compaction.
        if use_llm:
            summarize_tool_results(self.messages, self.ctx, self.query_handler)
            summarized = len([m for m in self.messages
                              if isinstance(m.get('content', ''), str)
                              and m.get('content', '').startswith('[Summarized tool result')])
            if summarized:
                print(colorize(
                    f"[Compact] LLM-summarized {summarized} tool result(s).",
                    'info'), file=sys.stderr)
            else:
                print(colorize(
                    "[Compact] No eligible tool results to summarize.",
                    'muted'), file=sys.stderr)

        # Phase 1: LLM narrative summary (when requested) or sliding-window compaction
        if use_llm:
            self.messages = llm_compact_messages(self.messages, self.ctx, self.query_handler)
            llm_summary_used = (
                len(self.messages) > 1
                and isinstance(self.messages[1].get('content', ''), str)
                and self.messages[1]['content'].startswith(LLM_COMPACT_SUMMARY_MARKER)
            )
            if llm_summary_used:
                print(colorize(
                    "[Compact] LLM-summarized the older conversation.",
                    'info'), file=sys.stderr)
            else:
                print(colorize(
                    "[Compact] LLM summarization failed; fell back to mechanical compaction.",
                    'muted'), file=sys.stderr)
        else:
            self.messages = compact_messages(self.messages, self.ctx, force=force)
        after_tokens = self.ctx.calculate_context_tokens(self.messages)

        # Phase 2: if still over the target budget (COMPACTION_TARGET) and
        # there's a giant message, truncate it
        if self.ctx.context_window_size > 0:
            target = int(self.ctx.context_window_size * COMPACTION_TARGET)
            if after_tokens > target and len(self.messages) > 1:
                after_tokens = self._compact_truncate_largest(target)

        self.ctx.current_context_tokens = after_tokens
        print(colorize(
            f"[Compact] {before_len} → {len(self.messages)} messages, "
            f"{before_tokens} → {after_tokens} tokens "
            f"(saved {before_tokens - after_tokens} tokens)",
            'success'), file=sys.stderr)
        return False

    def _compact_truncate_largest(self, target: int) -> int:
        """Phase 2: truncate the largest single message to fit the token budget.

        If the total still exceeds `target` and one message is large enough to be
        worth shrinking, cut roughly `excess * 4` characters from its tail.

        Returns the updated total token count (unchanged if nothing was truncated).
        """
        total = self.ctx.calculate_context_tokens(self.messages)
        if total <= target or len(self.messages) <= 1:
            return total
        largest_idx = -1
        largest_tokens = 0
        for i, msg in enumerate(self.messages):
            if i == 0:
                continue
            t = msg.get('_tokens', self.ctx.estimate_tokens(msg.get('content', '')))
            if t > largest_tokens:
                largest_tokens = t
                largest_idx = i
        if largest_idx <= 0 or largest_tokens <= target * 0.3:
            return total
        msg = self.messages[largest_idx]
        content = msg.get('content', '')
        if not isinstance(content, str):
            return total
        excess = total - target
        chars_to_cut = int(excess * 4)  # rough tokens→chars
        if chars_to_cut <= 0 or chars_to_cut >= len(content):
            return total
        role = msg.get('role', 'unknown')
        msg['content'] = content[:len(content) - chars_to_cut] + \
            f"\n\n[... truncated {chars_to_cut} chars to fit context budget ...]"
        msg.pop('_tokens', None)
        self.ctx.stamp_tokens(msg)
        updated = self.ctx.calculate_context_tokens(self.messages)
        print(colorize(
            f"[Compact] Truncated large {role} message (index {largest_idx}) to fit budget.",
            'warning'), file=sys.stderr)
        return updated

    def _compact_list(self) -> bool:
        """List all messages with sizes (like /tokencount)."""
        if not hasattr(self, 'messages') or not self.messages:
            print(colorize("[Compact] No messages in session.", 'warning'), file=sys.stderr)
            return False
        total = 0
        print(colorize("\n--- Conversation Messages ---", 'info'), file=sys.stderr)
        for i, msg in enumerate(self.messages):
            role = msg.get('role', 'unknown')
            name = msg.get('name', '')
            content = msg.get('content', '')
            if '_tokens' in msg:
                tokens = msg['_tokens']
                marker = "·"
            else:
                tokens = self.ctx.estimate_tokens(content) + 2
                marker = "~"
            total += tokens
            preview = content[:60].replace('\n', ' ') if isinstance(content, str) else str(content)[:60]
            if len(content) > 60:
                preview += '...'
            tag = f" ({name})" if name else ""
            print(colorize(f"  [{i:3d}] {role:10s}{tag:<14s}{marker}{tokens:6d} tok  {preview}",
                           'info'), file=sys.stderr)
        print(colorize(f"\n  Total: {total} tokens", 'info'), file=sys.stderr)
        if self.ctx.context_window_size > 0:
            pct = total / self.ctx.context_window_size
            print(colorize(f"  Context: {total}/{self.ctx.context_window_size} ({pct:.1%})", 'info'), file=sys.stderr)
        print(colorize("--- End ---\n", 'info'), file=sys.stderr)
        return False

    def _print_compact_status(self) -> bool:
        """Display current compaction settings like /agentic status."""
        c = self.ctx
        print(colorize("\n[Compaction Settings - Use /compact <option> [value]]", 'info'), file=sys.stderr)
        print("  Subcommands: now, force [index], llm, threshold <v>, list, status", file=sys.stderr)
        print(file=sys.stderr)

        settings = [
            ("threshold", "Auto-compact trigger", f"{c.compaction_threshold:.0%}"),
            ("target",    "Compact down to",      f"{COMPACTION_TARGET:.0%}"),
            ("keep",      "Recent messages kept", str(COMPACTION_KEEP_RECENT)),
        ]
        for name, desc, value in settings:
            print(f"  > {name:<12} [{value:<8}] {desc}", file=sys.stderr)

        total = c.calculate_context_tokens(getattr(self, 'messages', [])) if hasattr(self, 'messages') else 0
        if c.context_window_size > 0:
            pct = total / c.context_window_size
            threshold_tok = int(c.context_window_size * c.compaction_threshold)
            print(f"\n  Window size:       {c.context_window_size}", file=sys.stderr)
            print(f"  Current usage:     {total}/{c.context_window_size} ({pct:.1%})", file=sys.stderr)
            print(f"  Auto-compact at:   {threshold_tok} tokens", file=sys.stderr)
        else:
            print(f"\n  Current usage:     {total} tokens", file=sys.stderr)
            print("  Window size:       unknown (auto-compact disabled)", file=sys.stderr)
        print(file=sys.stderr)
        return False

    def _compact_threshold(self, parts: list) -> bool:
        """Set the auto-compaction trigger threshold.

        Args:
            parts: Split /compact args (parts[2] is the threshold value).
        """
        if len(parts) < 3:
            print(colorize("[Usage: /compact threshold <value>]  (e.g. 0.6 or 60)", 'warning'), file=sys.stderr)
            return False
        try:
            val = float(parts[2])
        except ValueError:
            print(colorize(f"[Compact] '{parts[2]}' is not a number.", 'error'), file=sys.stderr)
            return False
        if val > 1:
            val = val / 100.0
        if not (0.0 < val < 1.0):
            print(colorize("[Compact] Threshold must be between 0 and 1 (or 1-99 as percent).", 'error'), file=sys.stderr)
            return False
        self.ctx.compaction_threshold = val
        print(colorize(f"[Compact] Auto-compaction threshold set to {val:.0%}.", 'success'), file=sys.stderr)
        return False

    def _compact_force_message(self, index_str: str) -> bool:
        """Force-summarize (tool result) or truncate a single message by index."""
        try:
            idx = int(index_str)
        except ValueError:
            print(colorize(f"[Compact] '{index_str}' is not a valid index.", 'error'), file=sys.stderr)
            return False
        if not hasattr(self, 'messages') or not (0 < idx < len(self.messages)):
            print(colorize(f"[Compact] Index {idx} out of range (1-{len(self.messages) - 1}).", 'error'), file=sys.stderr)
            return False
        msg = self.messages[idx]
        content = msg.get('content', '')
        if not isinstance(content, str) or len(content) <= SUMMARIZE_TOOL_MIN_CHARS:
            print(colorize(f"[Compact] Message {idx} is already small ({len(content)} chars).", 'muted'), file=sys.stderr)
            return False
        before = len(content)
        role = msg.get('role', 'unknown')
        name = msg.get('name', '')
        if role == 'tool' and name and name not in SUMMARIZE_TOOL_EXCLUDE:
            summary = _llm_summarize_tool_result(self.ctx, self.query_handler, name, content)
            if summary:
                msg['content'] = f"[Summarized tool result ({name})]\n{summary}"
                msg.pop('_tokens', None)
                self.ctx.stamp_tokens(msg)
                print(colorize(
                    f"[Compact] LLM-summarized message {idx} ({before} → {len(msg['content'])} chars).",
                    'success'), file=sys.stderr)
                return False
            print(colorize(f"[Compact] LLM summarization failed for message {idx}; truncating instead.", 'warning'), file=sys.stderr)
        max_chars = 2000
        msg['content'] = content[:max_chars] + f"\n\n[... truncated {before - max_chars} chars by /compact force ...]"
        msg.pop('_tokens', None)
        self.ctx.stamp_tokens(msg)
        print(colorize(
            f"[Compact] Truncated message {idx} ({role}) from {before} → {len(msg['content'])} chars.",
            'success'), file=sys.stderr)
        return False

    def run_handle_drop(self, full_input: str) -> Optional[bool]:
        """Handle /drop command to remove specific messages by index."""
        if full_input.startswith('/drop'):
            parts = full_input.split()
            if len(parts) < 2:
                print(colorize("[Drop] Usage: /drop <index> [index2 ...] — use /tokencount to see indices", 'warning'), file=sys.stderr)
                return False
            if not hasattr(self, 'messages') or not self.messages:
                print(colorize("[Drop] No messages in session.", 'warning'), file=sys.stderr)
                return False

            indices = []
            for p in parts[1:]:
                try:
                    idx = int(p)
                    if idx == 0:
                        print(colorize("[Drop] Cannot drop system prompt (index 0).", 'error'), file=sys.stderr)
                        return False
                    if 0 < idx < len(self.messages):
                        indices.append(idx)
                    else:
                        print(colorize(f"[Drop] Index {idx} out of range (1-{len(self.messages)-1}).", 'error'), file=sys.stderr)
                        return False
                except ValueError:
                    print(colorize(f"[Drop] '{p}' is not a valid index.", 'error'), file=sys.stderr)
                    return False

            indices.sort(reverse=True)
            dropped_tokens = 0
            for idx in indices:
                msg = self.messages[idx]
                tokens = msg.get('_tokens', self.ctx.estimate_tokens(msg.get('content', '')) + 2)
                dropped_tokens += tokens
                del self.messages[idx]

            self.ctx.current_context_tokens = self.ctx.calculate_context_tokens(self.messages)
            print(colorize(
                f"[Drop] Removed {len(indices)} message(s), freed ~{dropped_tokens} tokens. "
                f"Now: {len(self.messages)} msgs, {self.ctx.current_context_tokens} tokens.",
                'success'), file=sys.stderr)
            return False
        return None

    def run_handle_tokencount(self, full_input: str) -> Optional[bool]:
        """Handle /tokencount command to show token breakdown.

        ` /tokencount <index>` additionally unfolds the full content of that
        message below the summary table.
        """
        if full_input.startswith('/tokencount'):
            if not hasattr(self, 'messages') or not self.messages:
                print(colorize("[TokenCount] No messages in session.", 'warning'), file=sys.stderr)
                return False
            parts = full_input.split()
            unfold_index = None
            if len(parts) > 1:
                try:
                    unfold_index = int(parts[1])
                except ValueError:
                    unfold_index = None
                if unfold_index is not None and not (0 <= unfold_index < len(self.messages)):
                    print(colorize(
                        f"[TokenCount] Index {unfold_index} out of range (0-{len(self.messages) - 1}).",
                        'error'), file=sys.stderr)
                    return False
            total = 0
            exact_count = 0
            print(colorize("\n--- Token Breakdown ---", 'info'), file=sys.stderr)
            for i, msg in enumerate(self.messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if '_tokens' in msg:
                    tokens = msg['_tokens']
                    marker = "·"
                    exact_count += 1
                else:
                    tokens = self.ctx.estimate_tokens(content) + 2
                    marker = "~"
                total += tokens
                preview = content[:60].replace('\n', ' ') if isinstance(content, str) else str(content)[:60]
                if len(content) > 60:
                    preview += '...'
                if role == 'tool' and isinstance(content, str):
                    ok = 'OK' if 'ERROR' not in content[:100] else 'KO'
                    preview = f"[{ok}] {preview}"
                print(colorize(f"  [{i:3d}] {role:10s} {marker}{tokens:6d} tok  {preview}", 'info'), file=sys.stderr)
            print(colorize(f"\n  Total: {total} tokens ({exact_count}/{len(self.messages)} exact, · = exact, ~ = estimated)", 'info'), file=sys.stderr)
            if self.ctx.context_window_size > 0:
                pct = total / self.ctx.context_window_size
                print(colorize(f"  Context: {total}/{self.ctx.context_window_size} ({pct:.1%})", 'info'), file=sys.stderr)
            if unfold_index is not None:
                self._print_full_message(unfold_index)
            print(colorize("--- End Token Breakdown ---\n", 'info'), file=sys.stderr)
            return False
        return None

    def _print_full_message(self, index: int) -> None:
        """Print the complete content of message `index` to stderr."""
        msg = self.messages[index]
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        tokens = msg.get('_tokens') or self.ctx.estimate_tokens(content) + 2
        print(colorize(f"\n--- Full Message [{index}] ({role}, {tokens} tok) ---", 'info'), file=sys.stderr)
        if isinstance(content, str):
            print(content, file=sys.stderr)
        else:
            print(json.dumps(content, indent=2, ensure_ascii=False), file=sys.stderr)
        print(colorize(f"--- End Message [{index}] ---", 'info'), file=sys.stderr)

    def run_handle_sessions(self, full_input: str) -> Optional[bool]:
        """Handle /sessions command to list saved sessions."""
        if full_input.startswith('/sessions'):
            parts = full_input.split()
            limit = 10
            if len(parts) > 1:
                try:
                    limit = int(parts[1])
                except ValueError:
                    pass
            sm = SessionManager()
            sessions = sm.list_sessions(limit=limit)
            if not sessions:
                print(colorize("[Sessions] No saved sessions found.", 'warning'), file=sys.stderr)
                return False
            print(colorize(f"\n--- Saved Sessions (last {len(sessions)}) ---", 'info'), file=sys.stderr)
            for s in sessions:
                model = s.get('model', 'unknown')
                msgs = s.get('message_count', 0)
                summary = s.get('summary', '')[:60]
                created = s.get('created', '')[:16]
                print(colorize(f"  {s['id']}  {model:20s}  {msgs:3d} msgs  {created}  {summary}", 'info'), file=sys.stderr)
            print(colorize("--- End Sessions ---\n", 'info'), file=sys.stderr)
            return False
        return None

    def run_handle_resume(self, full_input: str) -> Optional[bool]:
        """Handle /resume command to resume a previous session."""
        if full_input.startswith('/resume'):
            parts = full_input.split()
            if len(parts) < 2:
                print(colorize("[Resume] Usage: /resume <session_id>", 'warning'), file=sys.stderr)
                return False
            session_id = parts[1]
            sm = SessionManager()
            session_data = sm.load_session(session_id)
            if not session_data:
                print(colorize(f"[Resume] Session '{session_id}' not found.", 'error'), file=sys.stderr)
                return False
            self.messages = session_data.get('messages', [])
            msg_count = len(self.messages)
            self.ctx.current_context_tokens = self.ctx.calculate_context_tokens(self.messages)
            model = session_data.get('model', '')
            print(colorize(
                f"[Resume] Loaded session {session_data['id']} ({msg_count} messages, "
                f"model: {model}, tokens: {self.ctx.current_context_tokens})",
                'success'), file=sys.stderr)
            return False
        return None

    def run_handle_save(self, full_input: str) -> Optional[bool]:
        """Handle /save command to save current session."""
        if full_input.startswith('/save'):
            if not hasattr(self, 'messages') or len(self.messages) <= 1:
                print(colorize("[Save] Nothing to save (empty session).", 'warning'), file=sys.stderr)
                return False
            sm = SessionManager()
            session_id = sm.save_session(self.messages, self.ctx)
            print(colorize(f"[Save] Session saved as {session_id}", 'success'), file=sys.stderr)
            return False
        return None

    def run_process_query(self, full_input: str) -> None:
        """Process regular user query (non-command input)."""
        if self.ctx.agentic_mode:
            return self.run_agentic_query(full_input)

        final_content = process_inline_commands(full_input)
        if not final_content.strip():
            return

        if not hasattr(self, 'messages'):
            sys_msg = {'role': 'system', 'content': self.ctx.system_prompt}
            self.ctx.stamp_tokens(sys_msg)
            self.messages = [sys_msg]
        user_msg = {'role': 'user', 'content': final_content}
        self.ctx.stamp_tokens(user_msg)
        self.messages.append(user_msg)
        sanitize_tool_pairing(self.messages)

        # Auto-compaction: compact if approaching context limit
        if self.ctx.context_window_size > 0:
            msg_tokens = self.ctx.calculate_context_tokens(self.messages)
            threshold = int(self.ctx.context_window_size * self.ctx.compaction_threshold)
            if msg_tokens > threshold:
                before_len = len(self.messages)
                self.messages = compact_messages(self.messages, self.ctx)
                after_tokens = self.ctx.calculate_context_tokens(self.messages)
                print(colorize(
                    f"[Auto-compact] {before_len} → {len(self.messages)} msgs, "
                    f"{msg_tokens} → {after_tokens} tokens",
                    'warning'), file=sys.stderr)

        payload_messages = list(self.messages)

        if payload_messages[-1]['role'] == 'user':
            if not self.ctx.model:
                print(colorize("\n[ERROR] No model selected. Use /switchmodel <name> to select a model first.", 'error'), file=sys.stderr)
                return

            response = self.query_handler.query_stream(
                payload_messages,
                self.ctx.model,
                stream_enabled=self.ctx.stream_enabled,
                debug=self.ctx.debug_mode,
                show_thinking=(not self.ctx.force_no_thinking),
                context_size=self.ctx.context_size,
                images=([] if self.ctx.supports_vision is False else self.ctx.current_images),
                no_thinking=self.ctx.force_no_thinking,
                reasoning_effort=self.ctx.reasoning_effort
            )

            if response:
                assistant_msg = {'role': 'assistant', 'content': response}
                self.ctx.stamp_tokens(assistant_msg)
                self.messages.append(assistant_msg)
                self.ctx.current_context_tokens = self.ctx.calculate_context_tokens(self.messages)
                print()


# ============================================================================
# ============= MAIN ENTRY POINT ===========================================
# ============================================================================


def list_models_llamacpp(base_url: str, filter_arg: Optional[str] = None,
                         api_key: Optional[str] = None) -> None:
    """List Llama.cpp / Gemini models.

    Args:
        base_url: Server URL.
        filter_arg: Optional name filter string.
        api_key: Optional API key for cloud backends.
    """
    models = fetch_models_llamacpp(base_url, api_key=api_key)

    if not models:
        print("\n[No models found via llamacpp API]", file=sys.stderr)
        return

    search_term = None
    if filter_arg:
        parts = filter_arg.lower().split()
        if 'name' in parts:
            parts.remove('name')

        if parts:
            search_term = parts[0]

    if search_term:
        models = [m for m in models if search_term in m.get('name', '').lower()]

    models.sort(key=lambda x: x.get('name', ''))

    print(f"\n{'NAME':<50} | {'OWNED BY'}")
    print("-" * 60)
    for m in models:
        print(f"{m['name']:<50} | {m.get('owned_by', 'N/A')}")

    print()


def _normalize_models_for_display(models: list, filter_arg: Optional[str], sort_by: str) -> tuple:
    """Filter, sort, and normalize model size fields for display.

    Args:
        models: Raw model list from the API.
        filter_arg: Optional name/size filter string.
        sort_by: 'name' or 'size'.

    Returns:
        (models, search_term) — normalized + sorted list and the applied term.
    """
    search_term = None

    if filter_arg:
        parts = filter_arg.lower().split()
        if 'size' in parts:
            sort_by = 'size'
            parts.remove('size')
        elif 'name' in parts:
            sort_by = 'name'
            parts.remove('name')
        if parts:
            search_term = parts[0]

    if search_term:
        models = [m for m in models if search_term in m['name'].lower()]

    for m in models:
        # Handle different possible size fields from the API
        raw_size = m.get("size") or m.get("size_bytes") or m.get("model_size") or 0
        try:
            m['size_bytes'] = int(raw_size)
        except (TypeError, ValueError):
            m['size_bytes'] = 0

    if sort_by == 'size':
        models.sort(key=lambda x: x['size_bytes'], reverse=True)
    else:
        models.sort(key=lambda x: x.get('name', ''))

    return models, search_term


def _print_models_table(models: list, file: object) -> None:
    """Print the NAME/SIZE/MODIFIED model table.

    Args:
        models: Normalized model list.
        file: Output stream.
    """
    header = f"{'NAME':<40} | {'SIZE':<12} | {'MODIFIED'}"
    print(colorize(header, 'muted'), file=file)
    print(colorize("-" * len(header), 'muted'), file=file)
    for m in models:
        size_str = parse_size(m.get('size') or m.get('size_bytes') or m.get('model_size') or 0)
        modified = m.get('modified_at', 'Unknown')[:10]
        print(f"{m['name']:<40} | {size_str:<12} | {modified}", file=file)
    print(file=file)


def _print_models_with_capabilities(base_url: str, models: list, file: object) -> None:
    """Fetch and print the model table including per-model capabilities.

    Args:
        base_url: Ollama server URL.
        models: Normalized model list.
        file: Output stream.
    """
    # Retrieve capabilities for each model (extra API calls)
    for m in models:
        try:
            info = fetch_model_info_ollama(base_url, m['name'])
            m['capabilities'] = ",".join(info.get('capabilities', []))
        except Exception:
            m['capabilities'] = ''
    header = f"{'NAME':<40} | {'SIZE':<12} | {'MODIFIED':<12} | {'CAPABILITIES'}"
    print(colorize(header, 'muted'), file=file)
    print(colorize("-" * len(header), 'muted'), file=file)
    for m in models:
        size_str = parse_size(m.get('size') or m.get('size_bytes') or m.get('model_size') or 0)
        modified = m.get('modified_at', 'Unknown')[:10]
        caps = m.get('capabilities', '')
        print(f"{m['name']:<40} | {size_str:<12} | {modified} | {caps}", file=file)
    print(file=file)


def list_models_ollama(base_url: str, filter_arg: Optional[str] = None,
                       include_capabilities: bool = False, file: object = None) -> None:
    """List models from an Ollama server, optionally with capabilities.

    Args:
        base_url: Ollama server URL.
        filter_arg: Optional name/size filter string.
        include_capabilities: Whether to fetch and show per-model capabilities.
        file: Output stream (defaults to sys.stdout).
    """
    if file is None:
        file = sys.stdout

    models = fetch_models_ollama(base_url)
    if not models:
        print(colorize(f"\nNo models found via Ollama API at {base_url}. Check if the server is running.\n", 'warning'), file=file)
        return

    sort_by = 'name'
    models, search_term = _normalize_models_for_display(models, filter_arg, sort_by)

    if search_term and not models:
        print(colorize(f"\nNo models found matching '{search_term}'.\n", 'warning'), file=file)
        return

    largest = max(models, key=lambda x: x['size_bytes']) if models else None
    if largest and largest['size_bytes'] > 0:
        l_size_gb = largest['size_bytes'] / (1024**3)
        print(f"\nChecking storage... Largest model in list: {largest['name']} ({l_size_gb:.2f} GB)\n", file=file)
    else:
        print(file=file)

    if include_capabilities:
        _print_models_with_capabilities(base_url, models, file)
    else:
        _print_models_table(models, file)
    print()


def show_model_info(base_url: str, model: str, args: argparse.Namespace) -> None:
    """Display model information.

    Args:
        base_url: Ollama server URL.
        model: Model name.
        args: Parsed CLI arguments (uses output_format).
    """
    info = fetch_model_info_ollama(base_url, model)

    if not info:
        print(f"[ERROR] No model info for '{model}'", file=sys.stderr)
        sys.exit(0)

    subset = {
        "details": info.get('details', {}),
        "model_info": info.get('model_info', {}),
        "capabilities": info.get('capabilities', [])
    }

    if args.output_format == "yaml" and HAVE_YAML:
        print(yaml.safe_dump(subset, sort_keys=False))
    else:
        print(json.dumps(subset, indent=2))

    sys.exit(0)


def show_model_details(base_url: str, model: str, args: argparse.Namespace) -> None:
    """Display full model details.

    Args:
        base_url: Ollama server URL.
        model: Model name.
        args: Parsed CLI arguments (uses output_format).
    """
    info = fetch_model_info_ollama(base_url, model)

    if not info:
        print(f"[ERROR] No model info for '{model}'. Check server and model name.",
              file=sys.stderr)
        sys.exit(0)

    if args.output_format == "yaml" and HAVE_YAML:
        print(yaml.safe_dump(info, sort_keys=False))
    else:
        print(json.dumps(info, indent=2))
    sys.exit(0)


def fetch_loaded_models_ollama(base_url: str) -> list:
    """Fetch models currently loaded in memory via Ollama /api/ps.

    Args:
        base_url: Ollama server URL.

    Returns:
        List of model dicts, or [] on failure.
    """
    try:
        url = f"{base_url}/api/ps"
        with _request_with_retry(Request(url, headers={'User-Agent': 'Mozilla/5.0'})) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('models', [])
    except Exception:
        return []

def fetch_loaded_models_context_ollama(base_url: str) -> list[tuple[str, int]]:
    """
    Query Ollama /api/ps and return a list of (model_name, context_size).

    Example output:
        [ ("nemotron-cascade-2:30b", 131072),
          ("llama2", 4096) ]
    """
    try:
        url = f"{base_url}/api/ps"
        with _request_with_retry(
            Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        ) as response:
            data = json.loads(response.read().decode('utf-8'))
            result = []
            for m in data.get("models", []):
                if isinstance(m, dict) and "name" in m:
                    val = m.get("context_size", m.get("context_length", 0))
                    if isinstance(val, (int, float)):
                        result.append((m["name"], int(val)))
                    else:
                        result.append((m["name"], 0))
            return result
    except Exception:                     # network error, parsing error, etc.
        sys.stderr.write(colorize("[ERROR] Could not read Ollama model info", "error"))
        return []


def check_backend_with_head(url: str, server_marker: str, timeout: float = 1) -> bool:
    """Attempt HEAD request to URL and check for server header.

    Args:
        url: URL to probe.
        server_marker: Marker to look for in the Server header.
        timeout: Socket timeout in seconds.
    """
    try:
        request = Request(url, method='HEAD')
        with urlopen(request, timeout=timeout) as response:  # startup-probe
            server_header = response.headers.get('Server', '').lower()
            return server_marker.lower() in server_header
    except HTTPError as e:
        # Even on error responses (e.g. 415 from llama.cpp root),
        # the Server header is still present and valid for detection
        server_header = e.headers.get('Server', '').lower()
        return server_marker.lower() in server_header
    except Exception:
        return False


def check_backend_with_get(url: str, server_marker: str, timeout: float = 1) -> bool:
    """Attempt GET request to URL and check for server marker.

    Args:
        url: URL to probe.
        server_marker: Marker to look for in the response body.
        timeout: Socket timeout in seconds.
    """
    try:
        request = Request(url, method='GET')
        with urlopen(request, timeout=timeout) as response:  # startup-probe
            my_response = response.read().decode('utf-8', errors='replace')
            my_response = my_response.lower()
            if server_marker.lower() in my_response:
                return True
    except Exception:
        return False



def check_lmstudio(url: str, timeout: float = 2) -> bool:
    """Check if LM Studio is running by querying /v1/models.

    Args:
        url: Base URL of the LM Studio server.
        timeout: Socket timeout in seconds.
    """
    try:
        request = Request(f"{url}/v1/models", method='GET')
        with urlopen(request, timeout=timeout) as response:  # startup-probe
            data = json.loads(response.read().decode('utf-8'))
            models = data.get('data', [])
            return bool(models)
    except Exception:
        return False


def check_gemini(url: str, api_key: Optional[str] = None, timeout: float = 2) -> bool:
    """Check if Gemini API is reachable by querying /v1/models.

    Args:
        url: Gemini OpenAI-compatible endpoint URL.
        api_key: API key (required; returns False without it).
        timeout: Socket timeout in seconds.
    """
    if not api_key:
        return False
    try:
        request = Request(f"{url}/v1/models", method='GET',
                          headers={'Authorization': f'Bearer {api_key}', 'User-Agent': 'Mozilla/5.0'})
        with _request_with_retry(request, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            models = data.get('data', [])
            return bool(models)
    except Exception:
        return False


def auto_detect_backend() -> tuple:
    """Auto-detect backend based on default ports using HEAD request.

    Checks sequentiall for:
    - 127.0.0.1:8080 for llama.cpp
    - 127.0.0.1:11434 for ollama
    - 127.0.0.1:1234 for lm studio

    Returns:
        tuple: (found, backend_name, url) or (None, '', '')
    """

    # Default URLs
    llama_cpp_url = DEFAULT_LLAMACPP_HOST
    ollama_url =    DEFAULT_OLLAMA_HOST
    lmstudio_url =  DEFAULT_LMSTUDIO_HOST

    # Check which backend is running

    sys.stderr.write(colorize("[INFO] AutoDetecting on : " + llama_cpp_url + " ", 'info'))
    if check_backend_with_head(llama_cpp_url, 'llama.cpp'):
        sys.stderr.write(colorize("Success\n", 'info'))
        return True,'llamacpp',llama_cpp_url
    else:
        sys.stderr.write(colorize("Fail\n", 'info'))

    sys.stderr.write(colorize("[INFO] AutoDetecting on : " + ollama_url    + " ", 'info'))
    if check_backend_with_get(ollama_url,     'ollama'):
        sys.stderr.write(colorize("Success\n", 'info'))
        return True,'ollama',ollama_url
    else:
        sys.stderr.write(colorize("Fail\n", 'info'))

    sys.stderr.write(colorize("[INFO] AutoDetecting on : " + lmstudio_url + " ", 'info'))
    if check_lmstudio(lmstudio_url):
        sys.stderr.write(colorize("Success\n", 'info'))
        return True,'lmstudio',lmstudio_url
    else:
        sys.stderr.write(colorize("Fail\n", 'info'))


    # grab the ip of the host
    try:
        list_of_ip = socket.gethostbyname_ex(socket.gethostname())[-1]
    except socket.error:
        list_of_ip = []
    for ip in list_of_ip:

        url="http://"+ip + ":" + str(DEFAULT_LLAMACPP_PORT)
        sys.stderr.write(colorize("[INFO] AutoDetecting on : " + url    + " ", 'info'))
        if check_backend_with_head(url, 'llama.cpp', timeout=0.2):
            sys.stderr.write(colorize("Success\n", 'info'))
            return True,'llamacpp',url
        else:
            sys.stderr.write(colorize("Fail\n", 'info'))

        url="http://"+ip + ":" + str(DEFAULT_OLLAMA_PORT)
        sys.stderr.write(colorize("[INFO] AutoDetecting on : " + url    + " ", 'info'))
        if check_backend_with_get("http://"+ip + ":" +  str(DEFAULT_OLLAMA_PORT),   'ollama', timeout=0.2):
            sys.stderr.write(colorize("Success\n", 'info'))
            return True,'ollama',url
        else:
            sys.stderr.write(colorize("Fail\n", 'info'))

        url="http://"+ip + ":" + str(DEFAULT_LMSTUDIO_PORT)
        sys.stderr.write(colorize("[INFO] AutoDetecting on : " + url    + " ", 'info'))
        if check_lmstudio(url, timeout=0.2):
            sys.stderr.write(colorize("Success\n", 'info'))
            return True,'lmstudio',url
        else:
            sys.stderr.write(colorize("Fail\n", 'info'))

    return None,'',''

def load_saved_backends() -> list:
    """Load the list of previously successful backend configurations.

    Returns:
        List of {"backend", "host"} dicts, or [] when absent/invalid.
    """
    config_file = os.path.expanduser("~/.ollamaquery.d/backends.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            sys.stderr.write(colorize("[WARNING] Failed to load saved backend config\n", 'warning'))
    return []

def save_backend_config(backend: str, host: str) -> None:
    """Save a successful connection to the top of the history list.

    Args:
        backend: Backend name.
        host: Backend base URL.
    """
    config_file = os.path.expanduser("~/.ollamaquery.d/backends.json")
    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        history = load_saved_backends()

        new_entry = {"backend": backend, "host": host}

        # Remove it if it already exists so we can bump it to the top
        history = [entry for entry in history if entry != new_entry]
        history.insert(0, new_entry)

        # Keep only the last 10 known servers to avoid bloat
        history = history[:10]

        with open(config_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        sys.stderr.write(colorize(f"[WARNING] Failed to save config: {e}\n", 'warning'))

def _probe_backend(backend: str, host: str, api_key: Optional[str] = None) -> bool:
    """Probe whether a backend is reachable at a given host URL.

    Args:
        backend: Backend name (ollama / llamacpp / lmstudio / cloud backends).
        host: Base URL of the server.
        api_key: Optional API key for cloud backends.

    Returns:
        bool: True if the server responded as expected.
    """
    if backend == "llamacpp":
        return check_backend_with_head(host, 'llama.cpp')
    if backend == "ollama":
        return check_backend_with_get(host, 'ollama')
    if backend == "lmstudio":
        return check_lmstudio(host)
    if backend in CLOUD_BACKENDS:
        key = api_key or os.environ.get(CLOUD_API_KEY_ENV.get(backend, ''), '')
        return check_gemini(host, key)
    return False


def _resolve_host_override(args: argparse.Namespace) -> tuple:
    """Resolve backend/host when the user passed an explicit -H host.

    Args:
        args: Parsed CLI arguments.

    Returns:
        (backend, base_url) tuple.

    Infers the backend from the port when -b is omitted; defaults to ollama.
    """
    base_url = args.host if args.host.startswith(('http://', 'https://')) else f"http://{args.host}"
    selected_backend = args.backend
    if not selected_backend:
        # Infer backend from known ports
        port_match = re.search(r':(\d+)(/|$)', base_url)
        if port_match:
            port = int(port_match.group(1))
            if port == DEFAULT_OLLAMA_PORT:
                selected_backend = "ollama"
            elif port == DEFAULT_LLAMACPP_PORT:
                selected_backend = "llamacpp"
            elif port == DEFAULT_LMSTUDIO_PORT:
                selected_backend = "lmstudio"
        if not selected_backend:
            selected_backend = "ollama"
    elif selected_backend == "gemini" and not base_url.startswith(('http://', 'https://')):
        # Treat bare host as gemini API URL, ensure https
        base_url = f"https://{base_url}" if not base_url.startswith('http') else base_url
    return selected_backend, base_url


def _probe_saved_backends(args: argparse.Namespace, saved_backends: list) -> Optional[tuple]:
    """Try each saved backend config (MRU first), returning the first reachable one.

    Args:
        args: Parsed CLI arguments.
        saved_backends: List of {"backend", "host"} config dicts.

    Returns:
        (backend, host) tuple, or None if none responded.
    """
    for config in saved_backends:
        s_backend = config.get('backend')
        s_host = config.get('host')

        # If user explicitly passed `-b`, skip history entries that don't match
        if args.backend and args.backend != s_backend:
            continue

        sys.stderr.write(colorize(f"[INFO] Testing known server: {s_backend} @ {s_host} ... ", 'muted'))

        is_valid = _probe_backend(s_backend, s_host, getattr(args, 'api_key', None))

        if is_valid:
            sys.stderr.write(colorize("Success\n", 'success'))
            save_backend_config(s_backend, s_host)  # Bump to top of list
            return s_backend, s_host
        sys.stderr.write(colorize("Offline\n", 'warning'))
    return None


def _resolve_fallback(args: argparse.Namespace) -> tuple:
    """Return the ultimate fallback (backend, host) when all probes failed.

    Args:
        args: Parsed CLI arguments.
    """
    fallback_backend = args.backend or "ollama"
    if fallback_backend in CLOUD_BACKENDS:
        env_var = CLOUD_HOST_ENV[fallback_backend]
        fallback_host = os.environ.get(env_var, CLOUD_DEFAULT_HOST[fallback_backend])
    elif fallback_backend == "llamacpp":
        fallback_host = os.environ.get('LLAMACPP_HOST', DEFAULT_LLAMACPP_HOST)
    elif fallback_backend == "lmstudio":
        fallback_host = os.environ.get('LMSTUDIO_HOST', DEFAULT_LMSTUDIO_HOST)
    else:
        fallback_host = os.environ.get('OLLAMA_HOST', DEFAULT_OLLAMA_HOST)
    return fallback_backend, fallback_host


def resolve_connection(args: argparse.Namespace) -> tuple:
    """
    Determines the correct backend and host by prioritizing:
    1. Explicit CLI overrides (-H and -b)
    2. Previously working configurations (tried Most Recently Used first)
    3. Network Auto-discovery
    4. Hardcoded defaults

    Args:
        args: Parsed CLI arguments.

    Returns:
        (backend, base_url) tuple.
    """
    # 1. Explicit user override (-H)
    if args.host:
        return _resolve_host_override(args)

    # Cloud services with fixed URLs — skip local probes
    if args.backend in CLOUD_BACKENDS:
        return args.backend, os.environ.get(CLOUD_HOST_ENV[args.backend], CLOUD_DEFAULT_HOST[args.backend])

    saved_backends = load_saved_backends()

    # 2. Iterate through history
    resolved = _probe_saved_backends(args, saved_backends)
    if resolved:
        return resolved

    # 3. If history failed or is empty, trigger Auto-Discovery
    sys.stderr.write(colorize("\n[INFO] Known servers offline. Initiating auto-discovery...\n", 'info'))
    autodetected, d_backend, d_url = auto_detect_backend()
    if autodetected:
        # If user explicitly passed `-b`, ensure the autodetected backend matches
        if not args.backend or args.backend == d_backend:
            save_backend_config(d_backend, d_url)
            return d_backend, d_url

    # 4. Ultimate Fallback
    sys.stderr.write(colorize("[WARNING] Auto-discovery failed. Falling back to defaults.\n", 'error'))
    return _resolve_fallback(args)


# ============================================================================
# ============= ARGUMENT PARSER ==============================================
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser with all options."""
    parser = argparse.ArgumentParser(
        description="Unified LLM Query Interface for Ollama, Llama.cpp & LM Studio"
    )

    parser.add_argument_group('Backend')
    parser.add_argument("-b", "--backend", choices=["ollama", "llamacpp", "lmstudio", "gemini", "opencodezen", "opencodego", "mistral", "deepseek"], default=None, help="API backend to use (auto-detected if omitted).")
    parser.add_argument('-H', '--host', help='Custom API URL')
    parser.add_argument('-k', '--api-key', help='API key for cloud backends (Gemini, etc.)')
    parser.add_argument('--version', action='store_true', help='Show version and exit')

    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument('-l', '--list', action="store_true", help='List all models and exit')
    list_group.add_argument('-la', '--list-all', action="store_true", help='List models with capabilities (Ollama only)')

    info_group = parser.add_mutually_exclusive_group()
    info_group.add_argument('--show', action="store_true", help='Show concise model details')
    info_group.add_argument('--show-details', action="store_true", help='Show full model information')

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-I', '--input-text', help='Direct query text')
    input_group.add_argument('-i', '--input-file', help='Input file path')
    input_group.add_argument('--input-dir', help='Directory of input files')

    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument('-c', '--chat', action="store_true", help='Start interactive chat session')
    batch_group.add_argument('-o', '--output', help='Output file path')
    batch_group.add_argument('--output-dir', help='Output directory for batches')

    parser.add_argument('-m', '--model', default=None, help='Model name')

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument('-P', '--profile', choices=list(BUILTIN_PROMPTS.keys()), help='Use a built-in system prompt profile')
    prompt_group.add_argument('--prompt', help='Custom system prompt text')

    parser.add_argument('--image', nargs='*', help='Image file(s) for multimodal models (space-separated)')
    parser.add_argument('-p', '--no-stream', action="store_true", help='Disable streaming output')
    parser.add_argument('--debug', action="store_true", help='Print raw JSON to stderr')
    parser.add_argument('--format', choices=["json", "yaml"], default="json", help='Output format for model info (default: json)')
    parser.add_argument('--theme', default=None, choices=["default", "minimal", "emacs_dark", "vim_dark", "high_contrast"], help='Color theme for output')
    parser.add_argument('--no-color', action="store_true", help='Disable colored output')
    parser.add_argument('--shell-timeout', type=int, default=5, help='Timeout in seconds for shell commands (default: 5)')

    return parser


def _try_known_port(backend: str, base_url: str, args: argparse.Namespace) -> Optional[tuple]:
    """Probe the backend's default port when -b was given but the bare host failed.

    Args:
        backend: Backend name.
        base_url: Bare host URL (no port).
        args: Parsed CLI arguments.

    Returns:
        (backend, url_with_port) on success, or None.
    """
    port_map = {"ollama": DEFAULT_OLLAMA_PORT, "llamacpp": DEFAULT_LLAMACPP_PORT, "lmstudio": DEFAULT_LMSTUDIO_PORT}
    port = port_map.get(backend, DEFAULT_OLLAMA_PORT)
    fallback = f"{base_url}:{port}"
    sys.stderr.write(colorize(f"[INFO] Checking {fallback}... ", 'muted'))
    ok = check_lmstudio(fallback) if backend == "lmstudio" else \
         check_backend_with_head(fallback, 'llama.cpp') if backend == "llamacpp" else \
         check_backend_with_get(fallback, 'ollama')
    if ok:
        sys.stderr.write(colorize(f"found {backend}\n", 'success'))
        return backend, fallback
    sys.stderr.write(colorize("no\n", 'warning'))
    return None


def _probe_all_default_ports(base_url: str) -> Optional[tuple]:
    """Probe each backend's default port when no -b was given.

    Args:
        base_url: Bare host URL (no port).

    Returns:
        (backend, url_with_port) on success, or None.
    """
    probes = [
        ("ollama", f"{base_url}:{DEFAULT_OLLAMA_PORT}", check_backend_with_get, 'ollama'),
        ("llamacpp", f"{base_url}:{DEFAULT_LLAMACPP_PORT}", check_backend_with_head, 'llama.cpp'),
        ("lmstudio", f"{base_url}:{DEFAULT_LMSTUDIO_PORT}", check_lmstudio, None),
    ]
    for probe_backend, probe_url, probe_fn, probe_marker in probes:
        sys.stderr.write(colorize(f"[INFO] Checking {probe_url}... ", 'muted'))
        try:
            ok = probe_fn(probe_url) if probe_marker is None else probe_fn(probe_url, probe_marker)
            if ok:
                sys.stderr.write(colorize(f"found {probe_backend}\n", 'success'))
                return probe_backend, probe_url
            sys.stderr.write(colorize("no\n", 'warning'))
        except Exception:
            sys.stderr.write(colorize("no\n", 'warning'))
            continue
    return None


def _verify_server(backend: str, base_url: str, args: argparse.Namespace) -> tuple:
    """Probe the server to confirm it's reachable, trying default ports if needed.

    Args:
        backend: Backend name.
        base_url: Base URL to probe.
        args: Parsed CLI arguments.

    Returns:
        (backend, base_url) — may fall back to a different backend/port
        if the initial guess was wrong. Exits with code 1 if unreachable.
    """
    server_reachable = False
    if backend in CLOUD_BACKENDS:
        api_key = args.api_key or os.environ.get(CLOUD_API_KEY_ENV.get(backend, ''), '')
        if not api_key:
            sys.stderr.write(colorize(f"[ERROR] {backend} requires an API key. Set --api-key, {CLOUD_API_KEY_ENV.get(backend, '')} env var, or save to config.\n", 'error'))
            sys.exit(1)
        server_reachable = check_gemini(base_url, api_key)
    else:
        server_reachable = _probe_backend(backend, base_url, args.api_key)

    if not server_reachable and not re.search(r':\d{2,5}(/|$)', base_url) and backend not in CLOUD_BACKENDS:
        if args.backend:
            result = _try_known_port(backend, base_url, args)
        else:
            result = _probe_all_default_ports(base_url)
        if result:
            backend, base_url = result
            server_reachable = True

    if not server_reachable:
        hint = ""
        has_port = re.search(r':\d{2,5}(/|$)', base_url)
        if backend in ("gemini", "opencodezen", "opencodego", "mistral", "deepseek"):
            hint = " (check your API key and internet connection)"
        elif backend == "llamacpp" and not has_port:
            hint = f" (try {base_url}:{DEFAULT_LLAMACPP_PORT})"
        elif backend == "lmstudio" and not has_port:
            hint = f" (try {base_url}:{DEFAULT_LMSTUDIO_PORT})"
        elif backend == "ollama" and not has_port:
            hint = f" (try {base_url}:{DEFAULT_OLLAMA_PORT})"
        sys.stderr.write(colorize(f"[ERROR] Cannot reach {backend} at {base_url}. Server may be offline.{hint}\n", 'error'))
        sys.exit(1)

    return backend, base_url


def _select_model(backend: str, base_url: str, args: argparse.Namespace) -> str:
    """Select the target model, auto-detecting from server if -m not given.

    Args:
        backend: Backend name.
        base_url: Backend base URL.
        args: Parsed CLI arguments.

    Returns:
        Model name or empty string if none available.
    """
    if args.model:
        return args.model

    if backend == "ollama":
        loaded = fetch_loaded_models_ollama(base_url)
        if loaded:
            model = loaded[0]['name']
            sys.stderr.write(colorize(f"[INFO] Auto-selected active model in memory: '{model}'\n", 'success'))
            return model
        all_models = fetch_models_ollama(base_url)
        if all_models:
            print(colorize("\n[No model loaded. Use /listmodel to see available models.]", 'info'), file=sys.stderr)
            return ""
        sys.stderr.write(colorize("[WARNING] No models available on Ollama server.\n", 'warning'))
        return ""

    LABELS = {"lmstudio": "LM Studio", "gemini": "Gemini", "opencodezen": "OpenCode Zen", "opencodego": "OpenCode Go", "mistral": "Mistral", "deepseek": "DeepSeek"}

    api_key = args.api_key or os.environ.get(CLOUD_API_KEY_ENV.get(backend, ''), '')
    available = fetch_models_llamacpp(base_url, api_key=api_key if backend in CLOUD_BACKENDS else None)
    if available:
        model = available[0]['name']
        label = LABELS.get(backend, "Llama.cpp")
        sys.stderr.write(colorize(f"[INFO] Auto-selected hosted model: '{model}'\n", 'success'))
        return model

    if backend in ("llamacpp", "lmstudio"):
        label = LABELS.get(backend, "Llama.cpp")
        sys.stderr.write(colorize(f"[INFO] {label} endpoint reachable but no model list exposed; using placeholder 'hosted-model'\n", 'info'))
        return "hosted-model"

    if backend in CLOUD_BACKENDS:
        label = LABELS.get(backend, backend)
        sys.stderr.write(colorize(f"[INFO] {label} endpoint reachable but no models listed; using placeholder '{backend}-model'\n", 'info'))
        return f"{backend}-model"

    label = LABELS.get(backend, "Llama.cpp")
    print(colorize(f"\n[No models available on {backend} server. Use /listmodel to see available models.]", 'warning'), file=sys.stderr)
    return ""


def _cloud_api_key(backend: str, api_key: Optional[str] = None) -> str:
    """Resolve the API key for a cloud backend from --api-key or the env.

    Args:
        backend: Backend name.
        api_key: Explicit --api-key value (may be None).

    Returns:
        str: The resolved key, or "" if neither is available.
    """
    return api_key or os.environ.get(CLOUD_API_KEY_ENV.get(backend, ''), '')


def _resolve_active_prompt(args: argparse.Namespace) -> str:
    """Resolve the system prompt from --prompt, --profile, or the default.

    Args:
        args: Parsed CLI arguments.
    """
    if args.prompt:
        return args.prompt
    if args.profile:
        return BUILTIN_PROMPTS[args.profile]
    return DEFAULT_SYSTEM_PROMPT


def _prepare_images_list(args: argparse.Namespace) -> Optional[list]:
    """Prepare image data list from --image args, or None when absent.

    Args:
        args: Parsed CLI arguments.
    """
    if not args.image:
        return None
    return [prepare_image_data(p) for p in args.image if p and prepare_image_data(p)]


def _make_query_handler(backend: str, base_url: str, args: argparse.Namespace) -> 'ModelQuery':
    """Build a ModelQuery handler configured with the resolved connection.

    Args:
        backend: Backend name.
        base_url: Backend base URL.
        args: Parsed CLI arguments.
    """
    qh = ModelQuery(context=CommandContext())
    qh.ctx.base_url = base_url
    qh.ctx.backend = backend
    qh.ctx.shell_timeout = args.shell_timeout
    qh.ctx.api_key = _cloud_api_key(backend, args.api_key)
    return qh


def _run_listing(backend: str, base_url: str, args: argparse.Namespace) -> None:
    """Handle -l / -la model listing, then exit.

    Args:
        backend: Backend name.
        base_url: Backend base URL.
        args: Parsed CLI arguments.
    """
    if backend in CLOUD_BACKENDS:
        list_models_llamacpp(base_url, filter_arg=args.model,
                             api_key=_cloud_api_key(backend, args.api_key))
    elif backend in ("llamacpp", "lmstudio"):
        list_models_llamacpp(base_url, filter_arg=args.model)
    elif args.list_all:
        list_models_ollama(base_url, filter_arg=args.model, include_capabilities=True)
    else:
        list_models_ollama(base_url, filter_arg=args.model, include_capabilities=False)
    sys.exit(0)


def _run_model_info(base_url: str, target_model: str, args: argparse.Namespace) -> None:
    """Handle --show / --show-details model info.

    Args:
        base_url: Backend base URL.
        target_model: Model name.
        args: Parsed CLI arguments.
    """
    if args.show:
        show_model_info(base_url, target_model, args)
    elif args.show_details:
        show_model_details(base_url, target_model, args)


def _run_chat(backend: str, base_url: str, target_model: str, args: argparse.Namespace) -> None:
    """Run the interactive chat loop and exit.

    Args:
        backend: Backend name.
        base_url: Backend base URL.
        target_model: Model name.
        args: Parsed CLI arguments.
    """
    ctx = CommandContext()
    ctx.base_url = base_url
    ctx.backend = backend
    ctx.model = target_model
    ctx.system_prompt = args.prompt
    ctx.shell_timeout = args.shell_timeout
    ctx.api_key = _cloud_api_key(backend, args.api_key)

    images_list = _prepare_images_list(args)
    if images_list:
        ctx.current_images = images_list

    should_stream = not args.no_stream and sys.stdout.isatty()
    loop = ChatLoop(ctx)
    loop.run(stream_enabled=should_stream, debug=args.debug, images=images_list)
    sys.exit(0)


def _sync_response_text(response: object, backend: str) -> str:
    """Extract printable text from a query_sync response (dict or string).

    Args:
        response: Dict or str returned by query_sync.
        backend: Backend name for shape selection.

    Returns:
        Printable text (including API error rendering).

    Handles API error dicts, ollama `message.content`, and OpenAI-compatible
    `choices[0].message.content` shapes. Returns the raw string for strings.
    """
    if not isinstance(response, dict):
        return str(response)
    if "error" in response:
        err = response["error"]
        return f"[API ERROR] {err.get('message', err) if isinstance(err, dict) else err}"
    if backend == "ollama":
        return response.get('message', {}).get('content', '')
    choices = response.get('choices', [])
    if choices:
        return choices[0].get('message', {}).get('content', '')
    return response.get('message', {}).get('content', '')


def _response_content(response: object) -> str:
    """Extract only the content from a sync response for --output file writes.

    Args:
        response: Dict or str returned by query_sync.
    """
    if not isinstance(response, dict):
        return str(response)
    content = response.get('message', {}).get('content', '')
    if not content:
        choices = response.get('choices', [])
        if choices:
            content = choices[0].get('message', {}).get('content', '')
    return content


def _print_single_response(response: object, backend: str) -> None:
    """Print a single-query response to stdout, with reasoning to stderr.

    Args:
        response: Dict or str returned by query_sync.
        backend: Backend name for shape selection.
    """
    content = ""
    thinking = ""
    if isinstance(response, dict):
        if "error" in response:
            err = response["error"]
            content = f"[API ERROR] {err.get('message', err) if isinstance(err, dict) else err}"
        else:
            msg = response.get('message', {}) if backend == "ollama" else \
                (response.get('choices', [{}])[0].get('message', {}) if response.get('choices') else {})
            content = msg.get('content', '')
            if not content:
                content = response.get('message', {}).get('content', '')
            thinking = msg.get('reasoning_content', '') or msg.get('thought', '') or msg.get('thinking', '')
    else:
        content = str(response)
    if thinking:
        sys.stderr.write(f"\n<thinking>\n{thinking}\n</thinking>\n")
    print(content)


def _run_batch(backend: str, base_url: str, target_model: str, args: argparse.Namespace) -> None:
    """Process --input-dir of files, writing one `.output` per file, then exit.

    Args:
        backend: Backend name.
        base_url: Backend base URL.
        target_model: Model name.
        args: Parsed CLI arguments.
    """
    if not args.output_dir:
        print("[ERROR] --output-dir required for --input-dir", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    images_list = _prepare_images_list(args)
    for filename in sorted(os.listdir(args.input_dir)):
        input_path = os.path.join(args.input_dir, filename)
        if not os.path.isfile(input_path):
            continue
        print(f"[Processing: {filename}...]")
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        messages = [
            {'role': 'system', 'content': args.prompt},
            {'role': 'user', 'content': content}
        ]
        qh = _make_query_handler(backend, base_url, args)
        response = qh.query_sync(messages, target_model, context_size=None,
                                 show_thinking=True, debug=args.debug, images=images_list)
        output_text = _sync_response_text(response, backend)
        with open(os.path.join(args.output_dir, filename + '.output'), 'w', encoding='utf-8') as f:
            f.write(output_text)
    sys.exit(0)


def _run_single(backend: str, base_url: str, target_model: str, args: argparse.Namespace) -> None:
    """Handle a single -I / -i query (optionally writing to --output).

    Args:
        backend: Backend name.
        base_url: Backend base URL.
        target_model: Model name.
        args: Parsed CLI arguments.
    """
    images_list = _prepare_images_list(args)
    messages = [{'role': 'system', 'content': args.prompt}]
    if args.input_text:
        messages.append({'role': 'user', 'content': args.input_text})
    elif args.input_file and os.path.isfile(args.input_file):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            messages.append({'role': 'user', 'content': f.read()})

    qh = _make_query_handler(backend, base_url, args)
    response = qh.query_sync(messages, target_model, context_size=None,
                             show_thinking=True, debug=args.debug, images=images_list)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(_response_content(response))
        print(f"[Success: Output saved to {args.output}]", file=sys.stderr)
    elif response:
        _print_single_response(response, backend)


def _print_no_action_help(args: argparse.Namespace) -> None:
    """Print the "no action specified" banner and exit.

    Args:
        args: Parsed CLI arguments.
    """
    print(colorize(f"ollamaquery2 v{__version__} - LLM Query Interface", 'info'))
    if args.backend or args.host:
        print(f"  Backend configured ({args.backend or 'auto'} @ {args.host or 'auto'}), but no action specified.")
    print("  Start chat:  -c")
    print("  Single query: -I \"your prompt\"")
    print("  List models: -l")
    print("  Help:        --help")
    sys.exit(2)


def _exit_if_no_model(backend: str, base_url: str, args: argparse.Namespace) -> None:
    """Exit when no model is available/selected for the requested operation.

    Args:
        backend: Backend name.
        base_url: Backend base URL.
        args: Parsed CLI arguments.
    """
    sys.stderr.write(colorize(f"[INFO] Connected to {backend} at {base_url}\n", 'success'))
    if args.show or args.show_details:
        sys.stderr.write(colorize("[ERROR] No model selected. Use -m or --list to browse models.\n", 'error'))
        sys.exit(1)
    if args.input_text or args.input_file or args.input_dir:
        sys.stderr.write(colorize("[ERROR] No model selected. Use -m to specify a model.\n", 'error'))
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.version:
        print(f"ollamaquery2 v{__version__}")
        sys.exit(0)

    if args.no_color:
        os.environ['NO_COLOR'] = '1'
    elif args.theme is not None:
        os.environ['OLLAMAQUERY_THEME'] = args.theme

    args.prompt = _resolve_active_prompt(args)

    if not (args.chat or args.input_text or args.input_file or
             args.input_dir or args.list or args.show or args.show_details):
        _print_no_action_help(args)

    backend, base_url = resolve_connection(args)
    backend, base_url = _verify_server(backend, base_url, args)
    save_backend_config(backend, base_url)
    target_model = _select_model(backend, base_url, args)

    if not target_model:
        _exit_if_no_model(backend, base_url, args)

    # Listing operations
    if args.list or args.list_all:
        _run_listing(backend, base_url, args)

    # Model info operations
    if args.show or args.show_details:
        _run_model_info(base_url, target_model, args)

    # Interactive chat mode
    if args.chat:
        _run_chat(backend, base_url, target_model, args)

    # Batch / single query processing
    if args.input_dir:
        _run_batch(backend, base_url, target_model, args)

    if args.input_text or args.input_file:
        _run_single(backend, base_url, target_model, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\n[Exiting gracefully...]")
        sys.exit(0)


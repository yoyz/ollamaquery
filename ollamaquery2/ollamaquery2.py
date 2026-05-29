#!/usr/bin/python3
# vim: set ts=4 sw=4 sts=4 et:
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


__version__ = "0.1.4"


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
MAX_READ_FILE_SIZE = 102400    # 100KB max for agentic read_file tool
MAX_WRITE_FILE_SIZE = 1048576  # 1MB max for agentic write_file tool
MAX_FILE_INCLUSION_SIZE = 5 * 1024 * 1024  # 5MB max for @file inclusions
DEFAULT_OLLAMA_HOST    = 'http://127.0.0.1:11434'
DEFAULT_LLAMACPP_HOST  = 'http://127.0.0.1:8080'
DEFAULT_LMSTUDIO_HOST  = 'http://127.0.0.1:1234'
DEFAULT_OLLAMA_PORT    =  11434
DEFAULT_LLAMACPP_PORT  =  8080
DEFAULT_LMSTUDIO_PORT  =  1234



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
    'debug': {
        'aliases': ['/debug'],
        'category': 'Settings',
        'description': 'Configure per-category debug levels',
        'usage': '/debug <category> <level> | /debug list | /debug status',
        'handler': 'handle_debug_command'
    },
    'agentic': {
        'aliases': ['/agentic'],
        'category': 'Settings',
        'description': 'Configure agentic mode and sub-options',
        'usage': '/agentic [on|off|full|auto|sandbox|verbose|thinking|trace|log|iterations|timeout|status]',
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
}

# Category order for help display
COMMAND_CATEGORIES = ['Core', 'Model', 'Settings', 'I/O', 'Advanced']


def get_command_aliases():
    """Return flat list of all command aliases for readline completion."""
    aliases = []
    for cmd in COMMANDS.values():
        aliases.extend(cmd['aliases'])
    return aliases


def get_commands_by_category(category=None):
    """Return commands grouped by category, or filtered by category."""
    if category:
        return {k: v for k, v in COMMANDS.items() if v['category'] == category}
    
    # Group by category in defined order
    grouped = {}
    for cat in COMMAND_CATEGORIES:
        cat_cmds = {k: v for k, v in COMMANDS.items() if v['category'] == cat}
        if cat_cmds:
            grouped[cat] = sorted(cat_cmds.items(), key=lambda x: x[0])
    return grouped


def format_help_text(compact=False):
    """Generate formatted help text for display."""
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


def is_known_command(text):
    """Check if text matches any known command or alias."""
    text_lower = text.lower().strip()
    for cmd in COMMANDS.values():
        if text_lower in [a.lower() for a in cmd['aliases']]:
            return True, cmd
    return False, None


#
# === Color management 
# 


def colors_enabled():
    """Check if colors should be used (TTY check + NO_COLOR env var)."""
    if os.environ.get('NO_COLOR'):
        return False
    return sys.stdout.isatty()


def load_custom_themes():
    """Load custom themes from JSON file."""
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


def get_theme(theme_name: Optional[str] = None):
    """Get theme color dictionary."""
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


def colorize(text, role, theme=None, force_color=False, is_prompt=False):
    """Apply color to text using active theme."""
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

def _request_with_retry(req, max_retries=3, delay=1, **kwargs):
    """Open URL with retry on transient network errors.

    Retries on URLError, HTTPError (5xx only), ConnectionError, TimeoutError,
    and OSError. Does NOT retry HTTP 4xx client errors.

    Args:
        req: URL string or Request object to pass to urlopen
        max_retries: Maximum number of attempts (default 3)
        delay: Seconds to wait between retries (default 1)
        **kwargs: Additional arguments passed to urlopen (e.g. timeout)

    Returns:
        Same as urlopen(req, **kwargs)

    Raises:
        HTTPError: For 4xx errors or if all retries exhausted
        URLError: If all retries exhausted
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return urlopen(req, **kwargs)
        except (URLError, HTTPError, ConnectionResetError, ConnectionError, TimeoutError, OSError) as e:
            last_exception = e
            if isinstance(e, HTTPError) and e.code < 500:
                raise  # Don't retry 4xx client errors
            if attempt < max_retries - 1:
                sys.stderr.write(
                    colorize(f"\n[RETRY] Request failed ({e}), "
                             f"retrying in {delay}s (attempt {attempt+1}/{max_retries})\n",
                             'warning')
                )
                time.sleep(delay)
                continue
            raise
    raise last_exception  # Shouldn't reach here


# ============================================================================
# ============= CONTEXT WINDOW TRACKING ======================================
# ============================================================================

def get_ollama_context_size(base_url: str, model_name: str) -> int:
    """Get the actual context window size from Ollama's running model."""
    try:
        url = f"{base_url}/api/ps"
        with _request_with_retry(Request(url)) as response:
            data = json.loads(response.read().decode('utf-8'))
            models = data.get("models", [])
            for model in models:
                if model.get("name", "").startswith(model_name):
                    size = model.get("context_size", 0)
                    if size > 0:
                        return size
        
        url = f"{base_url}/api/show"
        payload = json.dumps({"name": model_name}).encode('utf-8')
        req = Request(url, data=payload, 
                     headers={'Content-Type': 'application/json'}, method='POST')
        with _request_with_retry(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get("context_size", 0)
    except Exception:
        sys.stderr.write(colorize(f"[WARNING] Failed to get Ollama context size for '{model_name}'\n", 'warning'))
    return 0


# update the ctx.context_window_size by querying the LLM server
# ctx is an object which contain ctx.context_window_size
def refresh_context_window_size(ctx):
    """Fetch and update context window size from the backend."""
    if ctx.backend == "ollama":
        size = get_ollama_context_size(ctx.base_url, ctx.model)
    elif ctx.backend == "lmstudio":
        size = 0
    else:
        size = get_llamacpp_context_size(ctx.base_url)
    
    if size > 0:
        ctx.context_window_size = size
        return True
    return False

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

def sanitize_shell_command(command):
    """
    Sanitize shell command to prevent injection attacks.

    Args:
        command (str): User input to sanitize

    Returns:
        str: Sanitized command or None if unsafe
    """
    if not command:
        return None

    # Remove dangerous characters that enable command chaining
    dangerous_patterns = [';', '|', '&&', '||', '`', '$(', '>', '<']
    for pattern in dangerous_patterns:
        if pattern in command:
            return None

    # Escape special shell characters
    command = command.replace('`', '')  # Remove backticks
    command = re.sub(r'\$\{[^}]+\}', 'SAFE_VAR', command)  # Hide variable expansion

    return command


def validate_shell_command_safety(command, max_length=500):
    """
    Validate that a shell command is safe to execute.

    Args:
        command (str): Command to validate
        max_length (int): Maximum allowed length

    Returns:
        bool: True if command is safe
    """
    if not command:
        return False

    # Length limit
    if len(command) > max_length:
        return False

    # Block dangerous utilities and substitution patterns
    dangerous = ['rm -rf', 'rm -r -f', 'rm -r /', 'mv / ', 'mkfs', 'dd if=', 'wget ', 'curl ',
                 'nc -e', 'python -c', 'perl -e', 'bash -c']
    for pattern in dangerous:
        if pattern.lower() in command.lower():
            return False

    # Block command substitution $() (modern shell syntax)
    if '$(' in command:
        return False

    # Check for shell escape sequences
    if '\\"' in command or "\\'" in command or '\\$' in command or '\\`' in command:
        return False

    # Block command chaining operators — only one command at a time
    chain_ops = ['&&', '||', ';', '|']
    for op in chain_ops:
        if op in command:
            return False

    return True


def strip_ansi(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def prepare_image_data(image_path):
    """Reads an image file and returns its base64 encoded string."""
    if not image_path or not os.path.isfile(image_path):
        return None

    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(colorize(f"[ERROR] Loading image {image_path}: {e}", 'error'), file=sys.stderr)
        return None


def guess_image_mime(b64_data):
    """Guess image MIME type from base64-encoded data prefix."""
    if b64_data.startswith('/9j/'):
        return "jpeg"
    if b64_data.startswith('iVBOR'):
        return "png"
    if b64_data.startswith('R0lGOD'):
        return "gif"
    if b64_data.startswith('UklGR'):
        return "webp"
    return "jpeg"


def fetch_models_ollama(base_url):
    """Fetch available models from Ollama API."""
    try:
        url = f"{base_url}/api/tags"
        with _request_with_retry(Request(url, headers={'User-Agent': 'Mozilla/5.0'})) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('models', [])
    except Exception:
        return []


def fetch_models_llamacpp(base_url):
    """Fetch available models from Llama.cpp API."""
    try:
        url = f"{base_url}/v1/models"
        with _request_with_retry(Request(url, headers={'User-Agent': 'Mozilla/5.0'})) as response:
            data = json.loads(response.read().decode('utf-8'))
            models = data.get('data', [])
            if not models:
                models = data.get('models', [])
            return [{'name': m.get('id', m.get('name', 'unknown'))} for m in models]
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
        sys.stderr.write(colorize(f"[WARNING] Failed to get Llama.cpp context size\n", 'warning'))
    return -1

def get_message_token_count_llamacpp(base_url: str, text: str) -> int:
    """Get exact token count using the /tokenize endpoint (no GPU overhead)."""
    try:
        url = f"{base_url}/tokenize"
        payload = json.dumps({"content": text}).encode('utf-8')
        req = Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with _request_with_retry(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            return len(data.get('tokens', []))
    except Exception as e:
        global _TOKEN_COUNT_WARNED
        if not _TOKEN_COUNT_WARNED:
            sys.stderr.write(colorize(f"[WARNING] Token counting failed (llama.cpp): {e}\n", 'warning'))
            _TOKEN_COUNT_WARNED = True
        return estimate_token_count(text)


def get_message_token_count_ollama(base_url: str, text: str, model: str) -> int:
    """Get exact token count using the /api/tokenize endpoint (no GPU overhead)."""
    if not model:
        return estimate_token_count(text)
    try:
        url = f"{base_url}/api/tokenize"
        payload = json.dumps({"model": model, "content": text}).encode('utf-8')
        req = Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with _request_with_retry(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            return len(data.get('tokens', []))
    except Exception as e:
        global _TOKEN_COUNT_WARNED
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

def is_available_llamacpp_model(base_url: str, model_name: str) -> bool:
    """
    Check if a model exists on the Llama.cpp server.
    
    Args:
        base_url: Llama.cpp server URL
        model_name: Model name to check
    
    Returns:
        True if model exists, False otherwise
    """
    try:
        models = fetch_models_llamacpp(base_url)
        model_names = [m.get('name', '') for m in models]
        return model_name in model_names
    except Exception:
        return False


def parse_size(size_bytes):
    """Parse size from the API into human-readable format."""
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


def estimate_token_count(text: str) -> int:
    """Estimate token count from text using regex-based heuristic.
    
    Falls back to a safe overestimate when API tokenization is unavailable.
    Detects code content and uses a higher multiplier for safety.
    """
    if not text:
        return 0
    tokens = len(re.findall(r'\b\w+\b|[^\w\s]', text))
    code_keywords = {'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'return', 'for', 'while', 'try', 'except', 'with', 'as', 'pass', 'raise', 'lambda', 'yield', 'async', 'await'}
    if any(kw in text for kw in code_keywords):
        return max(1, int(tokens * 2.0))
    return max(1, int(tokens * 1.5))


def fetch_model_info_ollama(base_url, model_name):
    """Fetch detailed model information via Ollama /api/show endpoint."""
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
    
    Call map:
      create_completer() → ChatCompleter
      create_query_handler() → ModelQuery
      create_executor() → Executor
      create_tool_registry() → ToolRegistry
      update_stats / get_cumulative_stats / estimate_tokens / calculate_context_tokens
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if CommandContext._initialized:
            return
        CommandContext._initialized = True
        self.shell_timeout: int = 5
        self.debug_manager = DebugManager() 
        # Connection info
        self._base_url: str = ""
        self._backend: str = "ollama"
        self._model: str = "llama3"
        
        # Session state
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.context_size: Optional[int] = None
        self.current_images: List[str] = []
        self.force_no_thinking: bool = False
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
        self.agentic_step_timeout: int = 120
        self.lazy_tool: bool = True  # Enabled by default: many models embed tool calls after thinking/preamble
        self.supports_vision: Optional[bool] = None  # None = unknown (treated as capable)
        
        # Context window tracking
        self.context_window_size: int = 0  # Will be fetched from server
        self.current_context_tokens: int = 0  # Updated after each query

    # === Properties ===
    @property
    def base_url(self) -> str:
        return self._base_url
    
    @base_url.setter
    def base_url(self, value: str):
        self._base_url = value

    @property
    def backend(self) -> str:
        return self._backend
    
    @backend.setter
    def backend(self, value: str):
        self._backend = value

    @property
    def model(self) -> str:
        return self._model
    
    @model.setter
    def model(self, value: str):
        self._model = value

    # === Helper Methods ===
    def reset(self):
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

    def create_completer(self):
        """Create a ChatCompleter using this context's connection info."""
        return ChatCompleter(self.base_url, self.backend)
    
    def create_query_handler(self):
        """Create a ModelQuery using this context's connection info."""
        return ModelQuery(self.base_url, self.backend)

    def create_executor(self):
        """Create an Executor for agentic tool execution."""
        return Executor()

    def create_tool_registry(self):
        """Create a ToolRegistry for agentic mode."""
        executor = self.create_executor()
        return ToolRegistry(ctx=self, executor=executor)

    def update_stats(self, tokens: int, prompt_tokens: int, time_spent: float, chars: int):
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

    def get_cumulative_stats(self):
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

    def calculate_context_tokens(self, messages: list) -> int:
        """Calculate estimated total tokens in conversation context."""
        total = 0
        for msg in messages:
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
            total += image_count * 1024
            total += 2
        return total


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
        model_name: Used to select format style from registry.
        tool_defs_block: Tool definitions block (from ToolRegistry.get_system_prompt_block()).
            Inserted after the role block so models see available tools before format instructions.
        include_tool_defs: If False, skip tool_defs_block (for models using native tools API).
    """
    style = get_prompt_style(model_name)
    blocks = [AGENTIC_ROLE_BLOCK]
    if include_tool_defs and tool_defs_block:
        blocks.append(tool_defs_block)
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
        # See doc/model-parameters.md for full test results.
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


def _extract_error_text(obj, depth=0):
    """Recursively extract error text from nested structures."""
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


def check_vision_error(response) -> bool:
    """Check if an API response dict indicates the model doesn't support vision.
    
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


def check_tools_error(response) -> bool:
    """Check if an API response dict indicates the model doesn't support native tools.
    
    Works for Ollama (/api/chat returns {"error": "..."}),
    and OpenAI-compatible APIs (/v1/chat/completions returns {"error": {"message": "..."}}).
    """
    if not isinstance(response, dict):
        return False
    error_text = _extract_error_text(response.get("error", ""))
    return any(kw in error_text.lower() for kw in _TOOLS_ERROR_KEYWORDS)


# ============================================================================
# ============= EXECUTOR (CONTAINER/HOST)  ===================================
# ============================================================================
# ============= EXECUTOR (CONTAINER/HOST)  ===================================
# ============================================================================

class Executor:
    """Runs shell commands on host or inside a container sandbox.
    
    Call map:
      run() → _run_shell() or _pre_pull_image()
      _run_shell() → subprocess.run / podman|docker exec
    """

    def __init__(self, mode="host", container_runtime=None, container_image=None):
        self.mode = mode
        self.runtime = container_runtime or os.environ.get("OLLAMAQUERY_CONTAINER_RT", "podman")
        self.image = container_image or os.environ.get("OLLAMAQUERY_CONTAINER_IMAGE",
                                                       "docker.io/library/python:3.12-alpine")

    def _pre_pull_image(self):
        """Pull the container image with a separate timeout so pulls don't consume command timeout."""
        try:
            subprocess.run(
                [self.runtime, "pull", self.image],
                capture_output=True, timeout=120, check=False
            )
        except Exception:
            pass

    def run(self, command: str, timeout: int = 120) -> dict:
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
        if not is_container and "python3 -c" not in command:
            if not validate_shell_command_safety(command, max_length=1000):
                return {"stdout": "", "stderr": "Command rejected by safety validator", "returncode": -1}
        try:
            args_list = shlex.split(command)
            proc = subprocess.run(
                args_list, shell=False,  # Intentional: shell=True would enable pipes/redirects but risks injection. Single commands only.
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout
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
        "description": "Read a file from disk (text, max 100KB). Path relative to CWD.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path relative to CWD"}
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


# ============================================================================
# ============= AGENTIC TOOL HANDLERS         ================================
# ============================================================================

def _tool_handle_fetch_url(self, args):
    url = args["url"]
    text, _tool = fetch_and_convert_url(url)
    return {"success": True, "output": text, "error": None}


def _tool_handle_read_file(self, args):
    base_dir = os.path.abspath(os.getcwd())
    filepath = os.path.abspath(os.path.join(os.getcwd(), args["file_path"]))
    if os.path.commonpath([base_dir, filepath]) != base_dir:
        return {"success": False, "output": "", "error": "Path traversal denied"}
    if not os.path.isfile(filepath):
        return {"success": False, "output": "", "error": "File not found"}
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_READ_FILE_SIZE)
        return {"success": True, "output": content, "error": None}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def _tool_handle_write_file(self, args):
    base_dir = os.path.abspath(os.getcwd())
    filepath = os.path.abspath(os.path.join(os.getcwd(), args["file_path"]))
    if os.path.commonpath([base_dir, filepath]) != base_dir:
        return {"success": False, "output": "", "error": "Path traversal denied"}
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


def _tool_handle_list_directory(self, args):
    base_dir = os.path.abspath(os.getcwd())
    path = os.path.abspath(os.path.join(os.getcwd(), args.get("path", ".")))
    if os.path.commonpath([base_dir, path]) != base_dir:
        return {"success": False, "output": "", "error": "Path traversal denied"}
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


def _tool_handle_glob(self, args):
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


def _tool_handle_run_python(self, args):
    if "code" in args:
        command = f"python3 -c {shlex.quote(args['code'])}"
    elif "file_path" in args:
        command = f"python3 {shlex.quote(args['file_path'])}"
    elif "file" in args:
        command = f"python3 {shlex.quote(args['file'])}"
    else:
        return {"success": False, "output": "", "error": "Provide either 'code' or 'file'"}
    cmd_timeout = int(args.get("timeout", 10))
    if cmd_timeout > 300:
        return {"success": False, "output": "", "error": f"Invalid timeout: {cmd_timeout}. Timeout is in seconds (max 300). Use a lower value."}
    result = self.executor.run(command, timeout=cmd_timeout)
    output = result["stdout"]
    if result["stderr"]:
        output += f"\n[stderr]\n{result['stderr']}"
    return {"success": result["returncode"] == 0, "output": output, "error": result["stderr"] or None}


def _tool_handle_run_command(self, args):
    command = args["command"]
    for op in ("&&", "||", "|", ";", "`", "$(", ">", "<"):
        if op in command:
            return {"success": False, "output": "", "error": f"Only one command at a time is supported. Remove shell operators like '{op}' and run commands separately."}
    cmd_timeout = int(args.get("timeout", 10))
    if cmd_timeout > 300:
        return {"success": False, "output": "", "error": f"Invalid timeout: {cmd_timeout}. Timeout is in seconds (max 300). Use a lower value."}
    result = self.executor.run(command, timeout=cmd_timeout)
    output = result["stdout"]
    if result["stderr"]:
        output += f"\n[stderr]\n{result['stderr']}"
    return {"success": result["returncode"] == 0, "output": output, "error": result["stderr"] or None}


def _tool_handle_diff(self, args):
    base_dir = os.path.abspath(os.getcwd())
    filepath1 = os.path.abspath(os.path.join(os.getcwd(), args["file1"]))
    filepath2 = os.path.abspath(os.path.join(os.getcwd(), args["file2"]))
    if os.path.commonpath([base_dir, filepath1]) != base_dir:
        return {"success": False, "output": "", "error": "Path traversal denied"}
    if os.path.commonpath([base_dir, filepath2]) != base_dir:
        return {"success": False, "output": "", "error": "Path traversal denied"}
    label1 = args.get("label1", args["file1"])
    label2 = args.get("label2", args["file2"])
    try:
        with open(filepath1, "r") as f:
            lines1 = f.readlines()
        with open(filepath2, "r") as f:
            lines2 = f.readlines()
        diff = list(difflib.unified_diff(lines1, lines2, fromfile=label1, tofile=label2))
        output = "".join(diff) if diff else "Files are identical"
        return {"success": True, "output": output, "error": None}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def _tool_handle_patch(self, args):
    import tempfile
    diff_text = args["diff"]
    diff_path = None
    target_arg = args.get("target", "").strip()
    if target_arg:
        base_dir = os.path.abspath(os.getcwd())
        target_raw = os.path.abspath(os.path.join(os.getcwd(), target_arg))
        if os.path.commonpath([base_dir, target_raw]) != base_dir:
            return {"success": False, "output": "", "error": "Path traversal denied"}
        target = shlex.quote(target_raw)
    else:
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


def _tool_handle_edit_file(self, args):
    base_dir = os.path.abspath(os.getcwd())
    filepath = os.path.abspath(os.path.join(os.getcwd(), args["file_path"]))
    if os.path.commonpath([base_dir, filepath]) != base_dir:
        return {"success": False, "output": "", "error": "Path traversal denied"}
    if not os.path.isfile(filepath):
        return {"success": False, "output": "", "error": f"File not found: {args['file_path']}"}
    old = args["old_string"]
    new_string = args["new_string"]
    if not old:
        return {"success": False, "output": "", "error": "old_string must not be empty"}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
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


def _parse_patch_sections(patch_text):
    """Parse patch text into sections for each file.
    
    Returns list of dicts:
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
            marker = line[4:].strip()
            if marker.startswith("Add File:"):
                path = marker[len("Add File:"):].strip()
                op = "add"
            elif marker.startswith("Update File:"):
                path = marker[len("Update File:"):].strip()
                op = "modify"
            elif marker.startswith("Delete File:"):
                path = marker[len("Delete File:"):].strip()
                sections.append({"path": path, "operation": "delete", "hunks": []})
                i += 1
                continue
            elif marker.startswith("Move to:"):
                dest = marker[len("Move to:"):].strip()
                # The marker line tells us the destination; the source comes from ---/+++ headers or a path before this
                path = dest  # We'll override if ---/+++ follow
                op = "move"
            else:
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
            if detected_path and not marker.startswith("Move to:"):
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
        if line.startswith("--- "):
            old_path = line[4:].strip()
            if i + 1 < n and lines[i + 1].startswith("+++ "):
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
                while i < n and not lines[i].startswith("--- "):
                    i += 1
                body = "".join(lines[body_start:i])
                hunks, _, _ = _parse_unified_hunks(body)

                if is_delete:
                    sections.append({"path": path, "operation": "delete", "hunks": []})
                elif is_new:
                    sections.append({"path": path, "operation": "add", "hunks": hunks})
                else:
                    sections.append({"path": path, "operation": "modify", "hunks": hunks})
                continue
        i += 1

    return sections


def _parse_unified_hunks(body):
    """Parse @@ hunks from a unified diff body.
    
    Returns (hunks, detected_path, is_delete).
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
            p = line[4:].strip()
            if p != "/dev/null":
                detected_path = p[2:] if p.startswith(("a/", "b/")) else p
            else:
                is_delete = False
            i += 1
            continue
        if line.startswith("+++ "):
            p = line[4:].strip()
            if p != "/dev/null":
                detected_path = p[2:] if p.startswith(("a/", "b/")) else p
            else:
                is_delete = True
            i += 1
            continue

        # Parse hunk header: @@ -start,count +start,count @@
        m = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        if m:
            start = int(m.group(1))
            old_count = int(m.group(2) or 1)
            new_count = int(m.group(4) or 1)
            i += 1

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

            hunks.append({"start": start, "old_count": old_count, "old_lines": old_lines, "new_lines": new_lines})
            continue

        i += 1

    return hunks, detected_path, is_delete


def _apply_unified_diff(patch_text):
    """Apply a unified diff patch to the filesystem. Pure Python.
    
    Supports standard unified diff (---/+++ headers, @@ hunks)
    and OpenCode-style markers (*** Add/Update/Delete/Move to: path).
    """
    sections = _parse_patch_sections(patch_text)
    if not sections:
        return {"success": False, "output": "", "error": "No valid patch sections found in patch_text"}

    applied = []
    base_dir = os.path.abspath(os.getcwd())
    for sec in sections:
        path = sec["path"]
        abspath = os.path.abspath(os.path.join(os.getcwd(), path))
        if os.path.commonpath([base_dir, abspath]) != base_dir:
            return {"success": False, "output": "", "error": f"Path traversal denied: {path}"}

        op = sec.get("operation", "modify")

        if op == "delete":
            if os.path.isfile(abspath):
                os.unlink(abspath)
            applied.append(f"Deleted {path}")
            continue

        if op == "move":
            dest = sec["destination"]
            absdest = os.path.abspath(os.path.join(os.getcwd(), dest))
            if os.path.commonpath([base_dir, absdest]) != base_dir:
                return {"success": False, "output": "", "error": f"Path traversal denied: {dest}"}
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
        for hunk in sorted(sec["hunks"], key=lambda h: h["start"], reverse=True):
            start_idx = hunk["start"] - 1
            old_count = hunk["old_count"]
            new_lines = hunk["new_lines"]

            if start_idx < 0:
                start_idx = 0
            if start_idx + old_count > len(content):
                old_count = len(content) - start_idx
            if old_count < 0:
                old_count = 0

            existing = content[start_idx:start_idx + old_count]
            expected = hunk.get("old_lines", [])
            existing_clean = [line.rstrip('\r\n') for line in existing]
            expected_clean = [line.rstrip('\r\n') for line in expected]
            if expected and existing_clean != expected_clean:
                applied.append(f"SKIPPED {path} hunk at line {hunk['start']} (context mismatch)")
                continue

            content[start_idx:start_idx + old_count] = new_lines

        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "w") as f:
            f.writelines(content)

        action = "Added" if op == "add" else "Patched"
        applied.append(f"{action} {path}")

    return {"success": True, "output": "\n".join(applied), "error": None}


def _tool_handle_apply_patch(self, args):
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
    "read_file": {"file_path": ["file", "path", "filename", "filepath"]},
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

    def __init__(self, ctx=None, executor=None):
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
        lines = ["## Available tools"]
        for name, defn in AGENTIC_TOOL_DEFS.items():
            params = defn["parameters"]["properties"]
            args_str = ", ".join(f"{n}: {d['description']}" for n, d in params.items())
            reqs = defn["parameters"].get("required", [])
            required_str = f" (required: {', '.join(reqs)})" if reqs else ""
            lines.append(f"- {name}: {defn['description']} Arguments: {args_str}{required_str}")
        return "\n".join(lines)

    def list_tools_str(self) -> str:
        lines = []
        for name, defn in AGENTIC_TOOL_DEFS.items():
            destructive = "! " if name in DESTRUCTIVE_TOOLS else "  "
            lines.append(f"{destructive}{name:<16} {defn['description']}")
        return "\n".join(lines)

    def _confirm(self, tool_name: str, args: dict) -> bool:
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

    def __init__(self):
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

    def write(self, **data):
        data["timestamp"] = datetime.now().isoformat()
        self.file.write(json.dumps(data, default=str) + "\n")
        self.file.flush()

    def close(self):
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
    
    def __init__(self):
        # Each category stores its own level
        self._levels: Dict[str, int] = {cat: 0 for cat in self.CATEGORIES}
    
    def is_enabled(self, category: str) -> bool:
        """Quick check if any debugging is active for this category."""
        return self.get_level(category) > 0


    def set_level(self, category: str, level: str) -> bool:
        """Set debug level for a category. Returns True if valid."""
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
        """Check if a debug message should be emitted."""
        return self.get_level(category) >= min_level


    def get_status(self) -> dict:
        """Return current state for status display."""
        return {
            cat: level 
            for cat, level in self._levels.items() 
            if level > 0 or cat == 'all'
        }

# ============= DEBUG LOG function ============================================



def debug_log(debug_mgr, category: str, level: int, message: str, 
              data=None, prefix: str = "DEBUG"):
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

    def __init__(self, base_url=None, backend=None, context=None):
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

    def _debug_request(self, url: str, payload: dict, headers: dict = None):
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
 
    def _debug_response_chunk(self, chunk: dict, is_first: bool = False, is_final: bool = False, *args, **kwargs):
        """Log incoming streaming chunk."""
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
       
    def _debug_final_stats(self, usage_stats: dict):
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
        """Replace large binary data with size indicators for logging."""
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
    def base_url(self):
        return self.ctx.base_url
    
    @property
    def backend(self):
        return self.ctx.backend

    def estimate_tokens(self, text):
        """Estimate token count. Delegates to CommandContext."""
        return self.ctx.estimate_tokens(text)

    def calculate_context_tokens(self, messages):
        """Calculate estimated total tokens in conversation context. Delegates to CommandContext."""
        return self.ctx.calculate_context_tokens(messages)

    def calculate_stats(self, total_time, content, usage=None, messages=None):
        """Calculate stats for current query AND update cumulative totals in context."""
        eval_count = 0
        prompt_tokens = 0
        total_context_tokens = 0
        
        if usage:
            eval_count = usage.get("completion_tokens", 0) or usage.get("eval_count", 0)
            prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("prompt_eval_count", 0)
            total_context_tokens = usage.get("total_tokens", 0) or (prompt_tokens + eval_count)
        
        if not total_context_tokens and messages:
            total_context_tokens = self.calculate_context_tokens(messages)

        if not eval_count and content:
            eval_count = len(content.split())
    
        tps = eval_count / total_time if eval_count > 0 and total_time > 0 else 0.0
        
        current_stats = {
            "eval_count": eval_count,
            "prompt_eval_count": prompt_tokens,
            "total_context_tokens": total_context_tokens,
            "total_time": total_time,
            "tps": tps,
            "content_length": len(content)
        }

        return current_stats
    

    def print_stats_display(self, stats):
        """Print formatted stats to stderr."""
        if not stats:
            return
            
        parts = [f"{stats['total_time']:.2f}s total"]
        
        if stats.get("eval_count", 0) > 0:
            parts.append(f"{stats['tps']:.2f} t/s")
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
    def _inject_images_into_messages(messages, images, backend):
        """Inject image data into the last user message, mutating in-place.

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

    def build_request_payload(self, messages, model, stream_enabled=False, **kwargs):
        """Build request payload for the backend."""
        self._inject_images_into_messages(messages, kwargs.get('images'), self.backend)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream_enabled
        }
        
        if kwargs.get('is_warmup'):
            if self.backend == "ollama":
                payload["options"] = {"num_predict": 1}
            elif self.backend in ("llamacpp", "lmstudio"):
                payload["max_tokens"] = 1
        elif context_size := kwargs.get('context_size'):
            if self.backend == "ollama":
                payload["options"] = {"num_ctx": context_size}
            elif self.backend in ("llamacpp", "lmstudio"):
                payload["max_tokens"] = context_size
        
        if tools := kwargs.get('tools'):
            payload["tools"] = tools
        
        # Pass inference params for backend
        if self.backend == "llamacpp":
            for param in ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty"]:
                if param in kwargs:
                    payload[param] = kwargs[param]
        elif self.backend == "lmstudio":
            for param in ["temperature", "top_p", "presence_penalty", "repeat_penalty"]:
                if param in kwargs:
                    payload[param] = kwargs[param]
        
        return payload

    def _get_chat_url(self, backend):
        """Return the chat API URL for the given backend."""
        return f"{self.base_url}/api/chat" if backend == "ollama" else f"{self.base_url}/v1/chat/completions"

    def query_sync(self, messages, model, stream_enabled=False, **kwargs):
        """Non-streaming sync query wrapper."""
        payload = self.build_request_payload(messages, model, stream_enabled=False, **kwargs)

        try:
            url = self._get_chat_url(self.backend)

            data = json.dumps(payload).encode('utf-8')
            req = Request(url, data=data, headers={'Content-Type': 'application/json'})
            self._debug_request(url, payload)


            with _request_with_retry(req) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            msg = f"[ERROR] Sync query failed: {e}"
            if isinstance(e, HTTPError) and e.code == 403 and ":cloud" in model:
                msg += "\n[HINT] Cloud models require authentication. Check your Ollama cloud API key or pull a local model instead."
            sys.stderr.write(colorize(msg, 'error'))
            return {}




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
        
        # Format 3: Root-level fields (some versions)
        if "prompt_eval_count" in chunk:
            usage["prompt_tokens"] = usage.get("prompt_tokens", 0) or chunk.get("prompt_eval_count", 0)
        if "eval_count" in chunk:
            usage["completion_tokens"] = usage.get("completion_tokens", 0) or chunk.get("eval_count", 0)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        
        return usage if usage else None

    def _build_stream_request(self, backend, messages, model, stream_enabled, context_size, **kwargs):
        """Build URL, payload and headers for the backend."""
        payload = self.build_request_payload(messages, model, stream_enabled=stream_enabled, context_size=context_size, **kwargs)
        return self._get_chat_url(backend), payload, {'Content-Type': 'application/json'}

    def _parse_chunk(self, chunk, backend):
        """Extract (thought, content, is_final, usage, tool_calls) from a chunk for any backend."""
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
                    }
            return thought, content, is_final, usage, tool_calls
        elif backend == "llamacpp":
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

    def _iter_stream_lines(self, response, backend):
        """Yield decoded JSON lines from streaming response, stripping SSE prefixes."""
        for line in response:
            decoded = line.decode('utf-8').strip()
            if not decoded:
                continue
            if backend in ("llamacpp", "lmstudio"):
                if decoded.startswith('data: '):
                    decoded = decoded[6:].strip()
            if decoded == '[DONE]':
                continue
            yield decoded

    def _update_context_tokens(self, backend, aggregated_usage, messages):
        """Update context token tracking after stream completes."""
        if backend == "ollama":
            total_tokens = (aggregated_usage.get("total_tokens", 0) or
                           aggregated_usage.get("prompt_eval_count", 0) + aggregated_usage.get("eval_count", 0))
            if total_tokens > 0:
                self.ctx.current_context_tokens = total_tokens
                if self.ctx.debug_manager.is_enabled('context'):
                    debug_log(self.ctx.debug_manager, 'context', 1,
                             f"Updated context tokens: {total_tokens}", prefix="CTX")
        elif backend in ("llamacpp", "lmstudio"):
            if messages:
                self.ctx.current_context_tokens = self.ctx.calculate_context_tokens(messages)

    def query_stream(
        self,
        messages, model, stream_enabled=True, debug=False,
        show_thinking=True, context_size=None, images=None,
        tool_calls_out=None, **kwargs
    ):
        """Stream response from any backend. Dispatches to backend-specific chunk parsing.
        
        Args:
            tool_calls_out: Optional list to populate with accumulated tool_calls from streaming.
                For OpenAI-compatible streaming, incremental tool_calls are merged by index.
        """
        full_content = ""
        start_time = time.time()
        backend = self.backend

        if images and messages and messages[-1].get("role") == "user":
            messages[-1] = dict(messages[-1])
            self._inject_images_into_messages(messages, images, self.backend)

        api_url, payload, headers = self._build_stream_request(
            backend, messages, model, stream_enabled, context_size, **kwargs)

        data = json.dumps(payload).encode('utf-8')
        req = Request(api_url, data=data, headers=headers)
        self._debug_request(api_url, payload)

        start_thinking = False
        started_content = False
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

                        self._debug_response_chunk(chunk, first_chunk, is_final)
                        first_chunk = False

                        # Accumulate tool_calls from streaming chunks
                        if tool_calls:
                            for tc in tool_calls:
                                if "index" not in tc:
                                    stream_tool_calls.append({
                                        "id": tc.get("id", ""),
                                        "type": tc.get("type", "function"),
                                        "function": {
                                            "name": tc.get("function", {}).get("name", ""),
                                            "arguments": tc.get("function", {}).get("arguments", ""),
                                        }
                                    })
                                    continue
                                idx = tc.get("index", 0)
                                if idx not in stream_tool_call_index:
                                    stream_tool_call_index[idx] = {
                                        "id": tc.get("id", ""),
                                        "type": tc.get("type", "function"),
                                        "function": {
                                            "name": tc.get("function", {}).get("name", ""),
                                            "arguments": tc.get("function", {}).get("arguments", ""),
                                        }
                                    }
                                else:
                                    existing = stream_tool_call_index[idx]
                                    if tc.get("id"):
                                        existing["id"] = tc["id"]
                                    if tc.get("function", {}).get("name"):
                                        existing["function"]["name"] = tc["function"]["name"]
                                    if tc.get("function", {}).get("arguments"):
                                        existing["function"]["arguments"] += tc["function"]["arguments"]

                        if is_final and usage:
                            self._debug_final_stats(usage)

                        if usage:
                            aggregated_usage.update(usage)

                        if debug and is_final:
                            formatted_json = json.dumps(chunk, indent=4)
                            sys.stderr.write(colorize(f"\n[DEBUG] Final JSON chunk from server:\n{formatted_json}\n", 'muted'))

                        # Thinking display
                        if thought and show_thinking:
                            if not start_thinking:
                                start_thinking = True
                                sys.stderr.write("\n<thinking>\n")
                            sys.stderr.write(thought)
                            sys.stderr.flush()

                        # Content display
                        if content:
                            if start_thinking and not started_content:
                                sys.stderr.write("\n</thinking>\n")

                            if not started_content:
                                print("\n[--- Response ---]", file=sys.stdout)
                                started_content = True

                            sys.stdout.write(content)
                            sys.stdout.flush()
                            full_content += content

                    except json.JSONDecodeError:
                        continue

            # Finalize accumulated tool_calls
            if stream_tool_call_index:
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
            if tool_calls_out is not None and stream_tool_calls:
                tool_calls_out.extend(stream_tool_calls)

            if start_thinking and not started_content:
                sys.stderr.write("\n</thinking>\n")

            total_time = time.time() - start_time
            self._update_context_tokens(backend, aggregated_usage, messages)
            usage = self.calculate_stats(total_time, full_content, aggregated_usage, messages)
            self.ctx.update_stats(usage["eval_count"], usage["prompt_eval_count"], total_time, usage["content_length"])
            self.print_stats_display(usage)

            return full_content

        except Exception as e:
            msg = f"\n[ERROR] {backend} streaming failed: {e}"
            if isinstance(e, HTTPError) and e.code == 403 and ":cloud" in model:
                msg += "\n[HINT] Cloud models require authentication. Check your Ollama cloud API key or pull a local model instead."
            sys.stderr.write(colorize(f"{msg}\n", 'error'))
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

    def __init__(self, base_url, backend):
        self.base_url = base_url
        self.backend = backend
        self.commands = get_command_aliases()  # ✅ Use registry function
        self.models = []

    def fetch_models(self):
        """Fetch available models from the backend."""
        if self.backend == "llamacpp":
            self.models = [m['name'] for m in fetch_models_llamacpp(self.base_url)]
        else:
            self.models = [m['name'] for m in fetch_models_ollama(self.base_url)]


    def complete(self, text, state):
        """The core readline autocompletion hook."""
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
            if not dirname: dirname = '.'

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


def gather_user_input(prompt_prefix, show_multiline=True):
    """
    Gather user input with multiline support.
    it support three double quote
    it support backslash
    """
    ctrl_c_count = 0
    ctrl_d_count = 0

    while True:
        try:
            # Setup prompts
            if READLINE_AVAILABLE:
                prompt_str = colorize(f"{prompt_prefix} > ", 'warning',is_prompt=True)
                cont_prompt_str = colorize(f"... > ", 'warning', is_prompt=True)
            else:
                prompt_str = colorize(f"{prompt_prefix}: ", 'warning')
                cont_prompt_str = colorize(f"... : ", 'warning')

            line = input(prompt_str)
            ctrl_c_count = 0

            if not line.strip():
                return line

            # 1. Handle """ Block Multiline
            if show_multiline and line.strip() == '"""':
                lines = []
                while True:
                    try:
                        m_line = input(cont_prompt_str)
                        if m_line.strip() == '"""':
                            break
                        lines.append(m_line)
                    except KeyboardInterrupt:
                        print(colorize("\n[Multiline entry cancelled]", 'warning'), file=sys.stderr)
                        return None # Escape out of multiline without quitting
                return "\n".join(lines)

            # 2. Handle \ Line Continuation
            if line.endswith('\\'):
                lines = [line[:-1]]  # Strip the trailing backslash
                while True:
                    try:
                        m_line = input(cont_prompt_str)
                        if m_line.endswith('\\'):
                            lines.append(m_line[:-1])
                        else:
                            lines.append(m_line)
                            break
                    except KeyboardInterrupt:
                        print(colorize("\n[Multiline entry cancelled]", 'warning'), file=sys.stderr)
                        return None
                return "\n".join(lines)

            # 3. Standard Single Line
            return line

        except KeyboardInterrupt:
            ctrl_c_count += 1
            if ctrl_c_count >= 2:
                print(f"\n[Cancelled]", file=sys.stderr)
                return None
            print(f"\n(Press Ctrl+C again to exit)", file=sys.stderr)

        except EOFError:
            print(f"\n[EOF received, one more and it exits]", file=sys.stderr)
            ctrl_d_count += 1
            if ctrl_d_count >= 2:
                print(f"\n[Exiting]", file=sys.stderr)
                sys.exit(1)

def _process_file_inclusions(text):
    """Scan text for @filepath mentions and load referenced files."""
    inclusions = []
    for word in text.split():
        if word.startswith('@') and len(word) > 1:
            raw_path = word[1:]
            filepath = raw_path.rstrip('.,?!;:)"\'')
            expanded_path = os.path.expanduser(filepath)
            if os.path.isfile(expanded_path):
                file_size = os.path.getsize(expanded_path)
                if file_size > MAX_FILE_INCLUSION_SIZE:
                    err_msg = f"[Failed to load `{filepath}`: File too large ({file_size / 1024 / 1024:.1f} MB, max 5 MB)]"
                    print(colorize(err_msg, 'error'), file=sys.stderr)
                    continue
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

                    inclusions.append(f"\n[Content of local file `{filepath}`]:\n```text\n{file_content}\n```\n")
                except UnicodeDecodeError:
                    err_msg = f"[Failed to load `{filepath}`: Appears to be a binary or non-UTF-8 file]"
                    print(colorize(err_msg, 'error'), file=sys.stderr)
                    inclusions.append(err_msg + "\n")
                except Exception as e:
                    err_msg = f"[Failed to load `{filepath}`: {e}]"
                    print(colorize(err_msg, 'error'), file=sys.stderr)
                    inclusions.append(err_msg + "\n")
    return inclusions


def _process_command_lines(text):
    """Process lines starting with ! (shell) or /curl (URL fetch)."""
    processed = []
    for line in text.split('\n'):
        stripped = line.lstrip()

        if stripped.startswith("!"):
            command = stripped[1:].strip()
            if command and validate_shell_command_safety(command):
                output_str = execute_os_command(sanitize_shell_command(command))
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


def process_inline_commands(full_input):
    """Process inline commands (!, /curl, @) within user input."""
    file_inclusions = _process_file_inclusions(full_input)
    processed_lines = _process_command_lines(full_input)
    final_output = "\n".join(processed_lines)
    if file_inclusions:
        final_output += "\n" + "".join(file_inclusions)
    return final_output



def execute_os_command(command, timeout=None):
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
    if not validate_shell_command_safety(command, max_length=500):
        msg = "[Command rejected: Invalid characters]"
        print(colorize(msg, 'error'), file=sys.stderr)
        return "[Command rejected: Invalid characters]"

    print(f"[--- Executing (max {timeout}s): {command} ---]", file=sys.stderr)
    output_lines = []

    try:
        args_list = shlex.split(command)
        process = subprocess.run(
            args_list, shell=False,  # Intentional: shell=False prevents pipes/redirects but blocks injection. Single commands only.
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            timeout=timeout
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


def fetch_and_convert_url(url):
    """Fetch URL and extract clean text using core standard libraries only."""
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

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.skip_tags:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self.skip_tags:
            self.skip_depth = max(0, self.skip_depth - 1)

    def handle_data(self, data):
        if self.skip_depth == 0:
            cleaned = data.strip()
            if cleaned:
                self.text_parts.append(cleaned)

    def get_text(self):
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

    def __init__(self, context: CommandContext):
        """Initialize chat loop with shared context, completer, and query handler."""
        self.ctx = context
        self.completer = context.create_completer()
        self.query_handler = context.create_query_handler()
        self.executor = context.create_executor()
        self.tool_registry = context.create_tool_registry()
        self.commands = get_command_aliases()

    def handle_debug_command(self, args: str) -> None:
        """Process /debug commands."""
        parts = args.strip().split()
    
        if not parts or parts[0] == 'status':
            self._print_debug_status()
            return
    
        if parts[0] == 'list':
            self._print_debug_categories()
            return
    
        if len(parts) == 2:
            category, level = parts
            if self.ctx.debug_manager.set_level(category, level):
                print(colorize(f"Debug: {category} → {level}", 'success'), 
                    file=sys.stderr)
            else:
                print(colorize(f"Invalid category or level. Use '/debug list'", 
                              'error'), file=sys.stderr)
            return
    
    # Legacy fallback for old /debug on|off
        if len(parts) == 1 and parts[0] in ('on', 'off'):
            level = 'verbose' if parts[0] == 'on' else 'off'
            self.ctx.debug_manager.set_level('all', level)
            print(colorize(f"Debug: {parts[0]} (all categories)", 'success'), 
                file=sys.stderr)

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

                result = self.run_handle_exit(full_input)
                if result is True:
                    break
                if result is False:
                    continue

                result = self.run_handle_help(full_input)
                if result is False:
                    continue

                result = self.run_handle_stats(full_input)
                if result is False:
                    continue

                result = self.run_handle_listmodel(full_input)
                if result is False:
                    continue

                result = self.run_handle_context_size(full_input)
                if result is False:
                    continue

                result = self.run_handle_clear(full_input)
                if result is False:
                    continue

                result = self.run_handle_image(full_input)
                if result is False:
                    continue

                result = self.run_handle_dumpcontext(full_input)
                if result is False:
                    continue

                result = self.run_handle_debug(full_input)
                if result is False:
                    continue

                result = self.run_handle_thinking(full_input)
                if result is False:
                    continue

                result = self.run_handle_cwd(full_input)
                if result is False:
                    continue

                result = self.run_handle_ls(full_input)
                if result is False:
                    continue

                result = self.run_handle_switchmodel(full_input)
                if result is False:
                    continue

                result = self.run_handle_spawnshell(full_input)
                if result is False:
                    continue

                result = self.run_handle_agentic(full_input)
                if result is False:
                    continue

                result = self.run_handle_listtool(full_input)
                if result is False:
                    continue

                self.run_process_query(full_input)

            except KeyboardInterrupt:
                print(f"\n[Interrupted]", file=sys.stderr)
                continue

            except Exception as e:
                if isinstance(e, EOFError):
                    print(f"[EOF - Goodbye!]", file=sys.stderr)
                    break
                else:
                    print(colorize(f"[ERROR] ChatLoop->run {e}", 'error'), file=sys.stderr)
                    if self.ctx.debug_mode or self.ctx.debug_manager.get_level('all') > 0:
                        traceback.print_exc(file=sys.stderr)

        return

    def list_models(self, filter_arg: Optional[str] = None) -> None:
        """List available models from the backend, with optional name filter."""
        if self.ctx.backend == "ollama":
            list_models_ollama(self.ctx.base_url, filter_arg, file=sys.stdout)
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
                self.ctx.context_size = val
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

            def read_output(fd):
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

    def _parse_shell_into_blocks(self, text):
        """Split shell session into command blocks by detecting prompt lines."""
        text = strip_ansi(text)
        import re
        parts = re.split(r'(?m)^.*[\$#] ', text)
        return [p.strip() for p in parts if p.strip()]

    def _filter_smart_blocks(self, blocks):
        """Filter out trivial commands (cd, ls, pwd, clear, echo, exit)."""
        boring_commands = {'cd', 'ls', 'pwd', 'clear', 'exit', 'echo'}
        return [b for b in blocks if b.split('\n')[0].strip().split()[0] not in boring_commands if b.split() and len(b.strip()) > 20]

    def _edit_session(self, content):
        """Open content in editor (VISUAL > EDITOR > vim) and return the edited result."""
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

    def _handle_shell_session(self, session_output):
        """Parse captured shell session and let user choose what to send."""
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
    def _parse_number_ranges(text, max_val):
        """Parse '1,3-5,7' into [1, 3, 4, 5, 7]. Returns None on invalid input."""
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
        print(colorize("Type /help for details\n", 'muted'), file=sys.stderr)

        if images:
            self.ctx.current_images = images

        if READLINE_AVAILABLE:
            try:
                readline.set_completer_delims(' \t\n')
                self.completer.fetch_models()
                readline.set_completer(self.completer.complete)
                readline.parse_and_bind('tab: complete')

                histfile = os.path.expanduser("~/.ollamaquery.d/session")
                histdir = os.path.dirname(histfile)
                if not os.path.exists(histdir):
                    os.makedirs(histdir, exist_ok=True)

                try:
                    readline.read_history_file(histfile)
                except Exception:
                    pass

                readline.set_history_length(1000)

                def save_history():
                    if READLINE_AVAILABLE:
                        try:
                            readline.write_history_file(histfile)
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
            prompt_prefix += colorize("[img]", 'info', is_prompt=True)
        return prompt_prefix

    def run_handle_exit(self, full_input: str) -> Optional[bool]:
        """Handle exit/quit commands. Returns True to break loop."""
        if full_input.lower() in ['exit', 'quit', '/exit', '/quit']:
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
            print(colorize(f"\n[Usage Summary]", 'info'), file=sys.stderr)
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
                paths = shlex.split(parts[1].strip())
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

    def run_handle_debug(self, full_input: str) -> Optional[bool]:
        """Handle /debug command."""
        if full_input.startswith('/debug'):
            parts = full_input.split(maxsplit=2)

            if len(parts) == 1 or (len(parts) == 2 and not parts[1].strip()):
                print(colorize("\n[Debug Categories - Use /debug <category> <level>]", 'info'), file=sys.stderr)
                print("  Levels: off (0), basic (1), verbose (2), trace (3)", file=sys.stderr)
                print()
                for cat, desc in DebugManager.CATEGORIES.items():
                    current_level = self.ctx.debug_manager.get_level(cat)
                    level_name = {0: 'off', 1: 'basic', 2: 'verbose', 3: 'trace'}.get(current_level, str(current_level))
                    marker = '>' if current_level > 0 else ' '
                    print(f"  {marker} {cat:<12} [{level_name:<7}] {desc}", file=sys.stderr)
                print()

            elif len(parts) == 2:
                arg = parts[1].lower()

                if arg == 'list':
                    print(colorize("\n[Debug Categories]", 'info'), file=sys.stderr)
                    for cat, desc in DebugManager.CATEGORIES.items():
                        current_level = self.ctx.debug_manager.get_level(cat)
                        level_name = {0: 'off', 1: 'basic', 2: 'verbose', 3: 'trace'}.get(current_level, str(current_level))
                        marker = '>' if current_level > 0 else ' '
                        print(f"  {marker} {cat:<12} [{level_name}]", file=sys.stderr)
                    print()

                elif arg == 'status':
                    status = self.ctx.debug_manager.get_status()
                    if any(level > 0 for level in status.values() if isinstance(level, int)):
                        print(colorize("\n[Active Debug Categories]", 'info'), file=sys.stderr)
                        for cat, level in status.items():
                            if isinstance(level, int) and level > 0:
                                level_name = {1: 'basic', 2: 'verbose', 3: 'trace'}.get(level, str(level))
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
                        level_name = {0: 'off', 1: 'basic', 2: 'verbose', 3: 'trace'}.get(current, str(current))
                        desc = DebugManager.CATEGORIES[arg]
                        print(colorize(f"\n[Debug: {arg} = {level_name}] - {desc}", 'info'), file=sys.stderr)
                        print("Usage: /debug {} [off|basic|verbose|trace]\n".format(arg), file=sys.stderr)
                    else:
                        print(colorize(f"\n[Unknown category: '{arg}']", 'error'), file=sys.stderr)
                        print(colorize("Use '/debug' to see available categories\n", 'muted'), file=sys.stderr)

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

        # on/off -> explicit toggle
        if subcmd in ("on", "off"):
            target = subcmd == "on"
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

        # full -> enable everything
        if subcmd == "full":
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

        # Named toggles (always toggle between on/off)
        toggle_map = {
            "auto":     ("auto_confirm",        "Auto-confirm"),
            "verbose":  ("agentic_verbose",     "Verbose"),
            "thinking": ("agentic_show_thinking", "Show thinking"),
            "trace":    ("agentic_trace",        "Trace"),
            "log":      ("agentic_logging",      "Logging"),
            "lazytool": ("lazy_tool",            "Lazy tool extraction"),
        }

        if subcmd == "sandbox":
            new_mode = "container" if self.executor.mode != "container" else "host"
            self.executor.mode = new_mode
            self.tool_registry.executor.mode = new_mode
            print(colorize(f"[Executor mode: {self.executor.mode}]", 'info'), file=sys.stderr)

        elif subcmd in ("iterations", "timeout"):
            if len(parts) < 3 or not parts[2].isdigit():
                print(colorize(f"[Usage: /agentic {subcmd} <number>]", 'warning'), file=sys.stderr)
                return False
            val = int(parts[2])
            attr = "agentic_max_iterations" if subcmd == "iterations" else "agentic_step_timeout"
            setattr(self.ctx, attr, val)
            label = "Max iterations" if subcmd == "iterations" else "Step timeout"
            print(colorize(f"[{label}: {val}]", 'info'), file=sys.stderr)

        elif subcmd in toggle_map:
            attr, label = toggle_map[subcmd]
            new_val = not getattr(self.ctx, attr)
            setattr(self.ctx, attr, new_val)
            state = "ON" if new_val else "OFF"
            print(colorize(f"[{label}: {state}]", 'info'), file=sys.stderr)

        else:
            print(colorize("[Usage: /agentic [on|off|full|auto|sandbox|verbose|thinking|trace|log|lazytool|iterations <N>|timeout <N>|status]]", 'warning'), file=sys.stderr)
        return False

    def _print_agentic_status(self):
        """Display current agentic settings like /debug output."""
        c = self.ctx
        print(colorize("\n[Agentic Settings - Use /agentic <option> [value]]", 'info'), file=sys.stderr)
        print("  Subcommands: on, off, full, auto, sandbox, verbose, thinking, trace, log, lazytool,", file=sys.stderr)
        print("               iterations <N>, timeout <N>, status", file=sys.stderr)
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
    def _find_tool_call_brace(text, pos=0):
        """Find the next { that introduces a tool call JSON (with tool/function/type key)."""
        idx = text.find('{', pos)
        while idx != -1:
            rest = text[idx+1:].lstrip()
            if rest.startswith(('"tool"', '"function"', '"type"')):
                return idx
            idx = text.find('{', idx + 1)
        return -1

    @staticmethod
    def _rfind_tool_call_brace(text):
        """Find the last { that introduces a tool call JSON, scanning right-to-left."""
        idx = text.rfind('{')
        while idx != -1:
            rest = text[idx+1:].lstrip()
            if rest.startswith(('"tool"', '"function"', '"type"')):
                return idx
            if idx == 0:
                break
            idx = text.rfind('{', 0, idx)
        return -1

    def _extract_json_balanced(self, text, start):
        """Extract balanced JSON from text starting at an opening brace.
        
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

    def parse_tool_call(self, text: str) -> Optional[dict]:
        """Extract tool call JSON from LLM response. Returns {"tool": ..., "arguments": ...} or None."""
        lazy = getattr(self.ctx, 'lazy_tool', False)
        text = text.strip()
        # Pass 1: full JSON parse
        result = self._normalize_tool_json(text)
        if result:
            return result
        # Pass 2: fenced JSON code block
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
                        if result:
                            args_str = json.dumps(result.get("arguments", {}))
                            if not ('<' in args_str and '>' in args_str):
                                return result
        # Pass 3: bare JSON object with "tool" key
        if lazy:
            brace_idx = ChatLoop._find_tool_call_brace(text)
            if brace_idx != -1:
                json_str = self._extract_json_balanced(text, brace_idx)
                if json_str:
                    result = self._normalize_tool_json(json_str)
                    if result:
                        args_str = json.dumps(result.get("arguments", {}))
                        if not ('<' in args_str and '>' in args_str):
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
            # Pass 4: bare JSON tool call at the END of the text (strict mode only)
            idx = ChatLoop._rfind_tool_call_brace(text)
            if idx >= 0:
                json_str = self._extract_json_balanced(text, idx)
                if json_str:
                    end = idx + len(json_str)
                    if not text[end:].strip():
                        prefix = text[:idx].strip()
                        if not prefix:
                            result = self._normalize_tool_json(json_str)
                            if result:
                                args_str = json.dumps(result.get("arguments", {}))
                                if not ('<' in args_str and '>' in args_str):
                                    return result
        return None

    def parse_tool_calls(self, text: str) -> list[dict]:
        """Extract ALL tool call JSONs from the response.
        
        First tries direct JSON parse (works for clean tool calls at start of text).
        Falls back to strict/lazy extraction for embedded or malformed JSON.
        
        In strict mode (default): only matches consecutive tool calls starting
        from the beginning of the response (no preamble).
        In lazy mode (/agentic lazytool): finds tool calls anywhere in the text.
        Returns list of {"tool": ..., "arguments": ...} dicts.
        """
        lazy = getattr(self.ctx, 'lazy_tool', False)
        text = text.strip()
        results = []
        
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
            # Lazy mode: find ALL tool call JSONs anywhere in the text
            seen_keys = set()
            pos = 0
            while pos < len(text):
                idx = ChatLoop._find_tool_call_brace(text, pos)
                if idx == -1:
                    break
                json_str = self._extract_json_balanced(text, idx)
                if not json_str:
                    break
                result = self._normalize_tool_json(json_str)
                if result:
                    key = (result["tool"], json.dumps(result.get("arguments", {}), sort_keys=True))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        results.append(result)
                pos = idx + len(json_str) if json_str else idx + 1
        else:
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

    @staticmethod
    def _is_stuck(text: str, threshold: float = 0.8) -> bool:
        """Detect if the model is repeating itself (stuck in a loop).
        
        Checks if the last ~500 chars show strong self-similarity.
        Returns True if the response appears stuck.
        """
        if len(text) < 200:
            return False
        tail = text[-500:]
        chunk_size = 50
        chunks = [tail[i:i + chunk_size] for i in range(0, len(tail) - chunk_size + 1, chunk_size)]
        if len(chunks) < 4:
            return False
        similar = 0
        for i in range(len(chunks) - 1):
            common = sum(1 for a, b in zip(chunks[i], chunks[i + 1]) if a == b)
            ratio = common / max(len(chunks[i]), len(chunks[i + 1]))
            if ratio > 0.85:
                similar += 1
        return similar / (len(chunks) - 1) > threshold

    @staticmethod
    def _call_with_timeout(func, timeout_sec: int, *args, **kwargs):
        """Call a function with a wall-clock timeout using a daemon thread."""
        result = [None]
        exception = [None]

        def worker():
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

    def _init_agentic_query(self, final_content):
        """Initialize messages, logger, and tool format for an agentic query.

        Returns (messages, logger, openai_tools, send_tools_api) or None on early exit.
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

    def _execute_tool_calls(self, tool_calls, last_tool_call, iteration, logger, messages, response_text, api_tool_calls):
        """Execute a list of tool calls, collecting observations.

        Returns (observations, abort_loop, last_tool_call, final_answer).
        """
        observations = []
        raw_observations = []
        abort_loop = False
        final_answer = ""
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call["tool"]
            tool_args = tool_call.get("arguments", {})
            args_display = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
            print(colorize(f"\n[Tool] {tool_name}({args_display})", 'warning'), file=sys.stderr, end="")
            sys.stderr.flush()

            if self.ctx.agentic_trace:
                print(colorize(f"\n[Trace] Full args: {json.dumps(tool_args, default=str)[:2000]}", 'muted'), file=sys.stderr)

            current_call = (tool_name, json.dumps(tool_args, sort_keys=True))
            if i == 0 and current_call == last_tool_call:
                print(colorize("[Agentic] Same tool call repeated, breaking loop.", 'warning'), file=sys.stderr)
                final_answer = "[Agentic: model stuck in tool loop]"
                abort_loop = True
                break
            if i == 0:
                last_tool_call = current_call

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
            if len(observation) > 4000:
                observation = observation[:4000] + "\n... [truncated]"

            timed_observation = json.dumps({"tool": tool_name, "duration_s": round(elapsed, 1), "success": result["success"], "output": observation})
            if self.ctx.agentic_trace:
                trace_out = result["output"] if len(result["output"]) < 2000 else result["output"][:2000] + "..."
                print(colorize(f"[Trace] Result: {json.dumps({'success': result['success'], 'output': trace_out, 'error': result['error']}, default=str)}", 'muted'), file=sys.stderr)

            if self.ctx.agentic_logging and logger:
                logger.write(type="result", iteration=iteration, tool_name=tool_name, tool_args=tool_args, result=result)

            if not result["success"] and "Cancelled" in (result.get("error") or ""):
                print(colorize("[Agentic] Tool cancelled by user, aborting.", 'warning'), file=sys.stderr)
                final_answer = "[Agentic query cancelled]"
                abort_loop = True
                break

            observations.append(f"[{tool_name}] {timed_observation}")
            raw_observations.append(observation)

        return observations, raw_observations, abort_loop, last_tool_call, final_answer

    def _finalize_agentic_query(self, messages, final_answer, final_content, send_tools_api, openai_tools, logger, iteration, response_text):
        """Stream final answer or execute pending tool call, then update self.messages."""
        if not final_answer:
            final_answer = response_text if response_text else "[Agentic: no answer produced]"

        if logger:
            logger.write(type="final", final_answer=final_answer)

        pending_tool = self.parse_tool_call(final_answer)
        if pending_tool:
            tool_name = pending_tool["tool"]
            tool_args = pending_tool.get("arguments", {})
            args_display = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
            print(colorize(f"\n[Tool] {tool_name}({args_display})", 'warning'), file=sys.stderr)
            result = self.tool_registry.execute(tool_name, tool_args)
            observation = result["output"] if result["success"] else f"ERROR: {result['error']}"
            if not observation:
                observation = "[Tool returned no output]"
            if len(observation) > 4000:
                observation = observation[:4000] + "\n... [truncated]"
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
                **stream_kwargs
            )
            if response or stream_tool_calls_out:
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
                if stream_tool_calls:
                    stream_observations = []
                    for stream_tc in stream_tool_calls:
                        tool_name = stream_tc["tool"]
                        tool_args = stream_tc.get("arguments", {})
                        args_display = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
                        print(colorize(f"\n[Tool] {tool_name}({args_display})", 'warning'), file=sys.stderr, end="")
                        sys.stderr.flush()
                        t_start = time.time()
                        result = self.tool_registry.execute(tool_name, tool_args)
                        elapsed = time.time() - t_start
                        status = "OK" if result["success"] else "ERROR"
                        print(colorize(f" → {status} ({elapsed:.1f}s)", 'info' if result["success"] else 'error'), file=sys.stderr)
                        observation = result["output"] if result["success"] else f"ERROR: {result['error']}"
                        if not observation:
                            observation = "[Tool returned no output]"
                        if len(observation) > 4000:
                            observation = observation[:4000] + "\n... [truncated]"
                        print(colorize(f"\n{observation}", 'info'), file=sys.stdout)
                        stream_observations.append(f"[{tool_name}] {observation}")
                    if stream_tool_calls_out:
                        stream_content = response or ""
                        if not stream_content:
                            inline_parts = []
                            for stc in stream_tool_calls_out:
                                fn = stc.get("function", {})
                                inline_parts.append(json.dumps({
                                    "tool": fn.get("name", ""),
                                    "arguments": fn.get("arguments", "{}")
                                }))
                            stream_content = "\n".join(inline_parts)
                        assistant_msg = {'role': 'assistant', 'content': stream_content}
                        assistant_msg['tool_calls'] = stream_tool_calls_out
                        self.messages.append(assistant_msg)
                    else:
                        self.messages.append({'role': 'assistant', 'content': response})
                    if stream_observations:
                        self.messages.append({'role': 'user', 'content': "Tool result:\n" + "\n---\n".join(stream_observations)})
                    print()
                else:
                    self.messages.append({'role': 'assistant', 'content': response})
                    print()
            else:
                print(colorize(f"\n{final_answer}", 'success'), file=sys.stdout)

    def run_agentic_query(self, full_input: str) -> None:
        """ReAct loop: query model, parse tool calls, execute tools, stream final answer."""
        logger = None
        try:
            final_content = process_inline_commands(full_input)
            if not final_content.strip():
                return

            if not getattr(self, 'messages', None):
                self.messages = [{'role': 'system', 'content': self.ctx.system_prompt}]
            self.messages.append({'role': 'user', 'content': final_content})

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

            while iteration < max_iterations:
                iteration += 1
                if logger:
                    logger.write(type="iteration", iteration=iteration)
                print(colorize(f"\r[Agentic] Step {iteration}/{max_iterations}…", 'muted'), file=sys.stderr, end="")
                sys.stderr.flush()

                images_to_send = [] if self.ctx.supports_vision is False else self.ctx.current_images
                sync_kwargs = dict(get_inference_params(self.ctx.model))
                if send_tools_api:
                    sync_kwargs["tools"] = openai_tools
                response = self._call_with_timeout(
                    self.query_handler.query_sync, step_timeout,
                    messages, self.ctx.model,
                    context_size=self.ctx.context_size,
                    images=images_to_send,
                    **sync_kwargs
                )

                if response is None:
                    print(colorize(f"\n[Agentic] Step timed out after {step_timeout}s.", 'error'), file=sys.stderr)
                    break

                if images_to_send and self.ctx.supports_vision is not False:
                    if check_vision_error(response):
                        self.ctx.supports_vision = False
                        print(colorize("\n[WARNING] Model does not support vision. Stripping images for subsequent queries.", 'warning'), file=sys.stderr)
                        continue

                if send_tools_api and check_tools_error(response):
                    send_tools_api = False
                    include_tool_defs = True
                    tool_defs_block = self.tool_registry.get_system_prompt_block()
                    messages[0] = {'role': 'system', 'content': get_agentic_prompt(self.ctx.model, tool_defs_block, include_tool_defs=True)}
                    print(colorize("\n[WARNING] Model does not support native tools API. Falling back to inline tool definitions.", 'warning'), file=sys.stderr)
                    continue

                response_text = ""
                api_tool_calls = []
                if isinstance(response, dict):
                    if self.ctx.backend == "ollama":
                        msg = response.get('message', {})
                        response_text = msg.get('content', '')
                        api_tool_calls = msg.get('tool_calls', [])
                    else:
                        choices = response.get('choices', [])
                        if choices:
                            msg = choices[0].get('message', {})
                            response_text = msg.get('content', '') or ''
                            api_tool_calls = msg.get('tool_calls', [])
                elif isinstance(response, str):
                    response_text = response

                if self.ctx.agentic_verbose and response_text:
                    truncated = len(response_text) > 500
                    display = response_text[:500] + ("..." if truncated else "")
                    print(colorize(f"\n[Verbose] {display}", 'muted'), file=sys.stderr)
                    if truncated:
                        print(colorize(f"[Verbose] ({len(response_text)} total chars, showing first 500)", 'muted'), file=sys.stderr)

                if self.ctx.agentic_show_thinking and isinstance(response, dict):
                    thinking = ""
                    if self.ctx.backend == "ollama":
                        thinking = response.get('message', {}).get('reasoning_content', '')
                    else:
                        choices = response.get('choices', [])
                        if choices:
                            msg = choices[0].get('message', {})
                            thinking = msg.get('reasoning_content', '') or msg.get('reasoning', '')
                    if thinking:
                        print(colorize(f"\n<thinking>\n{thinking}\n</thinking>", 'muted'), file=sys.stderr)

                if not response_text and not api_tool_calls:
                    messages.append({'role': 'user', 'content': 'Please provide a tool call or your final answer.'})
                    continue

                if response_text and self._is_stuck(response_text):
                    print(colorize("\n[Agentic] Model appears stuck (repetitive output), aborting.", 'warning'), file=sys.stderr)
                    break

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

                if logger:
                    first_call = tool_calls[0] if tool_calls else None
                    logger.write(type="turn", iteration=iteration, model_response=response_text, tool_call=first_call)

                if not tool_calls:
                    final_answer = response_text
                    break

                observations, raw_observations, abort_loop, last_tool_call, tool_final = self._execute_tool_calls(
                    tool_calls, last_tool_call, iteration, logger, messages, response_text, api_tool_calls
                )
                if tool_final:
                    final_answer = tool_final
                if abort_loop:
                    break

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

            if iteration >= max_iterations and not final_answer:
                final_answer = response_text

            self._finalize_agentic_query(messages, final_answer, final_content, send_tools_api, openai_tools, logger, iteration, response_text)

            if logger:
                logger.write(type="end", total_iterations=iteration)
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

    def run_process_query(self, full_input: str) -> None:
        """Process regular user query (non-command input)."""
        if self.ctx.agentic_mode:
            return self.run_agentic_query(full_input)

        final_content = process_inline_commands(full_input)
        if not final_content.strip():
            return

        if not hasattr(self, 'messages'):
            self.messages = [{'role': 'system', 'content': self.ctx.system_prompt}]
        self.messages.append({'role': 'user', 'content': final_content})

        payload_messages = list(self.messages)

        if self.ctx.force_no_thinking:
            payload_messages.append({
                'role': 'system',
                'content': 'Do NOT output reasoning or thoughts'
            })

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
                images=([] if self.ctx.supports_vision is False else self.ctx.current_images)
            )

            if response:
                self.messages.append({'role': 'assistant', 'content': response})
                print()


# ============================================================================
# ============= MAIN ENTRY POINT ===========================================
# ============================================================================

def get_base_url(args, backend):
    """Get base URL for the specified backend."""
    if args.host:
        base_url = args.host
    else:
        default = DEFAULT_LLAMACPP_HOST if backend in ("llamacpp", "lmstudio") else DEFAULT_OLLAMA_HOST
        env_var = f'{backend.upper()}_HOST'
        base_url = os.environ.get(env_var, default)

    # Ensure URL prefix
    if not base_url.startswith(('http://', 'https://')):
        base_url = f"http://{base_url}"

    return base_url


def list_models_llamacpp(base_url, filter_arg=None):
    """List Llama.cpp models."""
    models = fetch_models_llamacpp(base_url)

    if not models:
        print(f"\n[No models found via llamacpp API]", file=sys.stderr)
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


def list_models_ollama(base_url, filter_arg=None, include_capabilities=False, file=None):
    if file is None:
        file = sys.stdout

    models = fetch_models_ollama(base_url)
    if not models:
        print(colorize(f"\nNo models found via Ollama API at {base_url}. Check if the server is running.\n", 'warning'), file=file)
        return

    sort_by = 'name'
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
        if not models:
            print(colorize(f"\nNo models found matching '{search_term}'.\n", 'warning'), file=file)
            return

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

    largest = max(models, key=lambda x: x['size_bytes']) if models else None
    if largest and largest['size_bytes'] > 0:
        l_size_gb = largest['size_bytes'] / (1024**3)
        print(f"\nChecking storage... Largest model in list: {largest['name']} ({l_size_gb:.2f} GB)\n", file=file)
    else:
        print(file=file)

    if include_capabilities:
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
    else:
        header = f"{'NAME':<40} | {'SIZE':<12} | {'MODIFIED'}"
        print(colorize(header, 'muted'), file=file)
        print(colorize("-" * len(header), 'muted'), file=file)
        for m in models:
            size_str = parse_size(m.get('size') or m.get('size_bytes') or m.get('model_size') or 0)
            modified = m.get('modified_at', 'Unknown')[:10]
            print(f"{m['name']:<40} | {size_str:<12} | {modified}", file=file)
    print(file=file)
    print()


def show_model_info(base_url, model, args):
    """Display model information."""
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


def show_model_details(base_url, model, args):
    """Display full model details."""
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


def fetch_loaded_models_ollama(base_url):
    """Fetch models currently loaded in memory via Ollama /api/ps."""
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
            return [
                (m["name"], m.get("context_size", 0))
                for m in data.get("models", [])
                if isinstance(m, dict) and "name" in m
            ]
    except Exception:                     # network error, parsing error, etc.
        sys.stderr.write(colorize("[ERROR] Could not read Ollama model info", "error"))
        return []


def check_backend_with_head(url, server_marker):
    """Attempt HEAD request to URL and check for server header."""
    try:
        request = Request(url, method='HEAD')
        with urlopen(request, timeout=1) as response:  # startup-probe
            server_header = response.headers.get('Server', '').lower()
            return server_marker.lower() in server_header
    except Exception:
        return False


def check_backend_with_get(url, server_marker):
    """Attempt GET request to URL and check for server marker."""
    try:
        request = Request(url, method='GET')
        with urlopen(request, timeout=1) as response:  # startup-probe
            my_response = str(response.read())
            my_response = my_response.lower()
            if server_marker.lower() in my_response:
                return True
    except Exception:
        return False



def check_lmstudio(url):
    """Check if LM Studio is running by querying /v1/models."""
    try:
        request = Request(f"{url}/v1/models", method='GET')
        with urlopen(request, timeout=2) as response:  # startup-probe
            data = json.loads(response.read().decode('utf-8'))
            models = data.get('data', [])
            return bool(models)
    except Exception:
        return False

def auto_detect_backend():
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

    sys.stderr.write(colorize(f"[INFO] AutoDetecting on : " + llama_cpp_url + " ", 'info'))
    if check_backend_with_head(llama_cpp_url, 'llama.cpp'):
        sys.stderr.write(colorize(f"Success\n", 'info'))
        return True,'llamacpp',llama_cpp_url
    else:
        sys.stderr.write(colorize(f"Fail\n", 'info'))

    sys.stderr.write(colorize(f"[INFO] AutoDetecting on : " + ollama_url    + " ", 'info'))
    if check_backend_with_get(ollama_url,     'ollama'):
        sys.stderr.write(colorize(f"Success\n", 'info'))
        return True,'ollama',ollama_url
    else:
        sys.stderr.write(colorize(f"Fail\n", 'info'))

    sys.stderr.write(colorize(f"[INFO] AutoDetecting on : " + lmstudio_url + " ", 'info'))
    if check_lmstudio(lmstudio_url):
        sys.stderr.write(colorize(f"Success\n", 'info'))
        return True,'lmstudio',lmstudio_url
    else:
        sys.stderr.write(colorize(f"Fail\n", 'info'))


    # grab the ip of the host 
    try:
        list_of_ip = socket.gethostbyname_ex(socket.gethostname())[-1]
    except socket.error:
        list_of_ip = []
    for ip in list_of_ip:
        
        url="http://"+ip + ":" + str(DEFAULT_LLAMACPP_PORT)
        sys.stderr.write(colorize(f"[INFO] AutoDetecting on : " + url    + " ", 'info'))
        if check_backend_with_head(url, 'llama.cpp'):
            sys.stderr.write(colorize(f"Success\n", 'info'))
            return True,'llamacpp',url
        else:
            sys.stderr.write(colorize(f"Fail\n", 'info'))

        url="http://"+ip + ":" + str(DEFAULT_OLLAMA_PORT)
        sys.stderr.write(colorize(f"[INFO] AutoDetecting on : " + url    + " ", 'info'))
        if check_backend_with_get("http://"+ip + ":" +  str(DEFAULT_OLLAMA_PORT),   'ollama'):
            sys.stderr.write(colorize(f"Success\n", 'info'))
            return True,'ollama',url
        else:
            sys.stderr.write(colorize(f"Fail\n", 'info'))

        url="http://"+ip + ":" + str(DEFAULT_LMSTUDIO_PORT)
        sys.stderr.write(colorize(f"[INFO] AutoDetecting on : " + url    + " ", 'info'))
        if check_lmstudio(url):
            sys.stderr.write(colorize(f"Success\n", 'info'))
            return True,'lmstudio',url
        else:
            sys.stderr.write(colorize(f"Fail\n", 'info'))

    return None,'',''

def load_saved_backends():
    """Load the list of previously successful backend configurations."""
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

def save_backend_config(backend, host):
    """Save a successful connection to the top of the history list."""
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

def resolve_connection(args):
    """
    Determines the correct backend and host by prioritizing:
    1. Explicit CLI overrides (-H and -b)
    2. Previously working configurations (tried Most Recently Used first)
    3. Network Auto-discovery
    4. Hardcoded defaults
    """
    # 1. Explicit user override (-H)
    if args.host:
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
        return selected_backend, base_url

    saved_backends = load_saved_backends()
    
    # 2. Iterate through history
    for config in saved_backends:
        s_backend = config.get('backend')
        s_host = config.get('host')
        
        # If user explicitly passed `-b`, skip history entries that don't match
        if args.backend and args.backend != s_backend:
            continue

        sys.stderr.write(colorize(f"[INFO] Testing known server: {s_backend} @ {s_host} ... ", 'muted'))
        
        is_valid = False
        if s_backend == 'llamacpp':
            is_valid = check_backend_with_head(s_host, 'llama.cpp')
        elif s_backend == 'ollama':
            is_valid = check_backend_with_get(s_host, 'ollama')
        elif s_backend == 'lmstudio':
            is_valid = check_lmstudio(s_host)

        if is_valid:
            sys.stderr.write(colorize("Success\n", 'success'))
            save_backend_config(s_backend, s_host) # Bump to top of list
            return s_backend, s_host
        else:
            sys.stderr.write(colorize("Offline\n", 'warning'))

    # 3. If history failed or is empty, trigger Auto-Discovery
    sys.stderr.write(colorize(f"\n[INFO] Known servers offline. Initiating auto-discovery...\n", 'info'))
    autodetected, d_backend, d_url = auto_detect_backend()
    if autodetected:
        # If user explicitly passed `-b`, ensure the autodetected backend matches
        if not args.backend or args.backend == d_backend:
            save_backend_config(d_backend, d_url)
            return d_backend, d_url

    # 4. Ultimate Fallback
    sys.stderr.write(colorize(f"[WARNING] Auto-discovery failed. Falling back to defaults.\n", 'error'))
    fallback_backend = args.backend or "ollama"
    if fallback_backend == "llamacpp":
        fallback_host = os.environ.get('LLAMACPP_HOST', DEFAULT_LLAMACPP_HOST)
    elif fallback_backend == "lmstudio":
        fallback_host = os.environ.get('LMSTUDIO_HOST', DEFAULT_LMSTUDIO_HOST)
    else:
        fallback_host = os.environ.get('OLLAMA_HOST', DEFAULT_OLLAMA_HOST)

    return fallback_backend, fallback_host


# ============================================================================
# ============= ARGUMENT PARSER ==============================================
# ============================================================================

def _build_parser():
    """Build and return the argument parser with all options."""
    parser = argparse.ArgumentParser(
        description="Unified LLM Query Interface for Ollama, Llama.cpp & LM Studio"
    )

    parser.add_argument_group('Backend')
    parser.add_argument("-b", "--backend", choices=["ollama", "llamacpp", "lmstudio"], default=None, help="API backend to use (auto-detected if omitted).")
    parser.add_argument('-H', '--host', help='Custom API URL')
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


def _verify_server(backend, base_url, args):
    """Probe the server to confirm it's reachable, trying default ports if needed.

    Returns (backend, base_url) — may fall back to a different backend/port
    if the initial guess was wrong. Exits with code 1 if unreachable.
    """
    server_reachable = False
    if backend == "ollama":
        server_reachable = check_backend_with_get(base_url, 'ollama')
    elif backend == "llamacpp":
        server_reachable = check_backend_with_head(base_url, 'llama.cpp')
    elif backend == "lmstudio":
        server_reachable = check_lmstudio(base_url)

    if not server_reachable and not re.search(r':\d{2,5}(/|$)', base_url):
        port_map = {"ollama": DEFAULT_OLLAMA_PORT, "llamacpp": DEFAULT_LLAMACPP_PORT, "lmstudio": DEFAULT_LMSTUDIO_PORT}

        if args.backend:
            port = port_map.get(backend, DEFAULT_OLLAMA_PORT)
            fallback = f"{base_url}:{port}"
            sys.stderr.write(colorize(f"[INFO] Checking {fallback}... ", 'muted'))
            ok = check_lmstudio(fallback) if backend == "lmstudio" else \
                 check_backend_with_head(fallback, 'llama.cpp') if backend == "llamacpp" else \
                 check_backend_with_get(fallback, 'ollama')
            if ok:
                sys.stderr.write(colorize(f"found {backend}\n", 'success'))
                backend, base_url = backend, fallback
                server_reachable = True
            else:
                sys.stderr.write(colorize("no\n", 'warning'))
        else:
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
                        backend, base_url = probe_backend, probe_url
                        server_reachable = True
                        break
                    sys.stderr.write(colorize("no\n", 'warning'))
                except Exception:
                    sys.stderr.write(colorize("no\n", 'warning'))
                    continue

    if not server_reachable:
        hint = ""
        has_port = re.search(r':\d{2,5}(/|$)', base_url)
        if backend == "llamacpp" and not has_port:
            hint = f" (try {base_url}:{DEFAULT_LLAMACPP_PORT})"
        elif backend == "lmstudio" and not has_port:
            hint = f" (try {base_url}:{DEFAULT_LMSTUDIO_PORT})"
        elif backend == "ollama" and not has_port:
            hint = f" (try {base_url}:{DEFAULT_OLLAMA_PORT})"
        sys.stderr.write(colorize(f"[ERROR] Cannot reach {backend} at {base_url}. Server may be offline.{hint}\n", 'error'))
        sys.exit(1)

    return backend, base_url


def _select_model(backend, base_url, args):
    """Select the target model, auto-detecting from server if -m not given.

    Returns model name or empty string if none available.
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

    available = fetch_models_llamacpp(base_url)
    if available:
        model = available[0]['name']
        label = "LM Studio" if backend == "lmstudio" else "Llama.cpp"
        sys.stderr.write(colorize(f"[INFO] Auto-selected hosted model: '{model}'\n", 'success'))
        return model

    label = "LM Studio" if backend == "lmstudio" else "Llama.cpp"
    print(colorize(f"\n[No models available on {backend} server. Use /listmodel to see available models.]", 'warning'), file=sys.stderr)
    return ""


def main():
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

    if args.prompt:
        active_prompt = args.prompt
    elif args.profile:
        active_prompt = BUILTIN_PROMPTS[args.profile]
    else:
        active_prompt = DEFAULT_SYSTEM_PROMPT
    args.prompt = active_prompt

    if not (args.chat or args.input_text or args.input_file or
             args.input_dir or args.list or args.show or args.show_details):
        print(colorize(f"ollamaquery2 v{__version__} - LLM Query Interface", 'info'))
        if args.backend or args.host:
            print(f"  Backend configured ({args.backend or 'auto'} @ {args.host or 'auto'}), but no action specified.")
        print(f"  Start chat:  -c")
        print(f"  Single query: -I \"your prompt\"")
        print(f"  List models: -l")
        print(f"  Help:        --help")
        sys.exit(2)

    backend, base_url = resolve_connection(args)
    backend, base_url = _verify_server(backend, base_url, args)
    save_backend_config(backend, base_url)
    target_model = _select_model(backend, base_url, args)

    if not target_model:
        sys.stderr.write(colorize(f"[INFO] Connected to {backend} at {base_url}\n", 'success'))
        if args.show or args.show_details:
            sys.stderr.write(colorize("[ERROR] No model selected. Use -m or --list to browse models.\n", 'error'))
            sys.exit(1)
        if args.input_text or args.input_file or args.input_dir:
            sys.stderr.write(colorize("[ERROR] No model selected. Use -m to specify a model.\n", 'error'))
            sys.exit(1)

    # Listing operations
    if args.list or args.list_all:
        if backend in ("llamacpp", "lmstudio"):
            list_models_llamacpp(base_url, filter_arg=args.model)
        elif args.list_all:
            list_models_ollama(base_url, filter_arg=args.model, include_capabilities=True)
        else:
            list_models_ollama(base_url, filter_arg=args.model, include_capabilities=False)
        sys.exit(0)

    # Model info operations
    if args.show:
        show_model_info(base_url, target_model, args)
    elif args.show_details:
        show_model_details(base_url, target_model, args)

    # Interactive chat mode
    if args.chat:
        ctx = CommandContext()
        ctx.base_url = base_url
        ctx.backend = backend
        ctx.model = target_model
        ctx.system_prompt = args.prompt
        ctx.shell_timeout = args.shell_timeout

        images_list = None
        if args.image:
            images_list = [prepare_image_data(p) for p in args.image if p and prepare_image_data(p)]
            if images_list:
                ctx.current_images = images_list

        should_stream = not args.no_stream and sys.stdout.isatty()
        loop = ChatLoop(ctx)
        loop.run(stream_enabled=should_stream, debug=args.debug, images=images_list)
        sys.exit(0)

    # Batch / single query processing
    if args.input_dir:
        if not args.output_dir:
            print("[ERROR] --output-dir required for --input-dir", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
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
            images_list = [prepare_image_data(p) for p in args.image if p and prepare_image_data(p)] if args.image else None
            qh = ModelQuery(context=CommandContext())
            qh.ctx.base_url = base_url
            qh.ctx.backend = backend
            qh.ctx.shell_timeout = args.shell_timeout
            response = qh.query_sync(messages, target_model, context_size=None, show_thinking=True, debug=args.debug, images=images_list)
            output_text = ""
            if isinstance(response, dict):
                output_text = response.get('message', {}).get('content', '')
                if not output_text:
                    choices = response.get('choices', [])
                    if choices:
                        output_text = choices[0].get('message', {}).get('content', '')
            else:
                output_text = str(response)
            with open(os.path.join(args.output_dir, filename + '.output'), 'w', encoding='utf-8') as f:
                f.write(output_text)
        sys.exit(0)

    if args.input_text or args.input_file:
        images_list = [prepare_image_data(p) for p in args.image if p and prepare_image_data(p)] if args.image else None
        messages = [{'role': 'system', 'content': args.prompt}]
        if args.input_text:
            messages.append({'role': 'user', 'content': args.input_text})
        elif args.input_file and os.path.isfile(args.input_file):
            with open(args.input_file, 'r', encoding='utf-8') as f:
                messages.append({'role': 'user', 'content': f.read()})

        qh = ModelQuery(context=CommandContext())
        qh.ctx.base_url = base_url
        qh.ctx.backend = backend
        qh.ctx.shell_timeout = args.shell_timeout
        should_stream = not args.no_stream and sys.stdout.isatty()

        response = qh.query_sync(messages, target_model, context_size=None, show_thinking=True, debug=args.debug, images=images_list)

        if args.output:
            content = ""
            if isinstance(response, dict):
                content = response.get('message', {}).get('content', '')
                if not content:
                    choices = response.get('choices', [])
                    if choices:
                        content = choices[0].get('message', {}).get('content', '')
            else:
                content = str(response)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[Success: Output saved to {args.output}]", file=sys.stderr)
        elif response:
            content = ""
            thinking = ""
            if isinstance(response, dict):
                msg = response.get('message', {}) if backend == "ollama" else (response.get('choices', [{}])[0].get('message', {}) if response.get('choices') else {})
                content = msg.get('content', '')
                if not content and isinstance(response, dict):
                    content = response.get('message', {}).get('content', '')
                thinking = msg.get('reasoning_content', '') or msg.get('thought', '') or msg.get('thinking', '')
            else:
                content = str(response)
            if thinking:
                sys.stderr.write(f"\n<thinking>\n{thinking}\n</thinking>\n")
            print(content)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\n[Exiting gracefully...]")
        sys.exit(0)


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
import html
import base64
import argparse
import subprocess
import shutil
import threading
import time

from html.parser import HTMLParser
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin, urlparse
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


# ============================================================================
# ============= DEFAULT CONFIGURATION =======================================
# ============================================================================

BUILTIN_PROMPTS = {
    "default": (
        "You are a chatbot trying to help user. Try to rebound to the question as best "
        "as your knowledge goes but reply politely that you don't know if it is the case."
    ),
    "coder": (
        "You are an AI chatbot specialist in Python and C++ coding, as well as system and "
        "software engineering. You can use emoji to emphasize titles. If you are not fully sure, "
        "ask for more information to guide me properly."
    ),
    "sysadmin": (
        "You are a Linux system administrator. Reply briefly to my questions. And don't "
        "think too much. If I give you an image, summarize it briefly."
    ),
    "concise": (
        "You are a highly efficient AI assistant. Provide direct, factual answers without "
        "filler, pleasantries, or unnecessary explanations."
    )
}

DEFAULT_SYSTEM_PROMPT = BUILTIN_PROMPTS["default"]


MAX_CONTEXT_SIZE = 4192000  # 4M tokens maximum limit (prevent OOM)
DEFAULT_OLLAMA_HOST    = 'http://127.0.0.1:11434'
DEFAULT_LLAMACPP_HOST  = 'http://127.0.0.1:8080'
DEFAULT_OLLAMA_PORT    =  11434
DEFAULT_LLAMACPP_PORT  =  8080



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
        'aliases': ['/debug on', '/debug off'],
        'category': 'Settings',
        'description': 'Toggle debug mode (raw JSON output)',
        'usage': '/debug on|off',
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


def get_command_by_alias(alias):
    """Look up command metadata by any of its aliases."""
    alias_lower = alias.lower().strip()
    for name, info in COMMANDS.items():
        if alias_lower in [a.lower() for a in info['aliases']]:
            return name, info
    return None, None


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


def get_theme(theme_name: str = "default"):
    """Get theme color dictionary."""
    if os.environ.get('NO_COLOR'):
        return BUILTIN_THEMES["minimal"]
    
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
        theme = ACTIVE_THEME
    
    if not colors_enabled() and not force_color:
        return text

    start_code = theme.get(role, '')
    reset_code = theme['reset']
    
    # Wrap color codes in \x01 and \x02 so readline ignores their length
    if is_prompt and READLINE_AVAILABLE:
        start_code = f"\x01{start_code}\x02" if start_code else ""
        reset_code = f"\x01{reset_code}\x02" if reset_code else ""
        
    return f"{start_code}{text}{reset_code}"


def c(role, theme=None):
    """Get color code from active theme."""
    if theme is None:
        theme = ACTIVE_THEME
    if not colors_enabled():
        return ''
    return theme.get(role, '')


ACTIVE_THEME = get_theme()


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
    dangerous_patterns = [';', '|', '&&', '||', '`']
    for pattern in dangerous_patterns:
        if pattern in command:
            return None

    # Escape special shell characters
    command = re.sub(r'`+', '', command)  # Remove backticks
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

    # Block dangerous utilities
    dangerous = ['rm -rf', 'mkfs', 'dd if=', 'wget ', 'curl ',
                 'nc -e', 'python -c', 'perl -e', 'bash -c']
    for pattern in dangerous:
        if pattern.lower() in command.lower():
            return False

    # Check for shell escape sequences
    if re.search(r'\\[\"\'\$\`]', command):
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


def fetch_models_ollama(base_url):
    """Fetch available models from Ollama API."""
    try:
        url = f"{base_url}/api/tags"
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'})) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('models', [])
    except Exception:
        return []


def parse_size(size_bytes):
    """Parse size from the API into human-readable format."""
    if not size_bytes:
        return "N/A"
    try:
        size_bytes = int(size_bytes)
        if size_bytes > 0:
            return f"{size_bytes / (1024**3):.1f} GB"
    except (TypeError, ValueError):
        pass
    return "N/A"


def fetch_model_info_ollama(base_url, model_name):
    """Fetch detailed model information via Ollama /api/show endpoint."""
    try:
        url = f"{base_url}/api/show"
        payload = json.dumps({"name": model_name}).encode('utf-8')
        req = Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return {}


class CommandContext:
    """Singleton that holds all shared state for the application.
    
    Replaces the scattered self.* attributes across ChatLoop, ModelQuery,
    and other classes with a single centralized state object.
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
        """Reset session state without changing connection info."""
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.context_size = None
        self.current_images = []
        self.force_no_thinking = False
        self.models = []
        self.debug_mode = False
        self.stream_enabled = True
        self.total_queries = 0
        self.total_tokens_generated = 0
        self.total_prompt_tokens = 0
        self.total_time_spent = 0.0
        self.total_chars_generated = 0
        self.query_history = []

    def create_completer(self):
        """Create a ChatCompleter using this context's connection info."""
        return ChatCompleter(self.base_url, self.backend)
    
    def create_query_handler(self):
        """Create a ModelQuery using this context's connection info."""
        return ModelQuery(self.base_url, self.backend)

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
        """Estimate token count from text."""
        if not text:
            return 0
        tokens = len(re.findall(r'\b\w+\b|[^\w\s]', text))
        return max(1, int(tokens * 1.1))

    def calculate_context_tokens(self, messages: list) -> int:
        """Calculate estimated total tokens in conversation context."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += self.estimate_tokens(content)
            total += 2
        return total

    def colorize(self, text: str, role: str, force_color: bool = False, is_prompt: bool = False):
        """Delegate colorize to ACTIVE_THEME and colors_enabled."""
        return colorize(text, role, force_color=force_color, is_prompt=is_prompt)

    def c(self, role: str):
        """Get color code for a role."""
        return c(role)


# ============================================================================
# ============= MODEL QUERY CLASS ===========================================
# ============================================================================

class ModelQuery:
    """Unified query handler for both Ollama and Llama.cpp backends."""

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

    @property
    def base_url(self):
        return self.ctx.base_url
    
    @property
    def backend(self):
        return self.ctx.backend

    def estimate_tokens(self, text):
        """Estimate token count from text. Uses CommandContext for consistency."""
        if not text:
            return 0
        tokens = len(re.findall(r'\b\w+\b|[^\w\s]', text))
        return max(1, int(tokens * 1.1))

    def calculate_context_tokens(self, messages):
        """Calculate estimated total tokens in conversation context."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += self.estimate_tokens(content)
            total += 2
        return total

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
            "content_length": len(content),
            "cumulative_queries": self.ctx.total_queries,
            "cumulative_tokens": self.ctx.total_tokens_generated,
            "cumulative_avg_tps": self.ctx.total_tokens_generated / self.ctx.total_time_spent if self.ctx.total_time_spent > 0 else 0.0
        }
    
        # Update context cumulative stats instead of instance variables
        self.ctx.total_queries += 1
        self.ctx.total_tokens_generated += eval_count
        self.ctx.total_prompt_tokens += prompt_tokens
        self.ctx.total_time_spent += total_time
        self.ctx.total_chars_generated += len(content)
        
        # Update context rolling history instead of instance variable
        entry = {
            "timestamp": time.time(),
            "tokens": eval_count,
            "prompt_tokens": prompt_tokens,
            "time": total_time,
            "tps": tps
        }
        self.ctx.query_history.append(entry)
        if len(self.ctx.query_history) > self.ctx.max_history:
            self.ctx.query_history.pop(0)
        
        # Add cumulative stats from context to return value
        current_stats["cumulative_queries"] = self.ctx.total_queries
        current_stats["cumulative_tokens"] = self.ctx.total_tokens_generated
        current_stats["cumulative_avg_tps"] = (
            self.ctx.total_tokens_generated / self.ctx.total_time_spent 
            if self.ctx.total_time_spent > 0 else 0.0
        )
        
        return current_stats
    

    def print_stats_display(self, stats, show_cumulative=False):
        """Print formatted stats to stderr."""
        if not stats:
            return
            
        # Current query stats
        parts = [f"{stats['total_time']:.2f}s total"]
        
        if stats.get("eval_count", 0) > 0:
            parts.append(f"{stats['tps']:.2f} t/s")
            # Show real context size from backend, with fallback
            ctx = stats.get("total_context_tokens", 0)
            if not ctx:  # Fallback if backend didn't provide total_tokens
                ctx = stats.get("prompt_eval_count", 0) + stats['eval_count']
            parts.append(f"Context: {ctx} tokens")
        else:
            parts.append(f"Content: {stats.get('content_length', 0)} chars")
        
        # Append cumulative stats if requested
        if show_cumulative and stats.get("cumulative_queries", 0) > 1:
            parts.append(f"Cum: {stats['cumulative_queries']}q | "
                        f"{stats['cumulative_tokens']} tok | "
                        f"{stats['cumulative_avg_tps']:.2f} avg t/s")
        sys.stderr.write(colorize(f"\n--- Stats: {' | '.join(parts)} ---\n", 'muted'))


    def get_cumulative_stats(self):
        """Return a summary of all tracked usage from the context."""
        return {
            "total_queries": self.ctx.total_queries,
            "total_completion_tokens": self.ctx.total_tokens_generated,
            "total_prompt_tokens": self.ctx.total_prompt_tokens,
            "total_tokens": self.ctx.total_tokens_generated + self.ctx.total_prompt_tokens,
            "total_time_seconds": self.ctx.total_time_spent,
            "avg_tps": self.ctx.total_tokens_generated / self.ctx.total_time_spent if self.ctx.total_time_spent > 0 else 0.0,
            "avg_tokens_per_query": self.ctx.total_tokens_generated / self.ctx.total_queries if self.ctx.total_queries > 0 else 0.0
        }
    
    def reset_stats(self):
        """Reset all cumulative tracking via the context."""
        self.ctx.total_queries = 0
        self.ctx.total_tokens_generated = 0
        self.ctx.total_prompt_tokens = 0
        self.ctx.total_time_spent = 0.0
        self.ctx.total_chars_generated = 0
        self.ctx.query_history = []   
       

    def build_request_payload(self, messages, model, stream_enabled=False, **kwargs):
        """Build request payload for the backend."""
        if images := kwargs.get('images'):
            if messages and messages[-1].get("role") == "user":
                messages[-1]["images"] = images
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream_enabled
        }
        
        if context_size := kwargs.get('context_size'):
            if self.backend == "ollama":
                payload["options"] = {"num_ctx": context_size}
            else:  # llamacpp
                payload["max_tokens"] = context_size
        
        return payload

    def query_sync(self, messages, model, stream_enabled=False, **kwargs):
        """Non-streaming sync query wrapper."""
        payload = self.build_request_payload(messages, model, stream_enabled=False, **kwargs)

        try:
            url = f"{self.base_url}/api/chat" if self.backend == "ollama" else \
                  f"{self.base_url}/v1/chat/completions"

            data = json.dumps(payload).encode('utf-8')
            req = Request(url, data=data, headers={'Content-Type': 'application/json'})

            with urlopen(req) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            sys.stderr.write(colorize(f"[ERROR] Sync query failed: {e}\n", 'error'))
            return ""


    # ==========================================
    # === OLLAMA STREAMING FUNCTION ===
    # ==========================================

    def query_stream_ollama(
        self,
        messages, model, stream_enabled=True, debug=False,
        show_thinking=True, context_size=None, images=None
    ):
        """Stream response from Ollama API with thinking support."""
        full_content = ""
        start_time = time.time()
        # Always confirm debug mode at start
        if debug:
            sys.stderr.write(colorize(f"\n[DEBUG] Stream started | model={model} | messages={len(messages)}\n", 'muted'))

        try:
            # === Build payload for Ollama ===
            #if images: messages[1]["images"] = images
            if images and messages and messages[-1].get("role") == "user":
                messages[-1]["images"] = images
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream_enabled
            }
            if context_size is not None:
                payload["options"] = {"num_ctx": context_size}

            # === Send request and handle streaming chunks ===
            api_url = f"{self.base_url}/api/chat"
            data = json.dumps(payload).encode('utf-8')
            req = Request(api_url, data=data, headers={'Content-Type': 'application/json'})

            start_thinking = False
            thinking_buffer = ""
            started_content = False
            usage_stats = {}

            with urlopen(req) as response:
                for line in response:
                    decoded_line = line.decode('utf-8').strip()
                    if not decoded_line or decoded_line == '[DONE]':
                        continue

                    try:
                        chunk = json.loads(decoded_line)


                        # === DEBUG: Log final chunk to see usage structure ===
                        if debug and chunk.get("done", False):
                            #sys.stderr.write(colorize(f"\n[DEBUG] Final chunk keys: {list(chunk.keys())}\n", 'muted'))
                            formatted_json = json.dumps(chunk, indent=4)
                            sys.stderr.write(colorize(f"\n[DEBUG] Final JSON chunk from server:\n{formatted_json}\n", 'muted'))
                            if "usage" in chunk:
                                sys.stderr.write(colorize(f"[DEBUG] usage: {chunk['usage']}\n", 'muted'))
                                # Ollama sometimes puts usage fields at root level
                            #if "prompt_eval_count" in chunk or "eval_count" in chunk:
                            #    sys.stderr.write(colorize(f"[DEBUG] root usage fields: prompt_eval_count={chunk.get('prompt_eval_count')}, eval_count={chunk.get('eval_count')}\n", 'muted'))
                        # =====================================================


                        # Extract thinking/reasoning content in query_stream_ollama
                        thought = (chunk.get("message", {}).get("thought") or
                                   chunk.get("message", {}).get("thinking")) or ""
                        content = chunk.get("message", {}).get("content", "")

                        # Track usage stats from Ollama response (only in final chunk with done=True)
                        if chunk.get("done", False):
                            if "usage" in chunk:
                                usage_stats = chunk["usage"]
                            # Fallback to root level keys if no usage block exists
                            else:
                                # Ollama streaming often puts stats at the root level
                                usage_stats = {
                                    "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                                    "eval_count": chunk.get("eval_count", 0),
                                    "total_tokens": chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0)
                                }

                            if debug:  # ← Only print if --debug flag is used
                                sys.stderr.write(colorize(f"\n[DEBUG] Usage: {usage_stats}\n", 'muted'))

                        # Handle thinking with buffered display
                        if thought and show_thinking:
                            if start_thinking == False:
                                start_thinking = True
                                sys.stderr.write("\n<thinking>>\n")
                            sys.stdout.write(thought)
                            sys.stdout.flush()

                        # Flush accumulated thinking when content arrives
                        if content:
                            if thinking_buffer.strip() and started_content:
                                sys.stderr.write("\n<thinking>>\n")
                                thinking_buffer = ""

                            # Strip only leading/trailing whitespace, preserve internal spaces
                            #clean_content = content.strip(' \t\n\r\x0c')
                            clean_content = content

                            if not started_content:
                                print("\n[--- Response ---]", file=sys.stdout)
                                started_content = True

                            sys.stdout.write(clean_content)
                            sys.stdout.flush()
                            full_content += clean_content

                    except json.JSONDecodeError:
                        continue

            # === Calculate and print final stats ===
            total_time = time.time() - start_time
            #usage = self.calculate_stats(total_time, full_content, usage_stats)
            usage = self.calculate_stats(total_time, full_content, usage_stats, messages )
            self.print_stats_display(usage)

            return full_content

        except Exception as e:
            sys.stderr.write(colorize(f"\n[ERROR] Ollama streaming failed: {e}\n", 'error'))
            return full_content


    # ==========================================
    # === LLAMA.CPP STREAMING FUNCTION ===
    # ==========================================

    def query_stream_llamacpp(
        self,
        messages, model, stream_enabled=True, debug=False,
        show_thinking=True, context_size=None, images=None
    ):
        """Stream response from Llama.cpp /v1/chat/completions API with thinking support."""
        full_content = ""
        start_time = time.time()

        # 1. ADD DEBUG START LOG
        if debug:
            sys.stderr.write(colorize(f"\n[DEBUG] Llama.cpp Stream started | model={model} | messages={len(messages)}\n", 'muted'))

        try:
            # === Build payload for Llama.cpp ===
            #if images: messages[1]["images"] = images
            if images and messages and messages[-1].get("role") == "user":
                messages[-1]["images"] = images
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream_enabled
            }
            if context_size is not None:
                payload["max_tokens"] = context_size

            # === Send request and handle streaming chunks ===
            api_url = f"{self.base_url}/v1/chat/completions"
            data = json.dumps(payload).encode('utf-8')
            req = Request(api_url, data=data, headers={'Content-Type': 'application/json'})

            start_thinking = False
            started_content = False
            completion_data = {}
            is_final_chunk = False

            with urlopen(req) as response:
                for line in response:
                    decoded_line = line.decode('utf-8').strip()

                    if not decoded_line:
                        continue

                    # Strip SSE 'data: ' prefix
                    if decoded_line.startswith('data: '):
                        decoded_line = decoded_line[6:].strip()

                    if decoded_line == '[DONE]':
                        continue

                    try:
                        chunk = json.loads(decoded_line)

                        # Grab usage stats from the final chunk
                        if "usage" in chunk and chunk["usage"]:
                            completion_data.update(chunk["usage"])
                            is_final_chunk = True
                        # Native Llama.cpp style timings
                        elif "timings" in chunk:
                            completion_data["prompt_eval_count"] = chunk["timings"].get("prompt_n", 0)
                            completion_data["eval_count"] = chunk["timings"].get("predicted_n", 0)
                            is_final_chunk = True
                            
                        # Detect the end of the stream
                        choices = chunk.get("choices", [])
                        if choices and choices[0].get("finish_reason") is not None:
                            is_final_chunk = True

                        if debug and is_final_chunk:
                            formatted_json = json.dumps(chunk, indent=4)
                            sys.stderr.write(colorize(f"\n[DEBUG] Final JSON chunk from server:\n{formatted_json}\n", 'muted'))


                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})

                            # Extract content
                            thought = delta.get("reasoning_content") or ""
                            content = delta.get("content") or ""

                            # 1. Print thinking in real-time as it arrives
                            if thought and show_thinking:
                                if not start_thinking:
                                    sys.stderr.write(colorize("\n<thinking>\n", 'muted'))
                                    start_thinking = True
                                sys.stderr.write(colorize(thought, 'muted'))
                                sys.stderr.flush()

                            # 2. Print content in real-time
                            if content:
                                # Close the thinking block if it was open
                                if start_thinking and not started_content:
                                    sys.stderr.write(colorize("\n</thinking>\n", 'muted'))

                                # Print response header once
                                if not started_content:
                                    print(colorize("\n[--- Response ---]", 'success'), file=sys.stdout)
                                    started_content = True

                                # DO NOT strip spaces here, or words will mash together!
                                sys.stdout.write(content)
                                sys.stdout.flush()
                                full_content += content

                    except json.JSONDecodeError:
                        continue

            # Failsafe: close thinking block if the model thought but generated no content
            if start_thinking and not started_content:
                sys.stderr.write(colorize("\n</thinking>\n", 'muted'))

            # === Calculate and print final stats ===
            total_time = time.time() - start_time
            #usage = self.calculate_stats(total_time, full_content, completion_data)
            usage = self.calculate_stats(total_time, full_content, completion_data, messages)
            self.print_stats_display(usage)

            return full_content

        except Exception as e:
            sys.stderr.write(colorize(f"\n[ERROR] Llama.cpp streaming failed: {e}\n", 'error'))
            return full_content



    # ==========================================
    # === GENERIC STREAMING WRAPPER FUNCTION ===
    # ==========================================

    def query_stream(
        self,
        messages, model, stream_enabled=True, debug=False,
        show_thinking=True, context_size=None, images=None
    ):
        """Generic streaming wrapper that dispatches to correct backend-specific handler."""
        if self.backend == "ollama":
            return self.query_stream_ollama(
                messages, model, stream_enabled, debug,
                show_thinking, context_size, images
            )
        else:  # llamacpp
            return self.query_stream_llamacpp(
                messages, model, stream_enabled, debug,
                show_thinking, context_size, images
            )


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
    """

    def __init__(self, base_url, backend):
        self.base_url = base_url
        self.backend = backend
#        self.commands = [
#            '/?', '/help', '/listmodel', '/switchmodel',
#            '/cwd', '/ls', '/curl', '/spawnshell', '/clear',
#            '/thinkingon', '/thinkingoff', '/contextsizeset',
#            '/debug on', '/debug off', '/stats', '/quit', '/exit'
#        ]
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
        elif buffer.startswith('/cwd ') or buffer.startswith('/ls ') or text.startswith('/image ') or text.startswith('@'):
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

def process_inline_commands(full_input):
    """
    Process inline commands (!, /curl, @) within user input.
    """
    processed_lines = []
    file_inclusions = []
    
    # 1. Scan for inline @filepath mentions anywhere in the text
    # We split by whitespace to extract words starting with '@'
    for word in full_input.split():
        if word.startswith('@') and len(word) > 1:
            filepath = word[1:]
            expanded_path = os.path.expanduser(filepath)
            
            # Only attempt to load if it actually exists as a file
            if os.path.isfile(expanded_path):
                print(colorize(f"[--- Loading file: {filepath} ---]", 'muted'), file=sys.stderr)
                try:
                    with open(expanded_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    
                    lines_count = len(file_content.splitlines())
                    print(colorize(f"Successfully loaded {lines_count} lines ({len(file_content)} chars).", 'info'), file=sys.stderr)
                    
                    file_inclusions.append(f"\n[Content of local file `{filepath}`]:\n```text\n{file_content}\n```\n")
                except UnicodeDecodeError:
                    err_msg = f"[Failed to load `{filepath}`: Appears to be a binary or non-UTF-8 file]"
                    print(colorize(err_msg, 'error'), file=sys.stderr)
                    file_inclusions.append(err_msg + "\n")
                except Exception as e:
                    err_msg = f"[Failed to load `{filepath}`: {e}]"
                    print(colorize(err_msg, 'error'), file=sys.stderr)
                    file_inclusions.append(err_msg + "\n")

    # 2. Process line-by-line commands (! and /curl)
    for line in full_input.split('\n'):
        stripped_line = line.lstrip()

        if stripped_line.startswith("!"):
            # Execute shell command
            command = stripped_line[1:].strip()
            if command and validate_shell_command_safety(command):
                output_str = execute_os_command(sanitize_shell_command(command))
            else:
                output_str = f"[Command rejected: Invalid characters]"
            processed_lines.append(output_str)


        elif stripped_line.startswith("/curl "):
            # Fetch URL content
            url = stripped_line[6:].strip()
            if not url:
                processed_lines.append(line)
                continue
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            try:
                print(colorize(f"[--- Fetching URL: {url} ---]", 'muted'), file=sys.stderr)
                text_content, used_tool = fetch_and_convert_url(url)
                word_count = len(text_content.split()) if text_content else 0
                tool_suffix = ""
                if word_count>0 and used_tool and used_tool != "None":
                    tool_suffix = f" (via {used_tool})"
                if word_count>0:
                    print(colorize(f"[Successfully fetched {word_count} words from {url}{tool_suffix}]", 'muted'), file=sys.stderr)
                    processed_lines.append(text_content)
                    preview_length = 500
                    preview_text=""
                    header_text=""
                    if len(text_content) > preview_length:
                        header_text += "[Output truncated for terminal display. LLM received full text.]"
                    preview_text += text_content[:preview_length].strip()
                    print(colorize(f"{header_text}\n```", 'muted'), file=sys.stderr)
                    print(colorize(f"```text\n{preview_text}\n```", 'info'), file=sys.stderr)
                else:
                    print(colorize(f"[Warning: No text content could be extracted from {url}]", 'warning'), file=sys.stderr)
            except Exception as e:
                print(colorize(f"[Failed to fetch URL: {e}]", 'error'), file=sys.stderr)
        else:
            # Append normal text (including the text containing the @ tag itself)
            processed_lines.append(line)

    # 3. Combine the user's original prompt with the loaded file contents at the bottom
    final_output = "\n".join(processed_lines)
    if file_inclusions:
        final_output += "\n" + "".join(file_inclusions)
        
    return final_output



def execute_os_command(command):
    """
    Execute OS command with safety checks and timeout.

    Args:
        command (str): Command to execute

    Returns:
        str: Command output or error message
    """
    # Validate command safety
    if not validate_shell_command_safety(command, max_length=500):
        msg = "[Command rejected: Invalid characters]"
        print(colorize(msg, 'error'), file=sys.stderr)
        return "[Command rejected: Invalid characters]"

    print(f"[--- Executing (max 5s): {command} ---]", file=sys.stderr)
    output_lines = []

    try:
        # Use timeout wrapper for safety
        process = subprocess.run(
            command, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            timeout=5
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
        print("[Command timed out after 5 seconds!]", file=sys.stderr)
        output = "[Command execution interrupted: Time limit exceeded (5s)]"

    except Exception as e:
        print(f"[Failed to execute command: {e}]", file=sys.stderr)
        return f"[Execution error: {e}]"

    return f"\n[Command executed: `{command}`]\n```text\n{output.strip()}\n```\n"


def fetch_and_convert_url(url):
    """Fetch HTML from URL and convert to plain text. Returns (text, tool_name)."""
    try:
        html_bytes = get_html_bytes(url)
        if not html_bytes:
            return "", "None"

        tool_map = {
            'pandoc': 'pandoc',
            'html2text': 'html2text',
            'lynx': 'lynx',
        }
        for converter in ['pandoc' ,'html2text', 'lynx']:
            cmd = [converter, '-stdin'] if converter == 'lynx' else [converter]
            try:
                if converter not in ('html2text', 'pandoc') and not shutil.which(converter):
                    continue
                proc = subprocess.run(
                    cmd, input=html_bytes, capture_output=True,
                    timeout=10, check=False
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.decode('utf-8', errors='replace'), f"{tool_map[converter]}"
            except Exception:
                continue

        # Fallback: Simple HTML strip
        html_str = html_bytes.decode('utf-8', errors='ignore')
        return FallbackHTMLStripper().get_data(html_str), "FallbackHTMLStripper"

    except Exception as e:
        return f"[Failed to fetch URL: {e}]", "None"


def get_html_bytes(url, depth=0):
    """Fetch HTML bytes while following redirects gracefully."""
    if depth > 3:
        return b""

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    # Try curl/wget first, fallback to urllib
    for tool in ['curl', 'wget']:
        if shutil.which(tool):
            proc = subprocess.run(
                [tool, '-L', '-s', '-A', headers['User-Agent'], url],
                capture_output=True, timeout=15, check=False
            )
            return proc.stdout

    # Fallback to urllib
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as response:
            return response.read()
    except Exception:
        return b""


class FallbackHTMLStripper:
    """Simple HTML content stripper that removes tags."""

    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
        self.skip_tags = {'script', 'style', 'head', 'meta',
                          'title', 'link', 'noscript'}
        self.current_tag = ""

    def reset(self):
        """Reset stripper state."""
        super(FallbackHTMLStripper, self).reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
        self.skip_tags = {'script', 'style', 'head', 'meta',
                          'title', 'link', 'noscript'}
        self.current_tag = ""

    def handle_starttag(self, tag, attrs):
        """Handle opening HTML tags."""
        self.current_tag = tag

    def handle_endtag(self, tag):
        """Handle closing HTML tags."""
        self.current_tag = ""

    def handle_data(self, data):
        """Handle text data between tags."""
        if self.current_tag not in self.skip_tags:
            cleaned = data.strip()
            if cleaned:
                self.text.append(cleaned)

    def feed(self, html_text):
        """Parse HTML text."""
        from html.parser import HTMLParser
        parser = HTMLParser()
        parser.reset()
        parser.strict = False
        parser.convert_charrefs = True
        parser.skip_tags = self.skip_tags
        parser.current_tag = ""

        class LocalParser(html.parser.HTMLParser):
            def __init__(self, *args, **kwargs):
                html.parser.HTMLParser.__init__(self, *args, **kwargs)
                self.outer_text = []
                self.inner_text = ''

            def handle_starttag(self, tag, attrs):
                html.parser.HTMLParser.handle_starttag(self, tag, attrs)

            def handle_endtag(self, tag):
                html.parser.HTMLParser.handle_endtag(self, tag)

            def handle_data(self, data):
                html.parser.HTMLParser.handle_data(self, data)

        parser = LocalParser()
        parser.feed(html_text)

    def get_data(self, text=None):
        """Get stripped text content."""
        if not text and self.text:
            return '\n'.join(self.text)
        elif text:
            self.text.clear()

            class DataGatherer(HTMLParser):
                def __init__(self, *args, **kwargs):
                    HTMLParser.__init__(self, *args, **kwargs)
                    self.outer_text = []
                    self.inner_text = ''

                def handle_starttag(self, tag, attrs):
                    if tag not in self.skip_tags:
                        html.parser.HTMLParser.handle_starttag(self, tag, attrs)

                def handle_endtag(self, tag):
                    if tag not in self.skip_tags:
                        html.parser.HTMLParser.handle_endtag(self, tag)

                def handle_data(self, data):
                                                if self.current_tag not in self.skip_tags:
                                                    cleaned = data.strip()
                                                    if cleaned:
                                                        self.outer_text.append(cleaned)

                def get_outer(self):
                    return ''.join(self.outer_text)

            parser = DataGatherer(skip_tags=self.skip_tags, current_tag=self.current_tag)
            parser.feed(text)

            return parser.get_outer() if parser.outer_text else ""
        return ''


# ============================================================================
# ============= CHAT LOOP CLASS ============================================
# ============================================================================

class ChatLoop:
    """
    Unified chat loop that handles both Ollama and Llama.cpp backends.
    
    State is managed through a shared CommandContext singleton instead of
    individual self.* attributes.
    """

    def __init__(self, context: CommandContext):
        self.ctx = context
        self.completer = context.create_completer()
        self.query_handler = context.create_query_handler()
        self.commands = get_command_aliases()

    def fetch_models(self):
        """Fetch available models from the backend and auto-select if needed."""
        if self.ctx.backend == "llamacpp":
            self.ctx.models = [m['name'] for m in fetch_models_llamacpp(self.ctx.base_url)]
        else:
            self.ctx.models = [m['name'] for m in fetch_models_ollama(self.ctx.base_url)]
            
        if self.ctx.models and self.ctx.model not in self.ctx.models:
            old_model = self.ctx.model
            self.ctx.model = self.ctx.models[0]
            print(colorize(f"[INFO] Model '{old_model}' not found on server. Auto-selecting '{self.ctx.model}'", 'warning'), file=sys.stderr)

    def run(self, stream_enabled=True, debug=False, images=None):
        """Main chat loop - handles interactive session."""

        # Initialize state from args
        self.fetch_models()
        self.ctx.debug_mode = debug
        self.ctx.stream_enabled = stream_enabled

        # Print welcome message using command registry
        print(colorize(f"\n[{self.ctx.backend.upper()} Chat Mode]", 'info'), file=sys.stderr)
        print(format_help_text(compact=True), file=sys.stderr)
        print(colorize("Type /help for details\n", 'muted'), file=sys.stderr)

        # Load startup image if provided via --image
        if images:
            self.ctx.current_images = images

        # Setup readline completer if available
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
            except Exception as e:
                print(colorize(f"[ERROR] Readline setup failed: {e}", 'error'), file=sys.stderr)

        if images:
            print(colorize("[Image loaded for session]", 'success'), file=sys.stderr)

        from urllib.parse import urlparse
        host_clean = urlparse(self.ctx.base_url).netloc

        while True:
            try:
                # Build the dynamic prompt: backend@IPv4/model
                prompt_prefix = f"{self.ctx.backend}@{host_clean}/{self.ctx.model}"
                if self.ctx.current_images:
                    prompt_prefix += colorize("[🖼️]", 'info', is_prompt=True)

                full_input = gather_user_input(prompt_prefix)
                if full_input is None or not full_input.strip():
                    continue

                # Handle exit commands
                if full_input.lower() in ['exit', 'quit', '/exit', '/quit']:
                    print(colorize("\n[Goodbye!]", 'info'), file=sys.stderr)
                    break

                # Handle help command
                if full_input in ['/?', '/help']:
                    print(format_help_text(compact=False), file=sys.stderr)
                    continue

                # Handle stats command
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
                        model_ctx_list=fetch_loaded_models_context_ollama(self.ctx.base_url)
                        for m,c in model_ctx_list:
                            print(f"    model : {m} context : {c}", file=sys.stderr)
                    continue


                # Handle model listing
                if full_input.startswith('/listmodel'):
                    parts = full_input.split(maxsplit=1)
                    if full_input.strip() == '/listmodelall':
                        list_models_ollama(self.ctx.base_url, include_capabilities=True, file=sys.stderr)
                    else:
                        self.list_models(parts[1] if len(parts) > 1 else None)
                    continue

                # Handle context size
                if full_input.startswith('/contextsizeset'):
                    self.set_context_size(full_input)
                    continue

                # Handle clear command
                if full_input == '/clear':
                    print(colorize("[Context memory wiped clean]", 'success'), file=sys.stderr)
                    self.ctx.reset()
                    continue

                # Handle image attach/clear command
                if full_input.startswith('/image'):
                    parts = full_input.split(maxsplit=1)
                    if len(parts) < 2 or parts[1].strip() in ('', 'clear', 'none'):
                        self.ctx.current_images = []
                        print(colorize("[Image cleared]", 'info'), file=sys.stderr)
                    else:
                        img_path = os.path.expanduser(parts[1].strip())
                        if os.path.isfile(img_path):
                            img_data = prepare_image_data(img_path)
                            if img_data:
                                self.ctx.current_images = [img_data]
                                print(colorize(f"[Image attached: {os.path.basename(img_path)}]", 'success'), file=sys.stderr)
                            else:
                                print(colorize("[Error: Could not encode image]", 'error'), file=sys.stderr)
                        else:
                            print(colorize(f"[Error: File not found: {img_path}]", 'error'), file=sys.stderr)
                    continue



                # Handle debug mode
                if full_input.lower() == '/debug on':
                    self.ctx.debug_mode = True
                    print(colorize("[Debug mode ENABLED]", 'success'), file=sys.stderr)
                    continue
                elif full_input.lower() == '/debug off':
                    self.ctx.debug_mode = False
                    print(colorize("[Debug mode DISABLED]", 'info'), file=sys.stderr)
                    continue

                # Handle thinking control
                if full_input == '/thinkingoff':
                    self.ctx.force_no_thinking = True
                    print(colorize("[Model will skip reasoning phase]", 'warning'), file=sys.stderr)
                    continue
                elif full_input == '/thinkingon':
                    self.ctx.force_no_thinking = False
                    print(colorize("[Reasoning phase enabled]", 'success'), file=sys.stderr)
                    continue

                # Handle directory change
                if full_input.startswith('/cwd'):
                    parts = full_input.split(maxsplit=1)
                    if len(parts) > 1:
                        try:
                            os.chdir(os.path.expanduser(parts[1]))
                        except Exception as e:
                            print(colorize(f"[ERROR] {e}", 'error'), file=sys.stderr)
                    print(colorize(f"[Current directory: {os.getcwd()}]", 'info'), file=sys.stderr)
                    continue

                # Handle ls command
                if full_input.startswith('/ls'):
                    try:
                        subprocess.run("ls" + full_input[3:], shell=True, check=False)
                    except Exception as e:
                        print(colorize(f"[ERROR] {e}", 'error'), file=sys.stderr)
                    continue

                # Handle model switch
                if full_input.startswith('/switchmodel'):
                    parts = full_input.split()
                    if len(parts) > 1:
                        self.ctx.model = parts[1]
                        print(colorize(f"[Switched to '{self.ctx.model}']", 'success'), file=sys.stderr)
                    continue

                # Handle spawnshell command
                if full_input == '/spawnshell':
                    self.handle_spawnshell()
                    continue

                # Process inline commands and build message
                final_content = process_inline_commands(full_input)
                if not final_content.strip():
                    continue

                # Track conversation history within ChatLoop (not in shared context)
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
                    response = self.query_handler.query_stream(
                        payload_messages,
                        self.ctx.model,
                        stream_enabled=self.ctx.stream_enabled,
                        debug=self.ctx.debug_mode,
                        show_thinking=(not self.ctx.force_no_thinking),
                        context_size=self.ctx.context_size,
                        images=self.ctx.current_images
                    )

                    if response:
                        self.messages.append({'role': 'assistant', 'content': response})
                        print()

            except KeyboardInterrupt:
                print(f"\n[Interrupted]", file=sys.stderr)
                continue

            except Exception as e:
                if isinstance(e, EOFError):
                    print(f"[EOF - Goodbye!]", file=sys.stderr)
                    break
                else:
                    print(colorize(f"[ERROR] ChatLoop->run {e}", 'error'), file=sys.stderr)

        return

    def list_models(self, filter_arg=None):
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

    def set_context_size(self, full_input):
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

    def handle_spawnshell(self):
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
                    while True:
                        data = os.read(fd, 4096)
                        if not data:
                            break
                        output_lines.extend(data.decode('utf-8', errors='replace'))
                except OSError:
                    pass

            pty.spawn(shell_cmd, read_output)

        except Exception as e:
            print(f"[ERROR] Shell exited: {e}", file=sys.stderr)

        return "".join(output_lines).strip()


# ============================================================================
# ============= MAIN ENTRY POINT ===========================================
# ============================================================================

def get_base_url(args, backend):
    """Get base URL for the specified backend."""
    if args.host:
        base_url = args.host
    else:
        default = DEFAULT_LLAMACPP_HOST if backend == "llamacpp" else DEFAULT_OLLAMA_HOST
        env_var = f'{backend.upper()}_HOST'
        base_url = os.environ.get(env_var, default)

    # Ensure URL prefix
    if not base_url.startswith(('http://', 'https://')):
        base_url = f"http://{base_url}"

    return base_url


def build_messages(args, input_data, image_path):
    """Build messages payload from args."""
    messages = [
        {'role': 'system', 'content': args.prompt}
    ]

    if input_data:
        messages.append({'role': 'user', 'content': input_data})

    # Add image if provided
    if image_path:
        image_data = prepare_image_data(image_path)
        if image_data and messages[-1]['role'] == 'user':
            messages[-1]['images'] = [image_data]

    return messages


def list_models(base_url, backend):
    """List all available models and exit."""
    if backend == "llamacpp":
        list_models_llamacpp(base_url)
    else:
        list_models_ollama(base_url)
    sys.exit(0)


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
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'})) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('models', [])
    except Exception:
        return []

def fetch_loaded_models_context_ollama(base_url: str) -> list[tuple[str, int]]:
    """
    Query Ollama `/api/ps` and return a list of (model_name, context_length).

    Example output:
        [ ("nemotron-cascade-2:30b", 131072),
          ("llama2", 4096) ]
    """
    try:
        url = f"{base_url}/api/ps"
        with urlopen(
            Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        ) as response:
            data = json.loads(response.read().decode('utf-8'))
            return [
                (m["model"], m.get("context_length", 0))
                for m in data.get("models", [])
                if isinstance(m, dict) and "model" in m
            ]
    except Exception:                     # network error, parsing error, etc.
        sys.stderr.write(colorize("[ERROR] Could not read Ollama model info", "error"))
        return []


def check_backend_with_head(url, server_marker):
    """Attempt HEAD request to URL and check for server header."""
    try:
        request = Request(url, method='HEAD')
        with urlopen(request, timeout=1.0) as response:
            server_header = response.headers.get('Server', '').lower()
            return server_marker.lower() in server_header
    except (HTTPError, URLError, OSError, TimeoutError, Exception):
        return False



def check_backend_with_get(url, server_marker):
    """Attempt HEAD request to URL and check for server header."""
    try:
        request = Request(url, method='GET')
        with urlopen(request, timeout=1.0) as response:
            my_response = str(response.read())
            my_response = my_response.lower()
            #print(my_response)
            if "ollama" in my_response:
                return True
    except (HTTPError, URLError, OSError, TimeoutError, Exception):
        return False
 

def auto_detect_backend():
    """Auto-detect backend based on default ports using HEAD request.
    
    Checks if both backends are running at their default ports:
    - 127.0.0.1:8080 for llama.cpp
    - 127.0.0.1:11434 for ollama
    
    Uses HTTP HEAD request with timeout of 1 second and checks for
    'Server: llama.cpp' or 'Server: ollama' headers.
    
    Returns:
        str: 'llamacpp', 'ollama', or None if neither detected
    """
   
    # Default URLs
    llama_cpp_url = DEFAULT_LLAMACPP_HOST
    ollama_url =    DEFAULT_OLLAMA_HOST
    
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


    # grab the ip of the host 
    #  socket.gethostbyname_ex(socket.gethostname())[-1] 
    # ('myhost.home', [], ['192.168.1.19', '192.168.1.20'])
    list_of_ip = socket.gethostbyname_ex(socket.gethostname())[-1]
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
            pass
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
        selected_backend = args.backend or "ollama"  # Default to ollama if only host is provided
        save_backend_config(selected_backend, base_url)
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
    else:
        fallback_host = os.environ.get('OLLAMA_HOST', DEFAULT_OLLAMA_HOST)

    return fallback_backend, fallback_host


# ============================================================================
# ============= ARGUMENT PARSER ==============================================
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified LLM Query Interface for Ollama & Llama.cpp"
    )

    # Backend selection
    group = parser.add_argument_group('Backend')
    parser.add_argument("-b", "--backend", choices=["ollama", "llamacpp"], default=None, help="API backend to use (auto-detected if omitted).")
    group.add_argument('-H', '--host', help='Custom API URL')

    # Listing operations
    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument('-l', '--list', action="store_true",
                           help='List all models and exit')
    list_group.add_argument('-la', '--list-all', action="store_true",
                          help='List models with capabilities (Ollama only)')

    # Model info operations
    info_group = parser.add_mutually_exclusive_group()
    info_group.add_argument('--show', action="store_true",
                          help='Show concise model details')
    info_group.add_argument('--show-details', action="store_true",
                          help='Show full model information')

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-I', '--input-text', help='Direct query text')
    input_group.add_argument('-i', '--input-file', help='Input file path')
    input_group.add_argument('--input-dir', help='Directory of input files')

    # Batch processing options
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument('-c', '--chat', action="store_true",
                            help='Start interactive chat session')
    batch_group.add_argument('-o', '--output', help='Output file path')
    batch_group.add_argument('--output-dir', help='Output directory for batches')

    parser.add_argument('-m', '--model', default=None,
                       help='Model name')  # NOT mutually exclusive
    # System Prompt configuration
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument('-P', '--profile', choices=list(BUILTIN_PROMPTS.keys()),
                            help='Use a built-in system prompt profile')
    prompt_group.add_argument('--prompt', help='Custom system prompt text')

    # Image support
    parser.add_argument('--image', help='Image file for multimodal models')

    # Display options
    parser.add_argument('-p', '--no-stream', action="store_true",
                       help='Disable streaming output')
    parser.add_argument('--debug', action="store_true",
                       help='Print raw JSON to stderr')
    parser.add_argument('--format', choices=["json", "yaml"], default="json",
                       help='Output format for model info (default: json)')
    parser.add_argument('--theme', default=None,
                       choices=["default", "minimal", "emacs_dark", "vim_dark", "high_contrast"],
                       help='Color theme for output')
    parser.add_argument('--no-color', action="store_true",
                       help='Disable colored output')

    # Parse arguments
    args = parser.parse_args()

    # Initialize theme system
    global ACTIVE_THEME
    theme_to_use = "default"
    if args.no_color:
        os.environ['NO_COLOR'] = '1'
        theme_to_use = "minimal"
    elif args.theme is not None:
        theme_to_use = args.theme
    elif os.environ.get('OLLAMAQUERY_THEME'):
        theme_to_use = os.environ.get('OLLAMAQUERY_THEME')
        # Ensure we have a string value
        if theme_to_use is None:
            theme_to_use = "default"
    
    ACTIVE_THEME = get_theme(theme_to_use)
    if args.prompt:
        active_prompt = args.prompt
    elif args.profile:
        active_prompt = BUILTIN_PROMPTS[args.profile]
    else:
        active_prompt = DEFAULT_SYSTEM_PROMPT
    args.prompt = active_prompt
    # ---------------------------------

    # determine backend first so we know where to check for loaded models
    backend, base_url = resolve_connection(args)

    # 1. Use explicitly requested model if provided via -m
    if args.model:
        target_model = args.model
    # 2. Check for an already loaded model in memory for Ollama on^ly
    elif backend == "ollama":
        loaded_models = fetch_loaded_models_ollama(base_url)
        if loaded_models:
            target_model = loaded_models[0]['name']
            sys.stderr.write(colorize(f"[INFO] Auto-selected active model in memory: '{target_model}'\n", 'success'))
        else:
            target_model = "llama3" # Fallback if nothing is in memory
    # 3. Fetch the currently hosted model (Llama.cpp)
    elif backend == "llamacpp":
        available_models = fetch_models_llamacpp(base_url)
        if available_models:
            target_model = available_models[0]['name']
            sys.stderr.write(colorize(f"[INFO] Auto-selected hosted model: '{target_model}'\n", 'success'))
        else:
            target_model = "llama3" # Fallback

    # Handle listing operations
    if args.list or args.list_all:
        if backend == "llamacpp":
            list_models_llamacpp(base_url, filter_arg=args.model)
        if backend == "ollama":
            if args.list_all:
              list_models_ollama(
                  base_url,
                  filter_arg=args.model,
                  include_capabilities=(args.list_all))
            else:
              list_models_ollama(
                  base_url,
                  filter_arg=args.model,
                  include_capabilities=False)
            
        sys.exit(0)

    # Handle model info operations
    if args.show or args.show_details:
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

        image_data = prepare_image_data(args.image) if args.image else None
        images_list = [image_data] if image_data else None
        if images_list:
            ctx.current_images = images_list

        should_stream = not args.no_stream and sys.stdout.isatty()
        loop = ChatLoop(ctx)
        loop.run(stream_enabled=should_stream, debug=args.debug, images=images_list)
        sys.exit(0)

    # Batch processing with input text/file/directory
    else:
        if args.input_text:
            input_data = args.input_text
        elif args.input_file:
            if not os.path.isfile(args.input_file):
                print(f"[ERROR] File '{args.input_file}' not found", file=sys.stderr)
                sys.exit(1)

            with open(args.input_file, 'r', encoding='utf-8') as f:
                input_data = f.read()
        elif args.input_dir:
            if not args.output_dir:
                print("[ERROR] --output-dir required for --input-dir", file=sys.stderr)
                sys.exit(1)

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)

            for filename in sorted(os.listdir(args.input_dir)):
                input_path = os.path.join(args.input_dir, filename)
                print(f"[Processing: {filename}...]")

                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                messages = [
                    {'role': 'system', 'content': args.prompt},
                    {'role': 'user', 'content': content}
                ]

                if args.image:
                    image_data = prepare_image_data(args.image)
                    if image_data:
                        images_list = [image_data]
                else:
                    images_list = None

                query_handler = ModelQuery(context=CommandContext())
                query_handler.ctx.base_url = base_url
                query_handler.ctx.backend = backend

                response = query_handler.query_sync(
                    messages,
                    target_model,
                    context_size=None,
                    show_thinking=True,
                    debug=args.debug,
                    images=images_list
                )

                output_path = os.path.join(args.output_dir, filename + '.output')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response)

            sys.exit(0)

        # Single query with input text/file
        if args.input_text or args.input_file:
            image_data = prepare_image_data(args.image) if args.image else None

            messages = [
                {'role': 'system', 'content': args.prompt}
            ]

            if args.input_text:
                messages.append({'role': 'user', 'content': args.input_text})
            elif args.input_file and os.path.isfile(args.input_file):
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                messages.append({'role': 'user', 'content': content})

            # Add image if provided
            if image_data and messages[-1]['role'] == 'user':
                messages[-1]['images'] = [image_data]

            query_handler = ModelQuery(context=CommandContext())
            query_handler.ctx.base_url = base_url
            query_handler.ctx.backend = backend
            should_stream = not args.no_stream and sys.stdout.isatty()

            response = query_handler.query_sync(
                messages,
                target_model,
                context_size=None,
                show_thinking=True,
                debug=args.debug,
                images=image_data
            )

            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(response)
                print(f"[Success: Output saved to {args.output}]", file=sys.stderr)
            elif not should_stream and response:
                print(response)

        # Error handling for missing input
        if not (args.chat or args.input_text or args.input_file or
                 args.input_dir or args.list or args.show or args.show_details):
            parser.error(
                "You must provide an input (-I, -i, --input-dir), start chat (-c), "
                "or list models/info (-l/--show/--list)"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\n[Exiting gracefully...]")
        sys.exit(0)


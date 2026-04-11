#!/usr/bin/python3

# ============================================================================
# ============= IMPORTS & CONFIGURATION ======================================
# ============================================================================

import os
import sys
import json
import re
import html
import base64
import argparse
import subprocess
import threading
import time
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

DEFAULT_SYSTEM_PROMPT = (
    "You are a chatbot trying to help user. Try to rebound to the question as best "
    "as your knowledge goes but reply politely that you don't know if it is the case."
)

MAX_CONTEXT_SIZE = 4192000  # 4M tokens maximum limit (prevent OOM)
DEFAULT_OLLAMA_HOST = 'http://127.0.0.1:11434'
DEFAULT_LLAMACPP_HOST = 'http://127.0.0.1:8080'


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
        print(f"[ERROR] Loading image {image_path}: {e}", file=sys.stderr)
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


def fetch_models_llamacpp(base_url):
    """Fetch available models from Llama.cpp API."""
    try:
        url = f"{base_url}/v1/models"
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'})) as response:
            data = json.loads(response.read().decode('utf-8'))
            return [{"name": m.get("id", "unknown"),
                     "owned_by": m.get("owned_by", "N/A")} for m in data.get('data', [])]
    except Exception:
        return []


# ============================================================================
# ============= MODEL QUERY CLASS ===========================================
# ============================================================================

class ModelQuery:
    """Unified query handler for both Ollama and Llama.cpp backends."""

    def __init__(self, base_url, backend):
        self.base_url = base_url
        self.backend = backend

    # ==========================================
    # === STATISTICS TRACKING HELPER FUNCTIONS ===
    # ==========================================

    @staticmethod
    def calculate_stats(total_time, content, usage=None):
        """Calculate and return stats dictionary for display."""
        eval_count = 0
        prompt_tokens = 0

        if usage:
            eval_count = usage.get("completion_tokens", 0) or len(content.split())
            prompt_tokens = usage.get("prompt_tokens", 0)

        elif content:
            # Fallback to word count if no token data available
            eval_count = len(content.split())

        tps = 0.0
        if eval_count > 0 and total_time > 0:
            tps = eval_count / total_time

        return {
            "eval_count": eval_count,
            "prompt_eval_count": prompt_tokens,
            "total_time": total_time,
            "tps": tps,
            "content_length": len(content)
        }

    def print_stats_display(self, stats):
        """Print formatted stats to stderr."""
        if stats and stats.get("eval_count", 0):
            sys.stderr.write(
                f"\n--- Stats: {stats['total_time']:.2f}s total | "
                f"{stats['tps']:.2f} t/s | "
                f"Context: {stats.get('prompt_eval_count', 0) + stats['eval_count']} tokens ---\n",
                flush=True
            )
        elif stats and not stats["eval_count"]:
            # No token tracking, just show content length
            sys.stderr.write(
                f"\n--- Stats: {stats['total_time']:.2f}s total | "
                f"Content: {stats.get('content_length', 0)} chars ---\n",
                flush=True
            )
        else:
            # Minimal stats when streaming without usage data
            sys.stderr.write(
                f"\n--- Stats: {stats['total_time']:.2f}s total | "
                f"Tokens processed: {len(stats.get('content', '').split()) if stats.get('content') else 0} ---\n",
                flush=True
            )

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
            sys.stderr.write(f"[ERROR] Sync query failed: {e}\n")
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

        try:
            # === Build payload for Ollama ===
            if images: messages[1]["images"] = images
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

                        # Extract thinking/reasoning content
                        thought = (chunk.get("message", {}).get("thought") or
                                   chunk.get("message", {}).get("thinking")) or ""
                        content = chunk.get("message", {}).get("content", "")

                        # Track usage stats from Ollama response
                        if "usage" in chunk:
                            usage_stats = chunk["usage"]

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
            usage = self.calculate_stats(total_time, full_content, usage_stats)
            self.print_stats_display(usage)

            return full_content

        except Exception as e:
            sys.stderr.write(f"\n[ERROR] Ollama streaming failed: {e}\n")
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

        try:
            # === Build payload for Llama.cpp ===
            if images: messages[1]["images"] = images
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

            thinking_buffer = ""
            started_content = False
            completion_data = {}
            finish_reason = None

            with urlopen(req) as response:
                for line in response:
                    decoded_line = line.decode('utf-8').strip()
                    if not decoded_line or decoded_line == '[DONE]':
                        continue

                    try:
                        chunk = json.loads(decoded_line)

                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})

                            # Extract thinking/reasoning content
                            thought = delta.get("reasoning_content") or ""
                            content = delta.get("content") or ""
                            finish_reason = choices[0].get("finish_reason")

                            # Track usage stats from Llama.cpp response
                            if "usage" in chunk:
                                completion_data.update(chunk["usage"])

                            # Handle thinking with buffered display
                            if thought and show_thinking:
                                thinking_buffer += thought

                        # Flush accumulated thinking when content arrives
                        if content:
                            if thinking_buffer.strip() and started_content:
                                sys.stderr.write(f"\n<thinking>{thinking_buffer}</thinking>\n")
                                thinking_buffer = ""

                            # Strip only leading/trailing whitespace, preserve internal spaces
                            clean_content = content.strip(' \t\n\r\x0c')

                            if not started_content:
                                print("\n[--- Response ---]", file=sys.stdout)
                                started_content = True

                            sys.stdout.write(clean_content)
                            sys.stdout.flush()
                            full_content += clean_content

                    except json.JSONDecodeError:
                        continue

            # === Check for completion signal ===
            #if finish_reason and choices:
            #    break

            # === Display buffered thinking at end (if any remains) ===
            if thinking_buffer.strip():
                sys.stderr.write(f"\n<thinking>{thinking_buffer}</thinking>\n")

            # === Calculate and print final stats ===
            total_time = time.time() - start_time
            usage = self.calculate_stats(total_time, full_content, completion_data)
            self.print_stats_display(usage)

            return full_content

        except Exception as e:
            sys.stderr.write(f"\n[ERROR] Llama.cpp streaming failed: {e}\n", file=sys.stderr)
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
        self.commands = [
            '/?', '/help', '/listmodel', '/switchmodel',
            '/cwd', '/ls', '/curl', '/spawnshell', '/clear',
            '/thinkingon', '/thinkingoff', '/contextsizeset',
            '/debug on', '/debug off', '/quit', '/exit'
        ]
        self.models = []

    def fetch_models(self):
        """Fetch available models from the backend."""
        if self.backend == "llamacpp":
            self.models = [m['name'] for m in fetch_models_llamacpp(self.base_url)]
        else:
            self.models = [m['name'] for m in fetch_models_ollama(self.base_url)]


# ============================================================================
# ============= INPUT HANDLING CLASS =========================================
# ============================================================================

def gather_user_input(model, show_multiline=True):
    """
    Gather user input with multiline support.

    Args:
        model (str): Current model name for prompt display
        show_multiline (bool): Whether to enable multiline mode

    Returns:
        str or None: User input or None if cancelled/EOF
    """
    ctrl_c_count = 0
    ctrl_d_count = 0

    while True:
        try:
            # Use readline if available, fallback to input()
            if READLINE_AVAILABLE:
                prompt_str = f"you@{model} > "
                line = input(prompt_str)
                ctrl_c_count = 0
            else:
                prompt_str = f"you@{model}: "
                line = input(prompt_str)

            if not show_multiline or line.strip() != '"""':
                return line

        except KeyboardInterrupt:
            if not show_multiline:
                continue

            ctrl_c_count += 1
            if ctrl_c_count >= 2:
                print(f"\n[Cancelled]", file=sys.stderr)
                return None
            print(f"\n(Press Ctrl+C again to exit)", file=sys.stderr)

        except EOFError:
            print(f"\n[EOF received, one more and it exit]", file=sys.stderr)
            ctrl_d_count += 1
            if ctrl_d_count >= 2:
                print(f"\n[Exiting]", file=sys.stderr)
                sys.exit(1)



def process_inline_commands(full_input):
    """
    Process inline commands (!, /curl) within user input.

    Args:
        full_input (str): Raw user input

    Returns:
        str: Processed content with command outputs included
    """
    processed_lines = []
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
                text_content = fetch_and_convert_url(url)
                word_count = len(text_content.split()) if text_content else 0
                output_str = f"\n[Content from `{url}`]: {word_count} words"
                processed_lines.append(output_str)
            except Exception as e:
                output_str = f"[Failed to fetch URL: {e}]"
                processed_lines.append(output_str)

        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


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

        output_lines.extend(process.stdout.splitlines())
        output = "\n".join(output_lines) or "[Command executed successfully with no output]"

    except subprocess.TimeoutExpired:
        print("[Command timed out after 5 seconds!]", file=sys.stderr)
        output = "[Command execution interrupted: Time limit exceeded (5s)]"

    except Exception as e:
        print(f"[Failed to execute command: {e}]", file=sys.stderr)
        return f"[Execution error: {e}]"

    return f"\n[Command executed: `{command}`]\n```text\n{output.strip()}\n```\n"


def fetch_and_convert_url(url):
    """Fetch HTML from URL and convert to plain text."""
    try:
        html_bytes = get_html_bytes(url)
        if not html_bytes:
            return ""

        # Try html2text, pandoc, or lynx converters
        for converter in ['html2text', 'pandoc', 'lynx']:
            cmd = [converter, '-stdin'] if converter == 'lynx' else [converter]
            try:
                proc = subprocess.run(
                    cmd, input=html_bytes, capture_output=True,
                    timeout=10, check=False
                )
                return proc.stdout.decode('utf-8', errors='replace')
            except Exception:
                continue

        # Fallback: Simple HTML strip
        html_str = html_bytes.decode('utf-8', errors='ignore')
        return FallbackHTMLStripper().get_data(html_str)

    except Exception as e:
        return f"[Failed to fetch URL: {e}]"


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
        self.reset()
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
                        if cleaned and text is None:
                            self.outer_text.append(cleaned)

                def get_outer(self):
                    return ''.join(self.outer_text)

            parser = DataGatherer(skip_tags=self.skip_tags, current_tag=self.current_tag)
            parser.feed(text)

            return parser.get_outer() if parser.outer_text else ""
        return ''


# ============================================================================
# ============= STATISTICS OUTPUT ===========================================
# ============================================================================

def print_stats(total_time, chunk_or_result):
    """Print generation statistics to stderr."""
    try:
        eval_count = chunk_or_result.get("eval_count", 0) or \
                     chunk_or_result.get("usage", {}).get("completion_tokens", 0)

        tps = 0.0
        if eval_count > 0 and total_time > 0:
            tps = eval_count / total_time

        sys.stderr.write(f"\n[--- Stats: {total_time:.2f}s total | "
                         f"{tps:.2f} t/s ---]\n", flush=True)
    except Exception:
        pass


# ============================================================================
# ============= CHAT LOOP CLASS ============================================
# ============================================================================

class ChatLoop:
    """
    Unified chat loop that handles both Ollama and Llama.cpp backends.

    Features:
    - Interactive command-line interface
    - Model management (list/switch)
    - Context control
    - Debug/thinking modes
    - Shell integration
    - Persistent history
    """

    def __init__(self, base_url, backend, model="llama3", system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.base_url = base_url
        self.backend = backend  # "ollama" or "llamacpp"
        self.model = model
        self.system_prompt = system_prompt

        # Shared command handlers
        self.completer = ChatCompleter(base_url, backend)
        self.query_handler = ModelQuery(base_url, backend)

        # State management
        self.commands = [
            '/?', '/help', '/listmodel', '/switchmodel',
            '/cwd', '/ls', '/curl', '/spawnshell', '/clear',
            '/thinkingon', '/thinkingoff', '/contextsizeset',
            '/debug on', '/debug off', '/quit', '/exit'
        ]
        self.models = []
        self.context_size = None
        self.debug_mode = False
        self.force_no_thinking = False

    def fetch_models(self):
        """Fetch available models from the backend."""
        if self.backend == "llamacpp":
            self.models = [m['name'] for m in fetch_models_llamacpp(self.base_url)]
        else:
            self.models = [m['name'] for m in fetch_models_ollama(self.base_url)]

    def run(self, stream_enabled=True, debug=False, images=None):
        """Main chat loop - handles interactive session."""

        # Initialize model list
        self.fetch_models()

        # Setup readline completer if available
        if READLINE_AVAILABLE:
            try:
                readline.set_completer_delims(' \t\n')

                def completer(text, state):
                    buffer = readline.get_line_buffer()

                    if buffer.startswith('/switchmodel '):
                        matches = [m for m in self.models if m.startswith(text)]
                    elif buffer.startswith('/cwd ') or buffer.startswith('/ls '):
                        path = os.path.expanduser(text)
                        dirname = os.path.dirname(path)
                        basename = os.path.basename(path)
                        if not dirname: dirname = '.'

                        matches = []
                        try:
                            if os.path.exists(dirname) and os.path.isdir(dirname):
                                for item in os.listdir(dirname):
                                    if item.startswith(basename):
                                        full_path = os.path.join(dirname, item)
                                        prefix = os.path.dirname(text)
                                        if os.path.isdir(full_path):
                                            matches.append(os.path.join(prefix, item) + '/' if prefix else item + '/')
                                        elif not buffer.startswith('/cwd '):
                                            matches.append(os.path.join(prefix, item) if prefix else item)
                        except PermissionError:
                            pass
                    elif text.startswith('/') or text in ['e', 'ex', 'exi', 'q', 'qu', 'qui']:
                        matches = [c for c in self.commands if c.startswith(text)]

                    return matches[state] if state < len(matches) else None

                readline.set_completer(completer)
                readline.parse_and_bind('tab: complete')

                # Load history
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
                print(f"[ERROR] Readline setup failed: {e}", file=sys.stderr)

        messages = [{'role': 'system', 'content': self.system_prompt}]

        # Print welcome message
        print(f"\n[{self.backend.upper()} Chat Mode]", file=sys.stderr)
        print(f"Commands: /?, /help, /listmodel, /switchmodel", file=sys.stderr)
        print(f"          /cwd, /ls, /clear, /quit\n", file=sys.stderr)

        if images:
            print("[Image loaded for session]", file=sys.stderr)

        while True:
            try:
                full_input = gather_user_input(self.model)
                if full_input is None or not full_input.strip():
                    continue

                # Handle exit commands
                if full_input.lower() in ['exit', 'quit', '/exit', '/quit']:
                    print("\n[Goodbye!]", file=sys.stderr)
                    break

                # Handle help command
                if full_input in ['/?', '/help']:
                    print(f"\nCommands: {', '.join(self.commands[:6])}", file=sys.stderr)
                    print("  ! <cmd> - Execute shell", file=sys.stderr)
                    print("  /curl <url> - Fetch web content\n", file=sys.stderr)
                    continue

                # Handle model listing
                if full_input.startswith('/listmodel'):
                    parts = full_input.split(maxsplit=1)
                    self.list_models(parts[1] if len(parts) > 1 else None)
                    continue

                # Handle context size
                if full_input.startswith('/contextsizeset'):
                    self.set_context_size(full_input)
                    continue

                # Handle clear command
                if full_input == '/clear':
                    messages = [{'role': 'system', 'content': self.system_prompt}]
                    print("[Context memory wiped clean]", file=sys.stderr)
                    continue

                # Handle debug mode
                if full_input.lower() == '/debug on':
                    debug = True
                    debug_mode = True
                    continue
                elif full_input.lower() == '/debug off':
                    debug = False
                    continue

                # Handle thinking control
                if full_input == '/thinkingoff':
                    self.force_no_thinking = True
                    print("[Model will skip reasoning phase]", file=sys.stderr)
                    continue
                elif full_input == '/thinkingon':
                    self.force_no_thinking = False
                    print("[Reasoning phase enabled]", file=sys.stderr)
                    continue

                # Handle directory change
                if full_input.startswith('/cwd'):
                    parts = full_input.split(maxsplit=1)
                    if len(parts) > 1:
                        try:
                            os.chdir(os.path.expanduser(parts[1]))
                        except Exception as e:
                            print(f"[ERROR] {e}", file=sys.stderr)
                    print(f"[Current directory: {os.getcwd()}]", file=sys.stderr)
                    continue

                # Handle ls command
                if full_input.startswith('/ls'):
                    try:
                        subprocess.run("ls" + full_input[3:], shell=True, check=False)
                    except Exception as e:
                        print(f"[ERROR] {e}", file=sys.stderr)
                    continue

                # Handle model switch
                if full_input.startswith('/switchmodel'):
                    parts = full_input.split()
                    if len(parts) > 1:
                        self.model = parts[1]
                        print(f"[Switched to '{self.model}']", file=sys.stderr)
                    continue

                # Handle spawnshell command
                if full_input == '/spawnshell':
                    self.handle_spawnshell()
                    messages.append({'role': 'user', 'content': '[Shell session ended]'})
                    continue

                # Process inline commands and build message
                final_content = process_inline_commands(full_input)
                if not final_content.strip():
                    continue

                messages.append({'role': 'user', 'content': final_content})

                # Query model
                payload_messages = list(messages)

                # Add thinking suppression if requested
                if self.force_no_thinking:
                    payload_messages.append({
                        'role': 'system',
                        'content': 'Do NOT output reasoning or thoughts'
                    })

                # Execute query
                if messages[-1]['role'] == 'user':
                    response = self.query_handler.query_stream(
                        payload_messages,
                        self.model,
                        stream_enabled=stream_enabled,
                        debug=self.debug_mode,
                        show_thinking=(not self.force_no_thinking),
                        context_size=self.context_size,
                        images=images
                    )

                    if response:
                        messages.append({'role': 'assistant', 'content': response})
                        print()  # Add spacing after response

            except KeyboardInterrupt:
                print(f"\n[Interrupted]", file=sys.stderr)
                continue

            except Exception as e:
                if isinstance(e, EOFError):
                    print(f"[EOF - Goodbye!]", file=sys.stderr)
                    break
                else:
                    print(f"[ERROR] {e}", file=sys.stderr)

        return

    def list_models(self, filter_arg=None):
        """List available models from the backend."""
        if self.backend == "llamacpp":
            models = fetch_models_llamacpp(self.base_url)
        else:
            models = fetch_models_ollama(self.base_url)

        if not models:
            print(f"\n[No models found via {self.backend} API]", file=sys.stderr)
            return

        if filter_arg:
            search_term = None
            parts = filter_arg.lower().split()
            if 'name' in parts:
                parts.remove('name')

            if parts:
                search_term = parts[0]

        if search_term:
            models = [m for m in models if search_term in m.get('name', '').lower()]

        # Sort and display
        models.sort(key=lambda x: x.get('name', ''))

        print(f"\n{'NAME':<50} | {'SIZE' if self.backend == 'ollama' else 'OWNED BY'}")
        print("-" * 60)
        for m in models:
            size_str = f"{m['size_bytes'] / (1024**3):>8.2f} GB" if m.get('size_bytes', 0) > 0 else 'N/A'
            owned_by = m.get('owned_by', '')
            modified = m.get('modified_at', 'Unknown')[:10] if self.backend == 'ollama' else ''

            print(f"{m['name']:<50} | {size_str}{modified}")

        print()

    def set_context_size(self, full_input):
        """Set or reset context size."""
        parts = full_input.split()
        if len(parts) > 1 and parts[1].isdigit():
            val = int(parts[1])

            # Bounds checking
            if val == 0:
                self.context_size = None
                print("[Context size reset to default]", file=sys.stderr)
            else:
                if val > MAX_CONTEXT_SIZE:
                    print(f"[ERROR] Context size {val} exceeds maximum {MAX_CONTEXT_SIZE}",
                          file=sys.stderr)
                    return
                self.context_size = val
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


def list_models_ollama(base_url, filter_arg=None, include_capabilities=False):
    models = fetch_models_ollama(base_url)
    if not models:
        print(f"\n{YELLOW}No models found via Ollama API at {base_url}. Check if the server is running.{RESET}\n")
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
            print(f"\n{YELLOW}No models found matching '{search_term}'.{RESET}\n")
            return

    for m in models:
        try:
            m['size_bytes'] = int(m.get('size', 0))
        except (TypeError, ValueError):
            m['size_bytes'] = 0

    if sort_by == 'size':
        models.sort(key=lambda x: x['size_bytes'], reverse=True)
    else:
        models.sort(key=lambda x: x.get('name', ''))

    largest = max(models, key=lambda x: x['size_bytes']) if models else None
    if largest and largest['size_bytes'] > 0:
        l_size_gb = largest['size_bytes'] / (1024**3)
        print(f"\nChecking storage... Largest model in list: {largest['name']} ({l_size_gb:.2f} GB)\n")
    else:
        print()

    if include_capabilities:
        # Retrieve capabilities for each model (extra API calls)
        for m in models:
            try:
                info = fetch_model_info_ollama(base_url, m['name'])
                m['capabilities'] = ",".join(info.get('capabilities', []))
            except Exception:
                m['capabilities'] = ''
        header = f"{'NAME':<40} | {'SIZE':<10} | {'MODIFIED':<10} | {'CAPABILITIES'}"
        print(header)
        print("-" * 95)
        for m in models:
            size_str = f"{m['size_bytes'] / (1024**3):>8.2f} GB" if m['size_bytes'] > 0 else f"{'N/A':>11}"
            modified = m.get('modified_at', 'Unknown')[:10]
            caps = m.get('capabilities', '')
            print(f"{m['name']:<40} | {size_str} | {modified} | {caps}")
    else:
        print(f"{'NAME':<40} | {'SIZE':<10} | {'MODIFIED'}")
        print("-" * 75)
        for m in models:
            size_str = f"{m['size_bytes'] / (1024**3):>8.2f} GB" if m['size_bytes'] > 0 else f"{'N/A':>11}"
            modified = m.get('modified_at', 'Unknown')[:10]
            print(f"{m['name']:<40} | {size_str} | {modified}")
    print()



def fetch_model_info_ollama(base_url, model_name):
    """Fetch detailed model information from Ollama."""
    try:
        url = f"{base_url}/api/show"
        payload = json.dumps({"name": model_name}).encode('utf-8')
        req = Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return {}


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
    group.add_argument('-b', '--backend', choices=["ollama", "llamacpp"],
                      default="ollama", help='API backend (default: ollama)')
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

    parser.add_argument('-m', '--model', default="llama3",
                       help='Model name (default: llama3)')  # NOT mutually exclusive
    parser.add_argument('--prompt', default=DEFAULT_SYSTEM_PROMPT,
                       help='System prompt')  # Can be used with -m independently

    # Image support
    parser.add_argument('--image', help='Image file for multimodal models')

    # Display options
    parser.add_argument('-p', '--no-stream', action="store_true",
                       help='Disable streaming output')
    parser.add_argument('--debug', action="store_true",
                       help='Print raw JSON to stderr')
    parser.add_argument('--format', choices=["json", "yaml"], default="json",
                       help='Output format for model info (default: json)')

    # Parse arguments
    args = parser.parse_args()

    # Determine base URL
    backend = args.backend
    base_url = get_base_url(args, backend)

    # Handle listing operations
    if args.list or args.list_all:
        if args.backend == "llamacpp":
            list_models_llamacpp(base_url, filter_arg=args.model)
        if args.backend == "ollama":
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
        show_model_info(base_url, args.model, args)
    elif args.show_details:
        show_model_details(base_url, args.model, args)

    # Interactive chat mode
    if args.chat:
        image_data = prepare_image_data(args.image) if args.image else None
        images_list = [image_data] if image_data else None

        loop = ChatLoop(
            base_url=base_url,
            backend=backend,
            model=args.model,
            system_prompt=args.prompt
        )

        should_stream = not args.no_stream and sys.stdout.isatty()
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
                input_path = os.path.join(args.output_dir, filename)
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

                query_handler = ModelQuery(base_url, backend)

                response = query_handler.query_sync(
                    messages,
                    args.model,
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

            query_handler = ModelQuery(base_url, backend)
            should_stream = not args.no_stream and sys.stdout.isatty()

            response = query_handler.query_sync(
                messages,
                args.model,
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


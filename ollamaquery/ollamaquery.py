#!/usr/bin/python3
import os
import sys
import json
import argparse
import base64
# Optional YAML support – fall back to JSON if PyYAML not installed
try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

import urllib.request
import urllib.error
from urllib.parse import urljoin
import time
import subprocess
import threading
import re
import shutil
import atexit
import html
from html.parser import HTMLParser

# Handle Windows compatibility gracefully if readline isn't installed
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    readline = None
    READLINE_AVAILABLE = False

# Handle Unix PTY for the /spawnshell command
try:
    import pty
    PTY_AVAILABLE = True
except ImportError:
    pty = None
    PTY_AVAILABLE = False

# --- CONFIGURATION & COLORS ---
DEFAULT_SYSTEM_PROMPT = "You are a chatbot trying to help user. Try to rebound to the question as best as your knowledge goes but reply politely that you don't know if it is the case."
DARK_GRAY = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32;1m"
YELLOW = "\033[33;1m"
RESET = "\033[0m"

def prepare_image_data(image_path):
    """Reads an image file and returns its base64 encoded string."""
    if not image_path:
        return None
    try:
        with open(image_path, "rb") as img_file:
            #data = img_file.read()
            #data_b64_utf8=base64.b64encode(data).encode('utf-8')
            #print(data_b64_utf8)
            #return data_b64_utf8
            #return base64.b64encode(img_file.read().encode('utf-8'))
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        sys.stderr.write(f"{YELLOW}Error loading image {image_path}: {e}{RESET}\n")
        return None

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

class FallbackHTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
        self.skip_tags = {'script', 'style', 'head', 'meta', 'title', 'link', 'noscript'}
        self.current_tag = ""

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        self.current_tag = ""

    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            cleaned = data.strip()
            if cleaned:
                self.text.append(cleaned)

    def get_data(self):
        return '\n'.join(self.text)

def get_html_bytes(url, depth=0):
    """Fetches HTML bytes while gracefully following meta-refresh redirects."""
    if depth > 3:
        return b""
        
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    html_bytes = b""
    
    try:
        # Prefer system curl/wget as they handle SSL and HTTP redirects better
        if shutil.which('curl'):
            proc = subprocess.run(['curl', '-L', '-s', '-A', headers['User-Agent'], url], capture_output=True, timeout=15)
            html_bytes = proc.stdout
        elif shutil.which('wget'):
            proc = subprocess.run(['wget', '-qO-', '-U', headers['User-Agent'], url], capture_output=True, timeout=15)
            html_bytes = proc.stdout
        else:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html_bytes = response.read()
    except Exception as e:
        print(f"{YELLOW}Warning: HTTP fetch failed: {e}{RESET}")
        return b""
        
    html_str = html_bytes.decode('utf-8', errors='ignore')
    
    # Catch HTML client-side redirects (e.g. Google News)
    match = re.search(r'(?i)<meta[^>]+http-equiv=["\']?refresh["\']?[^>]+content=["\']?\d+;\s*url=([^"\'>]+)["\']?', html_str)
    if not match: # Sometimes attributes are reversed
        match = re.search(r'(?i)<meta[^>]+content=["\']?\d+;\s*url=([^"\'>]+)["\']?[^>]+http-equiv=["\']?refresh["\']?', html_str)
        
    if match:
        next_url = match.group(1).strip()
        next_url = html.unescape(next_url)
        next_url = urljoin(url, next_url)
        print(f"{DARK_GRAY}Following redirect to: {next_url}{RESET}")
        return get_html_bytes(next_url, depth + 1)
        
    return html_bytes

def fetch_and_convert_url(url):
    html_bytes = get_html_bytes(url)
    if not html_bytes:
        return ""

    if shutil.which('html2text'):
        proc = subprocess.run(['html2text'], input=html_bytes, capture_output=True)
        return proc.stdout.decode('utf-8', errors='replace')
    elif shutil.which('pandoc'):
        proc = subprocess.run(['pandoc', '-f', 'html', '-t', 'plain'], input=html_bytes, capture_output=True)
        return proc.stdout.decode('utf-8', errors='replace')
    elif shutil.which('lynx'):
        proc = subprocess.run(['lynx', '-dump', '-stdin'], input=html_bytes, capture_output=True)
        return proc.stdout.decode('utf-8', errors='replace')

    html_str = html_bytes.decode('utf-8', errors='ignore')
    stripper = FallbackHTMLStripper()
    stripper.feed(html_str)
    return stripper.get_data()

def fetch_models_ollama(base_url):
    try:
        url = f"{base_url}/api/tags"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('models', [])
    except Exception:
        return []

def fetch_model_info_ollama(base_url, model_name):
    """Fetch detailed model information via Ollama /api/show endpoint.
    
    Returns a dict with model metadata or empty dict on failure.
    """
    try:
        url = f"{base_url}/api/show"
        payload = json.dumps({"name": model_name}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data
    except Exception:
        return {}

    try:
        url = f"{base_url}/api/tags"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('models', [])
    except Exception:
        return []

def fetch_models_llamacpp(base_url):
    try:
        url = f"{base_url}/v1/models"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode('utf-8'))
            return [{"name": m.get("id", "unknown"), "owned_by": m.get("owned_by", "N/A")} for m in data.get('data', [])]
    except Exception:
        return []

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



def list_models_llamacpp(base_url, filter_arg=None):
    models = fetch_models_llamacpp(base_url)
    if not models:
        print(f"\n{YELLOW}No models found via llama.cpp API at {base_url}. Check if the server is running.{RESET}\n")
        return

    search_term = None
    if filter_arg:
        parts = filter_arg.lower().split()
        if 'name' in parts: parts.remove('name')
        if parts: search_term = parts[0]

    if search_term:
        models = [m for m in models if search_term in m['name'].lower()]
        if not models:
            print(f"\n{YELLOW}No models found matching '{search_term}'.{RESET}\n")
            return

    models.sort(key=lambda x: x.get('name', ''))

    print()
    print(f"{'NAME':<50} | {'OWNED BY'}")
    print("-" * 75)
    for m in models:
        print(f"{m['name']:<50} | {m.get('owned_by', 'N/A')}")
    print()

def print_stats(total_time, chunk_or_result):
    eval_count = chunk_or_result.get("eval_count", 0)
    eval_duration_ns = chunk_or_result.get("eval_duration", 0)
    prompt_eval_count = chunk_or_result.get("prompt_eval_count", 0)
    
    usage = chunk_or_result.get("usage", {})
    if not eval_count and usage:
        eval_count = usage.get("completion_tokens", 0)
        prompt_eval_count = usage.get("prompt_tokens", 0)
        
    timings = chunk_or_result.get("timings", {})
    if not eval_count and timings:
        eval_count = timings.get("predicted_n", 0)
        prompt_eval_count = timings.get("prompt_n", 0)
        eval_duration_ns = timings.get("predicted_ms", 0) * 1_000_000 
        
    total_context = prompt_eval_count + eval_count
    
    tps = 0.0
    if eval_count > 0:
        if eval_duration_ns > 0:
            tps = eval_count / (eval_duration_ns / 1e9)
        elif total_time > 0:
            tps = eval_count / total_time
            
    sys.stderr.write(f"\n{DARK_GRAY}--- Stats: {total_time:.2f}s total | {tps:.2f} t/s | Context: {total_context} tokens ---{RESET}\n")
    sys.stderr.flush()

# ==========================================
# AUTOCOMPLETE & INPUT GATHERING
# ==========================================

class ChatCompleter:
    def __init__(self, base_url, backend):
        self.base_url = base_url
        self.backend = backend
        self.commands = [
            '/?', '/help', '/listmodel', '/switchmodel', 
            '/cwd', '/ls', '/curl', '/spawnshell', '/clear', 
            '/thinkingon', '/thinkingoff', '/contextsizeset',
            '/debug on', '/debug off', '/quit', '/exit', 'exit', 'quit'
        ]
        self.models = []

    def fetch_models(self):
        if self.backend == "llamacpp":
            self.models = [m['name'] for m in fetch_models_llamacpp(self.base_url)]
        else:
            self.models = [m['name'] for m in fetch_models_ollama(self.base_url)]

    def complete(self, text, state):
        buffer = readline.get_line_buffer() if readline else ''
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
            except PermissionError: pass 
        elif text.startswith('/') or text in ['e', 'ex', 'exi', 'q', 'qu', 'qui']:
            matches = [c for c in self.commands if c.startswith(text)]
        else:
            matches = []

        return matches[state] if state < len(matches) else None

def setup_readline(base_url, backend):
    if READLINE_AVAILABLE and readline is not None:
        completer = ChatCompleter(base_url, backend)
        completer.fetch_models()
        readline.set_completer_delims(' \t\n')
        readline.set_completer(completer.complete)
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
        atexit.register(readline.write_history_file, histfile)
        
        print(f"{DARK_GRAY}[Tab Autocomplete & Persistent History Enabled]{RESET}\n")
    else:
        print()


def gather_user_input(model):
    user_input_lines = []
    in_multiline = False
    ctrl_c_count = 0
    
    if READLINE_AVAILABLE:
        p_yellow = f"\001{YELLOW}\002"
        p_reset = f"\001{RESET}\002"
    else:
        p_yellow = YELLOW
        p_reset = RESET
    
    while True:
        prompt_str = f"{p_yellow}you@{model} > {p_reset}" if not in_multiline else f"{p_yellow}... > {p_reset}"
        try:
            line = input(prompt_str)
            ctrl_c_count = 0 
        except KeyboardInterrupt:
            if in_multiline:
                print(f"\n{YELLOW}[Multiline input cancelled]{RESET}")
                return ""
            else:
                ctrl_c_count += 1
                if ctrl_c_count >= 2:
                    return None
                print(f"\n{YELLOW}(Press Ctrl+C again to exit){RESET}")
                return ""
        except EOFError:
            return None
        
        if line.strip() == '"""':
            if not in_multiline:
                in_multiline = True
                continue
            else:
                in_multiline = False
                break
        
        user_input_lines.append(line)
        if not in_multiline:
            break
            
    return "\n".join(user_input_lines)

# ==========================================
# INLINE COMMAND PROCESSING (SHELL & CURL)
# ==========================================

def execute_os_command(command):
    print(f"{DARK_GRAY}--- Executing (max 5s): {command} ---{RESET}")
    output_lines = []
    
    def read_output(pipe):
        for out_line in pipe:
            sys.stdout.write(out_line)
            sys.stdout.flush()
            output_lines.append(out_line)
    
    try:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        
        reader_thread = threading.Thread(target=read_output, args=(process.stdout,))
        reader_thread.daemon = True
        reader_thread.start()
        
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.kill()
            sys.stdout.write(f"\n{YELLOW}[Command timed out after 5 seconds!]{RESET}\n")
            output_lines.append("\n[Command execution interrupted: Time limit exceeded (5s)]\n")
        except KeyboardInterrupt:
            process.terminate()
            process.kill()
            sys.stdout.write(f"\n{YELLOW}[Command Interrupted by User]{RESET}\n")
            output_lines.append("\n[Command execution interrupted by user]\n")
        
        reader_thread.join(timeout=0.2)
        output = "".join(output_lines)
        if not output.strip():
            output = "[Command executed successfully with no output]"
        
        print(f"{DARK_GRAY}--- Execution finished ---{RESET}")
        return f"\n[Command executed: `{command}`]\n```text\n{output.strip()}\n```\n"
        
    except Exception as e:
        print(f"{CYAN}Failed to execute command: {e}{RESET}")
        return f"[Failed to execute `{command}`: {e}]"

def process_inline_commands(full_input):
    processed_lines = []
    for line in full_input.split('\n'):
        stripped_line = line.lstrip()
        
        if stripped_line.startswith("!"):
            command = stripped_line[1:].strip()
            if not command:
                processed_lines.append(line)
                continue
            output_str = execute_os_command(command)
            processed_lines.append(output_str)
            
        elif stripped_line.startswith("/curl "):
            url = stripped_line[6:].strip()
            if not url:
                processed_lines.append(line)
                continue
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            print(f"{DARK_GRAY}Fetching and converting {url}...{RESET}")
            try:
                text_content = fetch_and_convert_url(url)
                if text_content:
                    word_count = len(text_content.split())
                    print(f"{GREEN}Successfully fetched {len(text_content)} characters ({word_count} words).{RESET}")
                    output_str = f"\n[Content fetched from `{url}`]:\n```text\n{text_content}\n```\n"
                    processed_lines.append(output_str)
                else:
                    print(f"{YELLOW}No text could be extracted from the URL.{RESET}")
                    processed_lines.append(f"[Failed to extract text from `{url}`]")
            except Exception as e:
                print(f"{YELLOW}Failed to fetch URL: {e}{RESET}")
                processed_lines.append(f"[Failed to fetch `{url}`: {e}]")
                
        else:
            processed_lines.append(line)
            
    return "\n".join(processed_lines)

def handle_spawnshell():
    if not PTY_AVAILABLE:
        print(f"{YELLOW}The /spawnshell command is only available on Unix-like systems.{RESET}\n")
        return None
    
    print(f"{GREEN}Spawning interactive shell. Type 'exit' or press Ctrl+D to return to the chat.{RESET}")
    captured_bytes = bytearray()
    
    def read_and_capture(fd):
        try:
            data = os.read(fd, 1024)
            captured_bytes.extend(data)
            return data
        except OSError:
            return b""
    
    shell_cmd = os.environ.get('SHELL', '/bin/bash')
    try:
        pty.spawn(shell_cmd, read_and_capture) if pty else None
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"{YELLOW}Shell exited with error: {e}{RESET}")
    
    raw_output = captured_bytes.decode('utf-8', errors='replace')
    clean_output = strip_ansi(raw_output).strip()
    
    if not clean_output:
        print(f"{DARK_GRAY}No output captured.{RESET}\n")
        return None
    
    print(f"{DARK_GRAY}--- Shell session captured ({len(clean_output)} chars) ---{RESET}")
    return f"I spawned an interactive shell, executed some commands, and then exited. Here is the transcript of my session:\n```text\n{clean_output}\n```\nPlease analyze this session and provide any relevant insights, or simply acknowledge it if no action is needed."


# ==========================================
# BACKEND: OLLAMA
# ==========================================

def query_ollama(base_url, messages, model, stream_enabled=False, debug=False, show_thinking=True, context_size=None, images=None):
    start_time = time.time()
    full_content = ""
    try:
        # Try to insert the image into messages
        #     "messages": [{
        #                      "role": "user",
        #                      "content": "What is in this image?",
        #                      "images": ["'"$IMG"'"]
        #                  }],

        if images:
            messages[1]["images"] = images
        payload = {"model": model, "messages": messages, "stream": stream_enabled}
        if context_size is not None:
            payload["options"] = {"num_ctx": context_size}
        
        api_url = f"{base_url}/api/chat"
        data = json.dumps(payload).encode('utf-8')
        if debug:
            print(json.dumps(payload,indent=2))
        req = urllib.request.Request(api_url, data=data, headers={'Content-Type': 'application/json'})

        with urllib.request.urlopen(req) as response:
            if stream_enabled:
                started_content = False
                for line in response:
                    decoded_line = line.decode('utf-8').strip()
                    if not decoded_line: continue
                    if debug: sys.stderr.write(f"\n{CYAN}[DEBUG RAW]: {decoded_line}{RESET}\n")
                    
                    chunk = json.loads(decoded_line)
                    msg = chunk.get("message", {})
                    
                    thought = msg.get("thought") or msg.get("thinking") or ""
                    if thought and show_thinking:
                        sys.stderr.write(f"{DARK_GRAY}{thought}{RESET}")
                        sys.stderr.flush()
                    
                    content = msg.get("content", "")
                    if content:
                        if not started_content:
                            sys.stdout.write(f"\n{GREEN}--- Response ---{RESET}\n")
                            sys.stdout.flush()
                            started_content = True
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        full_content += content
                    
                    if chunk.get("done"):
                        sys.stdout.write("\n")
                        print_stats(time.time() - start_time, chunk)
                return full_content
            else:
                result = json.loads(response.read().decode('utf-8'))
                print_stats(time.time() - start_time, result)
                return result['message'].get('content', '')
                
    except KeyboardInterrupt:
        sys.stdout.write(f"\n\n{YELLOW}[Generation Interrupted by User]{RESET}\n")
        sys.stdout.flush()
        return full_content 
    except Exception as e:
        sys.stderr.write(f"\n{YELLOW}query_ollama Request Error: {e}{RESET}\n")
    return full_content

def chat_loop_ollama(base_url, system_prompt, initial_model, stream_enabled, debug, images=None):
      model = initial_model
      force_no_thinking = False
      context_size = None
    
      print(f"{CYAN}Entering Llama.cpp chat mode. Type '/?' for help.{RESET}")
      setup_readline(base_url, "llamacpp")
      messages = [{'role': 'system', 'content': system_prompt}]
    
      if images:
        print(f"{GREEN}Image loaded for session.{RESET}")

      while True:
        try:
            full_input = gather_user_input(model)
            if full_input is None:
                print(f"\n{CYAN}Goodbye!{RESET}")
                break
            
            full_input_stripped = full_input.strip()
            if not full_input_stripped: continue
            
            # Pure local commands
            if full_input_stripped.lower() in ['exit', 'quit', '/exit', '/quit']:
                print(f"{CYAN}Goodbye!{RESET}"); break
            elif full_input_stripped in ['/?', '/help']:
                print(f"\n{CYAN}Commands:{RESET} /listmodel, /switchmodel, /contextsizeset <int>, /cwd, /ls, /clear, /thinkingoff, /thinkingon, /debug on, /debug off, /quit\n")
                print(f"  {YELLOW}! <command>{RESET}        - Execute a shell command inline")
                print(f"  {YELLOW}/curl <url>{RESET}        - Fetch and extract text from a URL inline")
                print(f"  {YELLOW}/spawnshell{RESET}        - Capture a full interactive shell session\n")
                continue
            elif full_input_stripped.startswith('/listmodel'):
                parts = full_input_stripped.split(maxsplit=1)
                list_models_ollama(base_url, parts[1] if len(parts) > 1 else None)
                continue
            elif full_input_stripped.startswith('/contextsizeset'):
                parts = full_input_stripped.split()
                if len(parts) > 1 and parts[1].isdigit():
                    val = int(parts[1])
                    if val == 0:
                        context_size = None
                        print(f"{GREEN}Context size reset to default.{RESET}\n")
                    else:
                        context_size = val
                        print(f"{GREEN}Context size set to {context_size}.{RESET}\n")
                else:
                    print(f"{YELLOW}Usage: /contextsizeset <integer> (use 0 for default){RESET}\n")
                continue
            elif full_input_stripped == '/clear':
                messages = [{'role': 'system', 'content': system_prompt}]
                print(f"{GREEN}Context memory wiped clean.{RESET}\n")
                continue
            elif full_input_stripped.lower() == '/debug on':
                debug = True
                print(f"{GREEN}Debug mode ENABLED.{RESET}\n")
                continue
            elif full_input_stripped.lower() == '/debug off':
                debug = False
                print(f"{GREEN}Debug mode DISABLED.{RESET}\n")
                continue
            elif full_input_stripped == '/thinkingoff':
                force_no_thinking = True
                print(f"{GREEN}Instructing model to SKIP the reasoning phase.{RESET}\n")
                continue
            elif full_input_stripped == '/thinkingon':
                force_no_thinking = False
                print(f"{GREEN}Allowing model to use reasoning phase normally.{RESET}\n")
                continue
            elif full_input_stripped.startswith('/cwd'):
                parts = full_input_stripped.split(maxsplit=1)
                if len(parts) > 1:
                    try: os.chdir(os.path.expanduser(parts[1]))
                    except Exception as e: print(f"{YELLOW}Error: {e}{RESET}")
                print(f"{GREEN}Directory: {os.getcwd()}{RESET}\n")
                continue
            elif full_input_stripped.startswith('/ls'):
                try: subprocess.run("ls " + full_input_stripped[3:], shell=True)
                except Exception as e: print(f"{YELLOW}Error: {e}{RESET}")
                print()
                continue
            elif full_input_stripped.startswith('/switchmodel'):
                parts = full_input_stripped.split()
                if len(parts) > 1:
                    model = parts[1]
                    print(f"{GREEN}Switched model to '{model}'.{RESET}\n")
                continue
            
            # Commands that interact with the LLM
            elif full_input_stripped == '/spawnshell':
                shell_msg = handle_spawnshell()
                if not shell_msg: continue
                messages.append({'role': 'user', 'content': shell_msg})
            else:
                final_user_content = process_inline_commands(full_input)
                if not final_user_content.strip(): continue
                messages.append({'role': 'user', 'content': final_user_content})
            
            payload_messages = list(messages)
            if force_no_thinking:
                payload_messages.append({'role': 'system', 'content': 'CRITICAL INSTRUCTION: Do NOT output any internal thoughts, <think> tags, or reasoning. Output the final answer directly and immediately.'})
            
            if messages[-1]['role'] == 'user':
                assistant_response = query_ollama(base_url, payload_messages, model, stream_enabled, debug, images=images)
                if assistant_response:
                    messages.append({'role': 'assistant', 'content': assistant_response})
                print() 

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                sys.stdout.write(f"\n{YELLOW}[Action Interrupted]{RESET}\n")
                continue
            elif isinstance(e, EOFError):
                print(f"\n{CYAN}Goodbye!{RESET}")
                break
            else:
                raise e
            
            full_input_stripped = full_input.strip()
            if not full_input_stripped: continue
            
            # Pure local commands
            if full_input_stripped.lower() in ['exit', 'quit', '/exit', '/quit']:
                print(f"{CYAN}Goodbye!{RESET}"); break
            elif full_input_stripped in ['/?', '/help']:
                print(f"\n{CYAN}Commands:{RESET} /listmodel, /switchmodel, /contextsizeset <int>, /cwd, /ls, /clear, /thinkingoff, /thinkingon, /debug on, /debug off, /quit\n")
                print(f"  {YELLOW}! <command>{RESET}        - Execute a shell command inline")
                print(f"  {YELLOW}/curl <url>{RESET}        - Fetch and extract text from a URL inline")
                print(f"  {YELLOW}/spawnshell{RESET}        - Capture a full interactive shell session\n")
                continue
            elif full_input_stripped.startswith('/listmodel'):
                parts = full_input_stripped.split(maxsplit=1)
                list_models_ollama(base_url, parts[1] if len(parts) > 1 else None)
                continue
            elif full_input_stripped.startswith('/contextsizeset'):
                parts = full_input_stripped.split()
                if len(parts) > 1 and parts[1].isdigit():
                    val = int(parts[1])
                    if val == 0:
                        context_size = None
                        print(f"{GREEN}Context size reset to default.{RESET}\n")
                    else:
                        context_size = val
                        print(f"{GREEN}Context size set to {context_size}.{RESET}\n")
                else:
                    print(f"{YELLOW}Usage: /contextsizeset <integer> (use 0 for default){RESET}\n")
                continue
            elif full_input_stripped == '/clear':
                messages = [{'role': 'system', 'content': system_prompt}]
                print(f"{GREEN}Context memory wiped clean.{RESET}\n")
                continue
            elif full_input_stripped.lower() == '/debug on':
                debug = True
                print(f"{GREEN}Debug mode ENABLED.{RESET}\n")
                continue
            elif full_input_stripped.lower() == '/debug off':
                debug = False
                print(f"{GREEN}Debug mode DISABLED.{RESET}\n")
                continue
            elif full_input_stripped == '/thinkingoff':
                force_no_thinking = True
                print(f"{GREEN}Instructing model to SKIP the reasoning phase.{RESET}\n")
                continue
            elif full_input_stripped == '/thinkingon':
                force_no_thinking = False
                print(f"{GREEN}Allowing model to use reasoning phase normally.{RESET}\n")
                continue
            elif full_input_stripped.startswith('/cwd'):
                parts = full_input_stripped.split(maxsplit=1)
                if len(parts) > 1:
                    try: os.chdir(os.path.expanduser(parts[1]))
                    except Exception as e: print(f"{YELLOW}Error: {e}{RESET}")
                print(f"{GREEN}Directory: {os.getcwd()}{RESET}\n")
                continue
            elif full_input_stripped.startswith('/ls'):
                try: subprocess.run("ls " + full_input_stripped[3:], shell=True)
                except Exception as e: print(f"{YELLOW}Error: {e}{RESET}")
                print()
                continue
            elif full_input_stripped.startswith('/switchmodel'):
                parts = full_input_stripped.split()
                if len(parts) > 1:
                    model = parts[1]
                    print(f"{GREEN}Switched model to '{model}'.{RESET}\n")
                continue
            
            # Commands that interact with the LLM
            elif full_input_stripped == '/spawnshell':
                shell_msg = handle_spawnshell()
                if not shell_msg: continue
                messages.append({'role': 'user', 'content': shell_msg})
            else:
                final_user_content = process_inline_commands(full_input)
                if not final_user_content.strip(): continue
                messages.append({'role': 'user', 'content': final_user_content})
            
            payload_messages = list(messages)
            if force_no_thinking:
                payload_messages.append({'role': 'system', 'content': 'CRITICAL INSTRUCTION: Do NOT output any internal thoughts, <think> tags, or reasoning. Output the final answer directly and immediately.'})
            
            if messages[-1]['role'] == 'user':
                assistant_response = query_llamacpp(base_url, payload_messages, model, stream_enabled, debug, images=images)
                if assistant_response:
                    messages.append({'role': 'assistant', 'content': assistant_response})
                print() 

        except KeyboardInterrupt:
            sys.stdout.write(f"\n{YELLOW}[Action Interrupted]{RESET}\n")
            continue
        except EOFError:
            print(f"\n{CYAN}Goodbye!{RESET}")
            break
            
            full_input_stripped = full_input.strip()
            if not full_input_stripped: continue
            
            # Pure local commands
            if full_input_stripped.lower() in ['exit', 'quit', '/exit', '/quit']:
                print(f"{CYAN}Goodbye!{RESET}"); break
            elif full_input_stripped in ['/?', '/help']:
                print(f"\n{CYAN}Commands:{RESET} /listmodel, /switchmodel, /contextsizeset <int>, /cwd, /ls, /clear, /thinkingoff, /thinkingon, /debug on, /debug off, /quit\n")
                print(f"  {YELLOW}! <command>{RESET}        - Execute a shell command inline")
                print(f"  {YELLOW}/curl <url>{RESET}        - Fetch and extract text from a URL inline")
                print(f"  {YELLOW}/spawnshell{RESET}        - Capture a full interactive shell session\n")
                continue
            elif full_input_stripped.startswith('/listmodel'):
                parts = full_input_stripped.split(maxsplit=1)
                list_models_ollama(base_url, parts[1] if len(parts) > 1 else None)
                continue
            elif full_input_stripped.startswith('/contextsizeset'):
                parts = full_input_stripped.split()
                if len(parts) > 1 and parts[1].isdigit():
                    val = int(parts[1])
                    if val == 0:
                        context_size = None
                        print(f"{GREEN}Context size reset to default.{RESET}\n")
                    else:
                        context_size = val
                        print(f"{GREEN}Context size set to {context_size}.{RESET}\n")
                else:
                    print(f"{YELLOW}Usage: /contextsizeset <integer> (use 0 for default){RESET}\n")
                continue
            elif full_input_stripped == '/clear':
                messages = [{'role': 'system', 'content': system_prompt}]
                print(f"{GREEN}Context memory wiped clean.{RESET}\n")
                continue
            elif full_input_stripped.lower() == '/debug on':
                debug = True
                print(f"{GREEN}Debug mode ENABLED.{RESET}\n")
                continue
            elif full_input_stripped.lower() == '/debug off':
                debug = False
                print(f"{GREEN}Debug mode DISABLED.{RESET}\n")
                continue
            elif full_input_stripped == '/thinkingoff':
                force_no_thinking = True
                print(f"{GREEN}Instructing model to SKIP the reasoning phase.{RESET}\n")
                continue
            elif full_input_stripped == '/thinkingon':
                force_no_thinking = False
                print(f"{GREEN}Allowing model to use reasoning phase normally.{RESET}\n")
                continue
            elif full_input_stripped.startswith('/cwd'):
                parts = full_input_stripped.split(maxsplit=1)
                if len(parts) > 1:
                    try: os.chdir(os.path.expanduser(parts[1]))
                    except Exception as e: print(f"{YELLOW}Error: {e}{RESET}")
                print(f"{GREEN}Directory: {os.getcwd()}{RESET}\n")
                continue
            elif full_input_stripped.startswith('/ls'):
                try: subprocess.run("ls " + full_input_stripped[3:], shell=True)
                except Exception as e: print(f"{YELLOW}Error: {e}{RESET}")
                print()
                continue
            elif full_input_stripped.startswith('/switchmodel'):
                parts = full_input_stripped.split()
                if len(parts) > 1:
                    model = parts[1]
                    print(f"{GREEN}Switched model to '{model}'.{RESET}\n")
                continue
            
            # Commands that interact with the LLM
            elif full_input_stripped == '/spawnshell':
                shell_msg = handle_spawnshell()
                if not shell_msg: continue
                messages.append({'role': 'user', 'content': shell_msg})
            else:
                final_user_content = process_inline_commands(full_input)
                if not final_user_content.strip(): continue
                messages.append({'role': 'user', 'content': final_user_content})
            
            payload_messages = list(messages)
            if force_no_thinking:
                payload_messages.append({'role': 'system', 'content': 'CRITICAL INSTRUCTION: Do NOT output any internal thoughts, <think> tags, or reasoning. Output the final answer directly and immediately.'})
            
            if messages[-1]['role'] == 'user':
                assistant_response = query_llamacpp(base_url, payload_messages, model, stream_enabled, debug, images=images)
                if assistant_response:
                    messages.append({'role': 'assistant', 'content': assistant_response})
                print() 

        except KeyboardInterrupt:
            sys.stdout.write(f"\n{YELLOW}[Action Interrupted]{RESET}\n")
            continue
        except EOFError:
            print(f"\n{CYAN}Goodbye!{RESET}")
            break
            
            full_input_stripped = full_input.strip()
            if not full_input_stripped: continue
            
            # Pure local commands
            if full_input_stripped.lower() in ['exit', 'quit', '/exit', '/quit']:
                print(f"{CYAN}Goodbye!{RESET}"); break
            elif full_input_stripped in ['/?', '/help']:
                print(f"\n{CYAN}Commands:{RESET} /listmodel, /switchmodel, /contextsizeset <int>, /cwd, /ls, /clear, /thinkingoff, /thinkingon, /debug on, /debug off, /quit\n")
                print(f"  {YELLOW}! <command>{RESET}        - Execute a shell command inline")
                print(f"  {YELLOW}/curl <url>{RESET}        - Fetch and extract text from a URL inline")
                print(f"  {YELLOW}/spawnshell{RESET}        - Capture a full interactive shell session\n")
                continue
            elif full_input_stripped.startswith('/listmodel'):
                parts = full_input_stripped.split(maxsplit=1)
                list_models_ollama(base_url, parts[1] if len(parts) > 1 else None)
                continue
            elif full_input_stripped.startswith('/contextsizeset'):
                parts = full_input_stripped.split()
                if len(parts) > 1 and parts[1].isdigit():
                    val = int(parts[1])
                    if val == 0:
                        context_size = None
                        print(f"{GREEN}Context size reset to default.{RESET}\n")
                    else:
                        context_size = val
                        print(f"{GREEN}Context size set to {context_size}.{RESET}\n")
                else:
                    print(f"{YELLOW}Usage: /contextsizeset <integer> (use 0 for default){RESET}\n")
                continue
            elif full_input_stripped == '/clear':
                messages = [{'role': 'system', 'content': system_prompt}]
                print(f"{GREEN}Context memory wiped clean.{RESET}\n")
                continue
            elif full_input_stripped.lower() == '/debug on':
                debug = True
                print(f"{GREEN}Debug mode ENABLED.{RESET}\n")
                continue
            elif full_input_stripped.lower() == '/debug off':
                debug = False
                print(f"{GREEN}Debug mode DISABLED.{RESET}\n")
                continue
            elif full_input_stripped == '/thinkingoff':
                force_no_thinking = True
                print(f"{GREEN}Instructing model to SKIP the reasoning phase.{RESET}\n")
                continue
            elif full_input_stripped == '/thinkingon':
                force_no_thinking = False
                print(f"{GREEN}Allowing model to use reasoning phase normally.{RESET}\n")
                continue
            elif full_input_stripped.startswith('/cwd'):
                parts = full_input_stripped.split(maxsplit=1)
                if len(parts) > 1:
                    try: os.chdir(os.path.expanduser(parts[1]))
                    except Exception as e: print(f"{YELLOW}Error: {e}{RESET}")
                print(f"{GREEN}Directory: {os.getcwd()}{RESET}\n")
                continue
            elif full_input_stripped.startswith('/ls'):
                try: subprocess.run("ls " + full_input_stripped[3:], shell=True)
                except Exception as e: print(f"{YELLOW}Error: {e}{RESET}")
                print()
                continue
            elif full_input_stripped.startswith('/switchmodel'):
                parts = full_input_stripped.split()
                if len(parts) > 1:
                    model = parts[1]
                    print(f"{GREEN}Switched model to '{model}'.{RESET}\n")
                continue
            
            # Commands that interact with the LLM
            elif full_input_stripped == '/spawnshell':
                shell_msg = handle_spawnshell()
                if not shell_msg: continue
                messages.append({'role': 'user', 'content': shell_msg})
            else:
                final_user_content = process_inline_commands(full_input)
                if not final_user_content.strip(): continue
                messages.append({'role': 'user', 'content': final_user_content})
            
            payload_messages = list(messages)
            if force_no_thinking:
                payload_messages.append({'role': 'system', 'content': 'CRITICAL INSTRUCTION: Do NOT output any internal thoughts, <think> tags, or reasoning. Output the final answer directly and immediately.'})
            
            if messages[-1]['role'] == 'user':
                assistant_response = query_llamacpp(base_url, payload_messages, model, stream_enabled, debug, images=images)
                if assistant_response:
                    messages.append({'role': 'assistant', 'content': assistant_response})
                print() 

        except KeyboardInterrupt:
            sys.stdout.write(f"\n{YELLOW}[Action Interrupted]{RESET}\n")
            continue
        except EOFError:
            print(f"\n{CYAN}Goodbye!{RESET}")
            break

# ==========================================
# BACKEND: LLAMA.CPP / OPENAI-COMPATIBLE
# ==========================================

def query_llamacpp(base_url, messages, model, stream_enabled=False, debug=False, show_thinking=True, context_size=None, images=None):
    start_time = time.time()
    full_content = ""
    try:
        # Try to insert the image into messages
        #     "messages": [{
        #                      "role": "user",
        #                      "content": "What is in this image?",
        #                      "images": ["'"$IMG"'"]
        #                  }],

        if images:
            messages[1]["images"] = images
        payload = {"model": model, "messages": messages, "stream": stream_enabled}
        if context_size is not None:
            payload["max_tokens"] = context_size
    #    if image:
    #        payload["image"] = image
            
        api_url = f"{base_url}/v1/chat/completions"
        data = json.dumps(payload).encode('utf-8')
        if debug:
            print(json.dumps(payload,indent=2))

        req = urllib.request.Request(api_url, data=data, headers={'Content-Type': 'application/json'})

        with urllib.request.urlopen(req) as response:
            if stream_enabled:
                started_content = False
                for line in response:
                    decoded_line = line.decode('utf-8').strip()
                    if not decoded_line: continue
                    if decoded_line.startswith('data: '): decoded_line = decoded_line[6:].strip()
                    if decoded_line == '[DONE]': continue 
                        
                    if debug: sys.stderr.write(f"\n{CYAN}[DEBUG RAW]: {decoded_line}{RESET}\n")
                    try: chunk = json.loads(decoded_line)
                    except json.JSONDecodeError: continue

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        
                        thought = delta.get("reasoning_content") or ""
                        if thought and show_thinking:
                            sys.stderr.write(f"{DARK_GRAY}{thought}{RESET}")
                            sys.stderr.flush()
                        
                        content = delta.get("content") or ""
                        if content:
                            if not started_content:
                                sys.stdout.write(f"\n{GREEN}--- Response ---{RESET}\n")
                                sys.stdout.flush()
                                started_content = True
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            full_content += content
                        
                        if choices[0].get("finish_reason") is not None:
                            sys.stdout.write("\n")
                            print_stats(time.time() - start_time, chunk)
                            break
                return full_content
            else:
                result = json.loads(response.read().decode('utf-8'))
                print_stats(time.time() - start_time, result)
                return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
    except KeyboardInterrupt:
        sys.stdout.write(f"\n\n{YELLOW}[Generation Interrupted by User]{RESET}\n")
        sys.stdout.flush()
        return full_content 
    except Exception as e:
        sys.stderr.write(f"\n{YELLOW}query_llamacpp Request Error: {e}{RESET}\n")
    return full_content

def chat_loop_llamacpp(base_url, system_prompt, initial_model, stream_enabled, debug, images=None):
        model = initial_model
        force_no_thinking = False
        context_size = None
        
        print(f"{CYAN}Entering Llama.cpp chat mode. Type '/?' for help.{RESET}")
        setup_readline(base_url, "llamacpp")
        messages = [{'role': 'system', 'content': system_prompt}]
        
        if images:
            print(f"{GREEN}Image loaded for session.{RESET}")

        while True:
          try:
            full_input = gather_user_input(model)
            if full_input is None:
                print(f"\n{CYAN}Goodbye!{RESET}")
                break
            
            full_input_stripped = full_input.strip()
            if not full_input_stripped: continue
            
            if full_input_stripped.lower() in ['exit', 'quit', '/exit', '/quit']:
                print(f"{CYAN}Goodbye!{RESET}"); break
            elif full_input_stripped in ['/?', '/help']:
                print(f"\n{CYAN}Commands:{RESET} /listmodel, /switchmodel, /contextsizeset <int>, /cwd, /ls, /clear, /thinkingoff, /thinkingon, /debug on, /debug off, /quit\n")
                print(f"  {YELLOW}! <command>{RESET}        - Execute a shell command inline")
                print(f"  {YELLOW}/curl <url>{RESET}        - Fetch and extract text from a URL inline")
                print(f"  {YELLOW}/spawnshell{RESET}        - Capture a full interactive shell session\n")
                continue
            elif full_input_stripped.startswith('/listmodel'):
                parts = full_input_stripped.split(maxsplit=1)
                list_models_llamacpp(base_url, parts[1] if len(parts) > 1 else None)
                continue
            elif full_input_stripped.startswith('/contextsizeset'):
                parts = full_input_stripped.split()
                if len(parts) > 1 and parts[1].isdigit():
                    val = int(parts[1])
                    if val == 0:
                        context_size = None
                        print(f"{GREEN}Context size limits reset to default.{RESET}\n")
                    else:
                        context_size = val
                        print(f"{GREEN}Max tokens limit set to {context_size}.{RESET}\n")
                else:
                    print(f"{YELLOW}Usage: /contextsizeset <integer> (use 0 for default){RESET}\n")
                continue
            elif full_input_stripped == '/clear':
                messages = [{'role': 'system', 'content': system_prompt}]
                print(f"{GREEN}Context memory wiped clean.{RESET}\n")
                continue
            elif full_input_stripped.lower() == '/debug on':
                debug = True
                print(f"{GREEN}Debug mode ENABLED.{RESET}\n")
                continue
            elif full_input_stripped.lower() == '/debug off':
                debug = False
                print(f"{GREEN}Debug mode DISABLED.{RESET}\n")
                continue
            elif full_input_stripped == '/thinkingoff':
                force_no_thinking = True
                print(f"{GREEN}Instructing model to SKIP the reasoning phase.{RESET}\n")
                continue
            elif full_input_stripped == '/thinkingon':
                force_no_thinking = False
                print(f"{GREEN}Allowing model to use reasoning phase normally.{RESET}\n")
                continue
            elif full_input_stripped.startswith('/cwd'):
                parts = full_input_stripped.split(maxsplit=1)
                if len(parts) > 1:
                    try: os.chdir(os.path.expanduser(parts[1]))
                    except Exception as e: print(f"{YELLOW}Error: {e}{RESET}")
                print(f"{GREEN}Directory: {os.getcwd()}{RESET}\n")
                continue
            elif full_input_stripped.startswith('/ls'):
                try: subprocess.run("ls" + full_input_stripped[3:], shell=True)
                except Exception as e: print(f"{YELLOW}Error: {e}{RESET}")
                print() 
                continue
            elif full_input_stripped.startswith('/switchmodel'):
                parts = full_input_stripped.split()
                if len(parts) > 1:
                    model = parts[1]
                    print(f"{GREEN}Switched model to '{model}'.{RESET}\n")
                continue
            
            elif full_input_stripped == '/spawnshell':
                shell_msg = handle_spawnshell()
                if not shell_msg: continue
                messages.append({'role': 'user', 'content': shell_msg})
            else:
                final_user_content = process_inline_commands(full_input)
                if not final_user_content.strip(): continue
                messages.append({'role': 'user', 'content': final_user_content})
            
            payload_messages = list(messages)
            if force_no_thinking:
                payload_messages.append({'role': 'system', 'content': 'CRITICAL INSTRUCTION: Do NOT output any internal thoughts, <think> tags, or reasoning. Output the final answer directly and immediately.'})
            
            if messages[-1]['role'] == 'user':
                assistant_response = query_llamacpp(base_url, payload_messages, model, stream_enabled, debug, show_thinking=True, context_size=context_size,images=images)
                if assistant_response:
                    messages.append({'role': 'assistant', 'content': assistant_response})
                print() 

          except KeyboardInterrupt:
            sys.stdout.write(f"\n{YELLOW}[Action Interrupted]{RESET}\n")
            continue
          except EOFError:
            print(f"\n{CYAN}Goodbye!{RESET}")
            break

# ==========================================
# ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Query Ollama or Llama.cpp with TTY streaming, memory, command execution, and batch processing.")
    parser.add_argument("-b", "--backend", choices=["ollama", "llamacpp"], default="ollama", help="API backend to use (default: ollama).")
    parser.add_argument("-H", "--host", help="Custom API URL (e.g. http://127.0.0.1:8080). Overrides environment variables.")
    parser.add_argument("-l", "--list", action="store_true", help="List all available local models and exit.")
    parser.add_argument("-la", "--list-all", action="store_true", help="List all models with capabilities (may be slower).")
    
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("-I", "--input-text", help="Direct string input for the query.")
    input_group.add_argument("-i", "--input-file", help="Path to a single input file.")
    input_group.add_argument("--input-dir", help="Directory containing input files.")
    input_group.add_argument("-c", "--chat", action="store_true", help="Start an interactive chat session with memory.")
    
    parser.add_argument("-o", "--output", help="Path to save output file (for -I or -i).")
    parser.add_argument("--output-dir", help="Directory to save output files (required for --input-dir).")
    parser.add_argument("-p", "--prompt", default=DEFAULT_SYSTEM_PROMPT, help="The system prompt.")
    parser.add_argument("-m", "--model", default="llama3", help="Ollama model (default: llama3).")
    parser.add_argument("--image", help="Path to an image file to include in the prompt (for multimodal models).")
    parser.add_argument("--show", action="store_true", help="Show concise model details (details, model_info, capabilities).")
    parser.add_argument("--show-details", action="store_true", help="Show full model information (all fields).")
    parser.add_argument("--output-format", choices=["json", "yaml"], default="json", help="Format for --show output (default: json).")

    parser.add_argument("--no-stream", action="store_true", help="Disable real-time streaming output.")
    parser.add_argument("--debug", action="store_true", help="Print raw JSON API chunks to stderr.")
    args = parser.parse_args()

    if args.host:
        base_url = args.host
    else:
        if args.backend == "llamacpp":
            base_url = os.environ.get('LLAMACPP_HOST', 'http://127.0.0.1:8080')
        else:
            base_url = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
            
    if not base_url.startswith(('http://', 'https://')):
        base_url = f"http://{base_url}"

    if args.list:
        if args.backend == "llamacpp":
            list_models_llamacpp(base_url)
        else:
            list_models_ollama(base_url)
        sys.exit(0)

    if args.list_all:
        # List all models with capabilities (slower)
        if args.backend == "llamacpp":
            sys.stderr.write(f"{YELLOW}List-all not implemented for llamacpp backend.{RESET}\n")
            sys.exit(1)
        else:
            list_models_ollama(base_url, include_capabilities=True)
        sys.exit(0)

    # Concise... (rest unchanged)

    # Concise flag to show selected fields (details, model_info, capabilities)
    if args.show:
        if args.backend == "llamacpp":
            sys.stderr.write(f"{YELLOW}Show not implemented for llamacpp backend.{RESET}\n")
        else:
            info = fetch_model_info_ollama(base_url, args.model)
            if info:
                subset = {
                    "details": info.get('details', {}),
                    "model_info": info.get('model_info', {}),
                    "capabilities": info.get('capabilities', [])
                }
                if args.output_format == "yaml" and HAVE_YAML:
                    import yaml
                    print(yaml.safe_dump(subset, sort_keys=False))
                else:
                    print(json.dumps(subset, indent=2))
            else:
                sys.stderr.write(f"{YELLOW}No model info returned for '{args.model}'.{RESET}\n")
                print(json.dumps({}, indent=2))
        sys.exit(0)

    # Full detailed flag
    if args.show_details:
        if args.backend == "llamacpp":
            sys.stderr.write(f"{YELLOW}Show details not implemented for llamacpp backend.{RESET}\n")
        else:
            info = fetch_model_info_ollama(base_url, args.model)
            if info:
                if args.output_format == "yaml" and HAVE_YAML:
                    import yaml
                    print(yaml.safe_dump(info, sort_keys=False))
                else:
                    print(json.dumps(info, indent=2))
            else:
                sys.stderr.write(f"{YELLOW}No model info returned for '{args.model}'. Ensure the Ollata server is running and the model name is correct.{RESET}\n")
                print(json.dumps({}, indent=2))
        sys.exit(0)

    is_tty = sys.stdout.isatty()
    should_stream = is_tty and not args.no_stream and not args.output and not args.output_dir

    if args.chat:
        image_data = None
        if args.image:
            image_data = prepare_image_data(args.image)
            if image_data:
                print(f"{GREEN}Image loaded: {args.image}{RESET}")

        images_list = [image_data] if image_data else None
        if args.backend == "llamacpp":
            chat_loop_llamacpp(base_url, args.prompt, args.model, should_stream, args.debug,images=images_list)
        if args.backend == "ollama":
            chat_loop_ollama(base_url,   args.prompt, args.model, should_stream, args.debug,images=images_list)
        sys.exit(0)

    input_data = ""
    if args.input_text:
        input_data = args.input_text
    elif args.input_file:
        if not os.path.isfile(args.input_file):
            sys.stderr.write(f"Error: File '{args.input_file}' not found.\n")
            sys.exit(1)
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = f.read()

    if input_data:
        messages = [{'role': 'system', 'content': args.prompt}, {'role': 'user', 'content': input_data}]
        
        image_data = None
        if args.image:
            image_data = prepare_image_data(args.image)
            if image_data:
                print(f"{GREEN}Image loaded: {args.image}{RESET}")

        if args.backend == "llamacpp":
            response = query_llamacpp(base_url, messages, args.model, stream_enabled=should_stream, debug=args.debug, images=image_data)
        else:
            # For ollama, we need to pass images as a list
            images_list = [image_data] if image_data else None
            response = query_ollama(base_url, messages, args.model, stream_enabled=should_stream, debug=args.debug, images=images_list)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Success: Output saved to {args.output}")
        elif not should_stream and response:
            print(response)

    elif args.input_dir:
        if not args.output_dir:
            sys.stderr.write("Error: --output-dir is required when using --input-dir.\n")
            sys.exit(1)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for filename in sorted(os.listdir(args.input_dir)):
            input_path = os.path.join(args.input_dir, filename)
            if os.path.isfile(input_path):
                print(f"Processing: {filename}...")
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                messages = [{'role': 'system', 'content': args.prompt}, {'role': 'user', 'content': content}]
                if args.backend == "llamacpp":
                    response = query_llamacpp(base_url, messages, args.model, stream_enabled=False, debug=args.debug)
                else:
                    response = query_ollama(base_url, messages, args.model, stream_enabled=False, debug=args.debug)
                
                output_path = os.path.join(args.output_dir, filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response)
                print(f"  -> Saved to {output_path}")

    if not (args.input_text or args.input_file or args.input_dir or args.chat):
      parser.error("You must provide an input (-I, -i, --input-dir), start a chat (-c), or list models (-l).")




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write(f"\n{CYAN}Exiting gracefully...{RESET}\n")
        sys.exit(0)

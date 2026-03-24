#!/usr/bin/python3
import os
import sys
import json
import argparse
import urllib.request
import urllib.error
import time
import subprocess
import threading
import re

# Handle Windows compatibility gracefully if readline isn't installed
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

# Handle Unix PTY for the /spawnshell command
try:
    import pty
    PTY_AVAILABLE = True
except ImportError:
    PTY_AVAILABLE = False

# --- CONFIGURATION & COLORS ---
DEFAULT_SYSTEM_PROMPT = "You are a chatbot trying to help user. Try to reply to the question as best as your knowledge goes but reply politely that you don't know if it is the case."

DARK_GRAY = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32;1m"
YELLOW = "\033[33;1m"
RESET = "\033[0m"

# ==========================================
# UTILITIES & API HELPERS
# ==========================================

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def fetch_models_ollama(base_url):
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

def list_models_ollama(base_url, filter_arg=None):
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
    # 1. Native Ollama format
    eval_count = chunk_or_result.get("eval_count", 0)
    eval_duration_ns = chunk_or_result.get("eval_duration", 0)
    prompt_eval_count = chunk_or_result.get("prompt_eval_count", 0)
    
    # 2. OpenAI standard format fallback
    usage = chunk_or_result.get("usage", {})
    if not eval_count and usage:
        eval_count = usage.get("completion_tokens", 0)
        prompt_eval_count = usage.get("prompt_tokens", 0)
        
    # 3. Llama.cpp specific format fallback
    timings = chunk_or_result.get("timings", {})
    if not eval_count and timings:
        eval_count = timings.get("predicted_n", 0)
        prompt_eval_count = timings.get("prompt_n", 0)
        eval_duration_ns = timings.get("predicted_ms", 0) * 1_000_000  # ms to ns
        
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
            '/cwd', '/ls', '/spawnshell', '/clear', 
            '/thinkingon', '/thinkingoff', 
            '/debug on', '/debug off', '/quit', '/exit', 'exit', 'quit'
        ]
        self.models = []

    def fetch_models(self):
        if self.backend == "llamacpp":
            self.models = [m['name'] for m in fetch_models_llamacpp(self.base_url)]
        else:
            self.models = [m['name'] for m in fetch_models_ollama(self.base_url)]

    def complete(self, text, state):
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
            except PermissionError: pass 
        elif text.startswith('/') or text in ['e', 'ex', 'exi', 'q', 'qu', 'qui']:
            matches = [c for c in self.commands if c.startswith(text)]
        else:
            matches = []

        return matches[state] if state < len(matches) else None

def setup_readline(base_url, backend):
    if READLINE_AVAILABLE:
        completer = ChatCompleter(base_url, backend)
        completer.fetch_models()
        readline.set_completer_delims(' \t\n')
        readline.set_completer(completer.complete)
        readline.parse_and_bind('tab: complete')
        print(f"{DARK_GRAY}[Tab Autocomplete Enabled]{RESET}\n")
    else:
        print()

def gather_user_input(model):
    user_input_lines = []
    in_multiline = False
    ctrl_c_count = 0
    
    while True:
        prompt_str = f"{YELLOW}you@{model} > {RESET}" if not in_multiline else f"{YELLOW}... > {RESET}"
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
# SHELL TOOL EXTENSIONS
# ==========================================

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
        pty.spawn(shell_cmd, read_and_capture)
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

def process_exclamation_commands(full_input):
    processed_lines = []
    for line in full_input.split('\n'):
        if line.lstrip().startswith("!"):
            command = line.lstrip()[1:].strip()
            if not command:
                processed_lines.append(line)
                continue
            output_str = execute_os_command(command)
            processed_lines.append(output_str)
        else:
            processed_lines.append(line)
    return "\n".join(processed_lines)

# ==========================================
# BACKEND: OLLAMA
# ==========================================

def query_ollama(base_url, messages, model, stream_enabled=False, debug=False, show_thinking=True):
    start_time = time.time()
    full_content = ""
    try:
        payload = {"model": model, "messages": messages, "stream": stream_enabled}
        api_url = f"{base_url}/api/chat"
        data = json.dumps(payload).encode('utf-8')
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
        sys.stderr.write(f"\n{YELLOW}Ollama Request Error: {e}{RESET}\n")
    return full_content

def chat_loop_ollama(base_url, system_prompt, initial_model, stream_enabled, debug):
    model = initial_model
    force_no_thinking = False
    print(f"{CYAN}Entering Ollama chat mode. Type '/?' for help.{RESET}")
    setup_readline(base_url, "ollama")
    messages = [{'role': 'system', 'content': system_prompt}]
    
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
                print(f"\n{CYAN}Commands:{RESET} /listmodel, /switchmodel, /cwd, /ls, /spawnshell, /clear, /thinkingoff, /thinkingon, /debug on, /debug off, /quit, ! <cmd>\n")
                continue
            elif full_input_stripped.startswith('/listmodel'):
                parts = full_input_stripped.split(maxsplit=1)
                list_models_ollama(base_url, parts[1] if len(parts) > 1 else None)
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
                final_user_content = process_exclamation_commands(full_input)
                if not final_user_content.strip(): continue
                messages.append({'role': 'user', 'content': final_user_content})
            
            payload_messages = list(messages)
            if force_no_thinking:
                payload_messages.append({'role': 'system', 'content': 'CRITICAL INSTRUCTION: Do NOT output any internal thoughts, <think> tags, or reasoning. Output the final answer directly and immediately.'})
            
            assistant_response = query_ollama(base_url, payload_messages, model, stream_enabled, debug, show_thinking=True)
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

def query_llamacpp(base_url, messages, model, stream_enabled=False, debug=False, show_thinking=True):
    start_time = time.time()
    full_content = ""
    try:
        payload = {"model": model, "messages": messages, "stream": stream_enabled}
        api_url = f"{base_url}/v1/chat/completions"
        data = json.dumps(payload).encode('utf-8')
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
        sys.stderr.write(f"\n{YELLOW}llama.cpp Request Error: {e}{RESET}\n")
    return full_content

def chat_loop_llamacpp(base_url, system_prompt, initial_model, stream_enabled, debug):
    model = initial_model
    force_no_thinking = False
    print(f"{CYAN}Entering Llama.cpp chat mode. Type '/?' for help.{RESET}")
    setup_readline(base_url, "llamacpp")
    messages = [{'role': 'system', 'content': system_prompt}]
    
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
                print(f"\n{CYAN}Commands:{RESET} /listmodel, /switchmodel, /cwd, /ls, /spawnshell, /clear, /thinkingoff, /thinkingon, /debug on, /debug off, /quit, ! <cmd>\n")
                continue
            elif full_input_stripped.startswith('/listmodel'):
                parts = full_input_stripped.split(maxsplit=1)
                list_models_llamacpp(base_url, parts[1] if len(parts) > 1 else None)
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
                final_user_content = process_exclamation_commands(full_input)
                if not final_user_content.strip(): continue
                messages.append({'role': 'user', 'content': final_user_content})
            
            payload_messages = list(messages)
            if force_no_thinking:
                payload_messages.append({'role': 'system', 'content': 'CRITICAL INSTRUCTION: Do NOT output any internal thoughts, <think> tags, or reasoning. Output the final answer directly and immediately.'})
            
            assistant_response = query_llamacpp(base_url, payload_messages, model, stream_enabled, debug, show_thinking=True)
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
    
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("-I", "--input-text", help="Direct string input for the query.")
    input_group.add_argument("-i", "--input-file", help="Path to a single input file.")
    input_group.add_argument("--input-dir", help="Directory containing input files.")
    input_group.add_argument("-c", "--chat", action="store_true", help="Start an interactive chat session with memory.")
    
    parser.add_argument("-o", "--output", help="Path to save output file (for -I or -i).")
    parser.add_argument("--output-dir", help="Directory to save output files (required for --input-dir).")
    parser.add_argument("-p", "--prompt", default=DEFAULT_SYSTEM_PROMPT, help="The system prompt.")
    parser.add_argument("-m", "--model", default="llama3", help="Ollama model (default: llama3).")
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

    if not (args.input_text or args.input_file or args.input_dir or args.chat):
        parser.error("You must provide an input (-I, -i, --input-dir), start a chat (-c), or list models (-l).")

    is_tty = sys.stdout.isatty()
    should_stream = is_tty and not args.no_stream and not args.output and not args.output_dir

    if args.chat:
        if args.backend == "llamacpp":
            chat_loop_llamacpp(base_url, args.prompt, args.model, should_stream, args.debug)
        else:
            chat_loop_ollama(base_url, args.prompt, args.model, should_stream, args.debug)
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
        if args.backend == "llamacpp":
            response = query_llamacpp(base_url, messages, args.model, stream_enabled=should_stream, debug=args.debug)
        else:
            response = query_ollama(base_url, messages, args.model, stream_enabled=should_stream, debug=args.debug)
        
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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write(f"\n{CYAN}Exiting gracefully...{RESET}\n")
        sys.exit(0)

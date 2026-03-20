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

def get_base_url():
    host = os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')
    if not host.startswith(('http://', 'https://')):
        host = f"http://{host}"
    return host

def strip_ansi(text):
    """Removes terminal color codes and cursor movements from the captured shell session."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def list_models(filter_arg=None):
    url = f"{get_base_url()}/api/tags"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode('utf-8'))
            models = data.get('models', [])
            
            if not models:
                print("No models found.")
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

            if sort_by == 'size':
                models.sort(key=lambda x: x.get('size', 0), reverse=True)
            else:
                models.sort(key=lambda x: x.get('name', ''))

            largest = max(models, key=lambda x: x.get('size', 0)) if models else None
            if largest:
                l_size_gb = largest.get('size', 0) / (1024**3)
                print(f"\nChecking storage... Largest model in list: {largest['name']} ({l_size_gb:.2f} GB)\n")

            print(f"{'NAME':<40} | {'SIZE':<10} | {'MODIFIED'}")
            print("-" * 75)
            for m in models:
                size_gb = m.get('size', 0) / (1024**3)
                modified = m.get('modified_at', 'Unknown')[:10]
                print(f"{m['name']:<40} | {size_gb:>8.2f} GB | {modified}")
            print()
                
    except urllib.error.URLError as e:
        sys.stderr.write(f"Error connecting to Ollama: {e}\n")
    except Exception as e:
        sys.stderr.write(f"Error fetching models: {e}\n")

# --- AUTOCOMPLETER ---
class ChatCompleter:
    def __init__(self):
        self.commands = [
            '/?', '/help', '/listmodel', '/switchmodel', 
            '/cwd', '/ls', '/spawnshell', '/clear', 
            '/thinkingon', '/thinkingoff', 'exit', 'quit'
        ]
        self.models = []

    def fetch_models(self):
        try:
            url = f"{get_base_url()}/api/tags"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode('utf-8'))
                self.models = [m['name'] for m in data.get('models', [])]
        except Exception:
            self.models = []

    def complete(self, text, state):
        buffer = readline.get_line_buffer()
        
        if buffer.startswith('/switchmodel '):
            matches = [m for m in self.models if m.startswith(text)]
        elif buffer.startswith('/cwd ') or buffer.startswith('/ls '):
            path = os.path.expanduser(text)
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
                            prefix = os.path.dirname(text)
                            
                            if buffer.startswith('/cwd '):
                                if os.path.isdir(full_path):
                                    matches.append(os.path.join(prefix, item) + '/' if prefix else item + '/')
                            else:
                                if os.path.isdir(full_path):
                                    matches.append(os.path.join(prefix, item) + '/' if prefix else item + '/')
                                else:
                                    matches.append(os.path.join(prefix, item) if prefix else item)
            except PermissionError:
                pass 
        elif text.startswith('/') or text in ['e', 'ex', 'exi', 'q', 'qu', 'qui']:
            matches = [c for c in self.commands if c.startswith(text)]
        else:
            matches = []

        if state < len(matches):
            return matches[state]
        else:
            return None

def print_stats(total_time, chunk_or_result):
    eval_count = chunk_or_result.get("eval_count", 0)
    eval_duration_ns = chunk_or_result.get("eval_duration", 0)
    
    prompt_eval_count = chunk_or_result.get("prompt_eval_count", 0)
    total_context = prompt_eval_count + eval_count
    
    tps = 0.0
    if eval_count > 0 and eval_duration_ns > 0:
        tps = eval_count / (eval_duration_ns / 1e9)
        
    sys.stderr.write(f"\n{DARK_GRAY}--- Stats: {total_time:.2f}s total | {tps:.2f} t/s | Context: {total_context} tokens ---{RESET}\n")
    sys.stderr.flush()

def query_ollama(messages, model, stream_enabled=False, debug=False, show_thinking=True):
    start_time = time.time()
    full_content = ""
    
    # Wrap the entire process (including encoding payload) to catch interrupts early
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream_enabled
        }

        api_url = f"{get_base_url()}/api/chat"
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(api_url, data=data, headers={'Content-Type': 'application/json'})

        with urllib.request.urlopen(req) as response:
            if stream_enabled:
                started_content = False
                for line in response:
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        if debug:
                            sys.stderr.write(f"\n{CYAN}[DEBUG RAW]: {decoded_line.strip()}{RESET}\n")
                        
                        chunk = json.loads(decoded_line)
                        msg = chunk.get("message", {})
                        
                        thought = msg.get("thought") or msg.get("thinking") or ""
                        if thought:
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
                            total_time = time.time() - start_time
                            print_stats(total_time, chunk)
                            
                return full_content
            else:
                raw_res = response.read().decode('utf-8')
                result = json.loads(raw_res)
                total_time = time.time() - start_time
                print_stats(total_time, result)
                return result['message'].get('content', '')
                
    except KeyboardInterrupt:
        sys.stdout.write(f"\n\n{YELLOW}[Generation Interrupted by User]{RESET}\n")
        sys.stdout.flush()
        return full_content 
    except urllib.error.URLError as e:
        sys.stderr.write(f"\n{YELLOW}Error: Could not connect to Ollama at {api_url}.\n({e}){RESET}\n")
    except Exception as e:
        sys.stderr.write(f"\n{YELLOW}An error occurred: {e}{RESET}\n")
    
    return full_content

def chat_loop(system_prompt, initial_model, stream_enabled, debug):
    model = initial_model
    force_no_thinking = False
    
    print(f"{CYAN}Entering interactive chat mode. Type '/?' for help.{RESET}")
    
    if READLINE_AVAILABLE:
        completer = ChatCompleter()
        completer.fetch_models()
        readline.set_completer_delims(' \t\n')
        readline.set_completer(completer.complete)
        readline.parse_and_bind('tab: complete')
        print(f"{DARK_GRAY}[Tab Autocomplete Enabled]{RESET}\n")
    else:
        print()
    
    messages = [{'role': 'system', 'content': system_prompt}]
    ctrl_c_count = 0
    
    while True:
        try:
            user_input_lines = []
            in_multiline = False
            
            while True:
                prompt_str = f"{YELLOW}you@{model} > {RESET}" if not in_multiline else f"{YELLOW}... > {RESET}"
                try:
                    line = input(prompt_str)
                    ctrl_c_count = 0 
                except KeyboardInterrupt:
                    if in_multiline:
                        print(f"\n{YELLOW}[Multiline input cancelled]{RESET}")
                        in_multiline = False
                        user_input_lines = []
                        break
                    else:
                        ctrl_c_count += 1
                        if ctrl_c_count >= 2:
                            print(f"\n{CYAN}Goodbye!{RESET}")
                            return
                        print(f"\n{YELLOW}(Press Ctrl+C again to exit){RESET}")
                        user_input_lines = []
                        break
                
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
            
            full_input = "\n".join(user_input_lines)
            full_input_stripped = full_input.strip()
            
            if not full_input_stripped:
                continue
            
            # --- BUILT-IN COMMANDS PARSER ---
            if full_input_stripped.lower() in ['exit', 'quit']:
                print(f"{CYAN}Goodbye!{RESET}")
                break
                
            elif full_input_stripped in ['/?', '/help']:
                print(f"\n{CYAN}Available Commands:{RESET}")
                print(f"  {YELLOW}/?{RESET} or {YELLOW}/help{RESET}      - Show this help message")
                print(f"  {YELLOW}/listmodel [arg]{RESET}   - List models (e.g., '/listmodel size', '/listmodel qwen')")
                print(f"  {YELLOW}/switchmodel <m>{RESET}   - Switch to a different model")
                print(f"  {YELLOW}/thinkingoff{RESET}       - Instruct the model to SKIP reasoning (to speed up queries)")
                print(f"  {YELLOW}/thinkingon{RESET}        - Allow the model to use reasoning phase normally (default)")
                print(f"  {YELLOW}/cwd <path>{RESET}        - Change the current working directory")
                print(f"  {YELLOW}/ls [args]{RESET}         - Run 'ls' locally (output is NOT sent to the model)")
                print(f"  {YELLOW}/spawnshell{RESET}        - Spawn a shell, capture the session on exit, and send to the model")
                print(f"  {YELLOW}/clear{RESET}             - Clear the conversation memory/context")
                print(f"  {YELLOW}exit{RESET} or {YELLOW}quit{RESET}       - Exit the chat session")
                print(f"  {YELLOW}! <command>{RESET}        - Execute a shell command AND send its output to the model")
                print(f"  {YELLOW}\"\"\"{RESET}                - Start/end a multiline text block\n")
                continue
                
            elif full_input_stripped.startswith('/listmodel'):
                parts = full_input_stripped.split(maxsplit=1)
                filter_arg = parts[1] if len(parts) > 1 else None
                list_models(filter_arg)
                continue

            elif full_input_stripped == '/clear':
                messages = [{'role': 'system', 'content': system_prompt}]
                print(f"{GREEN}Context memory wiped clean.{RESET}\n")
                continue
                
            elif full_input_stripped == '/thinkingoff':
                force_no_thinking = True
                print(f"{GREEN}Instructing model to SKIP the reasoning phase. (We will still show it if the model disobeys).{RESET}\n")
                continue
                
            elif full_input_stripped == '/thinkingon':
                force_no_thinking = False
                print(f"{GREEN}Allowing model to use reasoning phase normally.{RESET}\n")
                continue
                
            elif full_input_stripped.startswith('/cwd'):
                parts = full_input_stripped.split(maxsplit=1)
                if len(parts) > 1:
                    target_dir = os.path.expanduser(parts[1])
                    try:
                        os.chdir(target_dir)
                        print(f"{GREEN}Changed directory to: {os.getcwd()}{RESET}\n")
                    except FileNotFoundError:
                        print(f"{YELLOW}Directory not found: {target_dir}{RESET}\n")
                    except Exception as e:
                        print(f"{YELLOW}Error changing directory: {e}{RESET}\n")
                else:
                    print(f"{GREEN}Current directory: {os.getcwd()}{RESET}\n")
                continue
                
            elif full_input_stripped.startswith('/ls'):
                cmd_to_run = "ls" + full_input_stripped[3:]
                try:
                    subprocess.run(cmd_to_run, shell=True)
                except Exception as e:
                    print(f"{YELLOW}Error running ls: {e}{RESET}")
                print() 
                continue

            elif full_input_stripped.startswith('/switchmodel'):
                parts = full_input_stripped.split()
                if len(parts) > 1:
                    new_model = parts[1]
                    print(f"{GREEN}Switched model from '{model}' to '{new_model}'.{RESET}\n")
                    model = new_model
                else:
                    print(f"{YELLOW}Usage: /switchmodel <model_name>{RESET}\n")
                continue
                
            elif full_input_stripped == '/spawnshell':
                if not PTY_AVAILABLE:
                    print(f"{YELLOW}The /spawnshell command is only available on Unix-like systems (Linux/macOS).{RESET}\n")
                    continue
                
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
                    continue
                
                print(f"{DARK_GRAY}--- Shell session captured ({len(clean_output)} chars) ---{RESET}")
                
                formatted_prompt = f"I spawned an interactive shell, executed some commands, and then exited. Here is the transcript of my session:\n```text\n{clean_output}\n```\nPlease analyze this session and provide any relevant insights, or simply acknowledge it if no action is needed."
                
                messages.append({'role': 'user', 'content': formatted_prompt})
                
                payload_messages = list(messages)
                if force_no_thinking:
                    payload_messages.append({
                        'role': 'system', 
                        'content': 'CRITICAL INSTRUCTION: Do NOT output any internal thoughts, <think> tags, or reasoning. Output the final answer directly and immediately.'
                    })
                
                assistant_response = query_ollama(payload_messages, model, stream_enabled, debug)
                
                if assistant_response:
                    messages.append({'role': 'assistant', 'content': assistant_response})
                print() 
                continue

            # --- OS COMMAND PARSER (OUTPUT SENT TO LLM) ---
            processed_lines = []
            for line in full_input.split('\n'):
                if line.lstrip().startswith("!"):
                    command = line.lstrip()[1:].strip()
                    if not command:
                        processed_lines.append(line)
                        continue
                    
                    print(f"{DARK_GRAY}--- Executing (max 5s): {command} ---{RESET}")
                    
                    output_lines = []
                    
                    def read_output(pipe):
                        for out_line in pipe:
                            sys.stdout.write(out_line)
                            sys.stdout.flush()
                            output_lines.append(out_line)
                    
                    try:
                        process = subprocess.Popen(
                            command, 
                            shell=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT, 
                            text=True,
                            bufsize=1
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
                        processed_lines.append(f"\n[Command executed: `{command}`]\n```text\n{output.strip()}\n```\n")
                        
                    except Exception as e:
                        print(f"{CYAN}Failed to execute command: {e}{RESET}")
                        processed_lines.append(f"[Failed to execute `{command}`: {e}]")
                else:
                    processed_lines.append(line)
            
            final_user_content = "\n".join(processed_lines)
            if not final_user_content.strip():
                continue
                
            messages.append({'role': 'user', 'content': final_user_content})
            
            # --- SEND TO OLLAMA ---
            payload_messages = list(messages)
            if force_no_thinking:
                payload_messages.append({
                    'role': 'system', 
                    'content': 'CRITICAL INSTRUCTION: Do NOT output any internal thoughts, <think> tags, or reasoning. Output the final answer directly and immediately.'
                })
            
            assistant_response = query_ollama(payload_messages, model, stream_enabled, debug)
            
            if assistant_response:
                messages.append({'role': 'assistant', 'content': assistant_response})
            print() 

        except KeyboardInterrupt:
            # Catch stray interrupts during text parsing or background prep to prevent crashing
            sys.stdout.write(f"\n{YELLOW}[Action Interrupted]{RESET}\n")
            sys.stdout.flush()
            continue
        except EOFError:
            print(f"\n{CYAN}Goodbye!{RESET}")
            break

def main():
    parser = argparse.ArgumentParser(description="Query Ollama with TTY streaming, memory, command execution, and batch processing.")
    
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

    if args.list:
        list_models()
        sys.exit(0)

    if not (args.input_text or args.input_file or args.input_dir or args.chat):
        parser.error("You must provide an input (-I, -i, --input-dir), start a chat (-c), or list models (-l).")

    is_tty = sys.stdout.isatty()
    should_stream = is_tty and not args.no_stream and not args.output and not args.output_dir

    if args.chat:
        chat_loop(args.prompt, args.model, stream_enabled=should_stream, debug=args.debug)
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
        messages = [
            {'role': 'system', 'content': args.prompt},
            {'role': 'user', 'content': input_data}
        ]
        response = query_ollama(messages, args.model, stream_enabled=should_stream, debug=args.debug)
        
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
                
                messages = [
                    {'role': 'system', 'content': args.prompt},
                    {'role': 'user', 'content': content}
                ]
                response = query_ollama(messages, args.model, stream_enabled=False, debug=args.debug)
                
                output_path = os.path.join(args.output_dir, filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response)
                print(f"  -> Saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # This will now only catch Ctrl+C if you aren't in chat mode (e.g. bulk file processing)
        sys.stderr.write(f"\n{CYAN}Exiting gracefully...{RESET}\n")
        sys.exit(0)

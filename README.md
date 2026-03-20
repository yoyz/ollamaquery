# ollamaquery

A lightweight, feature-rich command-line interface for interacting with local LLMs via Ollama. It supports single-shot queries, batch processing, and an advanced interactive chat mode with OS integration.

## Requirements

* Python 3.x
* [Ollama](https://ollama.com/) running locally (or remotely via the `OLLAMA_HOST` environment variable).
* Tested on Centos10 and Ubuntu 24 
* `readline` autocompletion and `pty` shell spawning

## Basic Usage

Run a single query:
```bash
python3 ollamaquery.py -I "Explain quantum computing in one sentence." -m llama3

```

Process a file:

```bash
python3 ollama_query.py -i input.txt -o output.txt -m qwen2.5:7b

```

List available models:

```bash
python3 ollama_query.py -l

```

## Interactive Chat Mode

The powerful way to use this tool is the interactive chat mode, which maintains conversation history and provides system integrations.

```bash
python3 ollama_query.py -c -m llama3

```

### Chat Mode Features

* **Tab Autocompletion**: Use `<Tab>` to autocomplete slash commands, model names, and file paths.
* **Multiline Input**: Type `"""` to start a multiline block. Type `"""` again on a new line to send it.
* **Graceful Interrupts**: Press `Ctrl+C` while the model is generating to stop it early while keeping the partial response in context. Press `Ctrl+C` at an empty prompt to clear the line, and twice to exit.

### Shell Integration

You can execute shell commands directly from the chat prompt. The output is captured and fed into the model's context window automatically.

* `! <command>`: Executes a command and sends the output to the LLM. (e.g., `! cat main.py` or `! df -h`). Commands have a 5-second timeout to prevent hanging.

### Slash Commands

Inside the chat mode, you can use the following built-in commands:

| Command | Description |
| --- | --- |
| `/?` or `/help` | Show the help menu. |
| `/listmodel [arg]` | List available local models. Supports filtering (e.g., `/listmodel size` or `/listmodel qwen`). |
| `/switchmodel <name>` | Swap the active model without losing conversation history. |
| `/cwd <path>` | Change the current working directory. |
| `/ls [args]` | Run `ls` locally to view files. This output is **not** sent to the LLM. |
| `/spawnshell` | Drops you into a full interactive Unix shell. When you type `exit`, the entire session transcript is sent to the LLM for analysis. |
| `/thinkingoff` | Instructs reasoning models (like DeepSeek-R1) to skip the thinking phase. |
| `/thinkingon` | Restores normal reasoning behavior. |
| `/clear` | Wipes the current conversation memory/context. |
| `exit` or `quit` | End the session. |

## Environment Variables

* `OLLAMA_HOST`: Set this to target a remote Ollama instance (e.g., `export OLLAMA_HOST="http://192.168.1.50:11434"`). Defaults to `127.0.0.1:11434`.

## Notes on "Thinking" Models

If you use a model that outputs reasoning (like DeepSeek-R1), the tool will automatically detect the `<think>` tags or `thought` objects and print them in dark gray to visually separate them from the final response. Performance metrics (Tokens per Second and Total Context Size) are printed at the end of every generation.

```


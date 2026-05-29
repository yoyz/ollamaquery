# Agentic Protocol — Deep Dive

This is a sophisticated, custom-built implementation of the **ReAct (Reasoning and Acting)** agentic framework. By integrating a dynamic prompt builder, a dedicated execution sandbox, and ephemeral context tracking, the script allows standard local LLMs to operate as autonomous agents.

Here is a deep dive into the mechanics of the agentic workflow, designed to give you everything you need for your documentation.

---

### 1. Prompt and Environment Construction (The Setup)

Before the LLM generates a single token, the script heavily manipulates both the system prompt and the backend inference parameters to optimize the model for tool-calling.

#### A. The Modular Prompt Builder (`get_agentic_prompt`)

Instead of a massive, hardcoded string, the agentic prompt is dynamically assembled from composable blocks. This prevents context confusion and allows the script to adapt to different models (e.g., models that natively speak JSON vs. models that need Markdown fences).

```text
[SCHEMA: Agentic Prompt Composition]
┌─────────────────────────────────────────────────────────────┐
│ 1. ROLE BLOCK    (Defines the persona & terminal environment) │
├─────────────────────────────────────────────────────────────┤
│ 2. TOOL DEFS     (Dynamically injected JSON schemas of tools) │
├─────────────────────────────────────────────────────────────┤
│ 3. FORMAT BLOCK  (Strict bare JSON vs Soft Markdown JSON)     │
├─────────────────────────────────────────────────────────────┤
│ 4. EXAMPLE BLOCK (ReAct protocol step-by-step instructions)   │
├─────────────────────────────────────────────────────────────┤
│ 5. RULES BLOCK   (Language mirroring, path strictness)        │
└─────────────────────────────────────────────────────────────┘

```

* **Format Selection:** The script checks `AGENTIC_PROMPT_STYLE_REGISTRY`. If you load `Nemotron-Cascade`, it uses the `"soft"` format (requests JSON inside fenced code blocks). If you use `Qwen`, it defaults to `"strict"` (requests bare, unfenced JSON).

#### B. The Inference Environment (`get_inference_params`)

Tool calling requires deterministic, highly focused outputs. Creative randomness (high temperature) causes malformed JSON.
The script intercepts the model name and looks it up in `MODEL_INFERENCE_PARAMS_REGISTRY`. It forcibly overrides the LLM's standard parameters:

* **Temperature (e.g., `0.6`):** Lowered to prevent hallucinations in JSON syntax.
* **Top_P (e.g., `0.95`):** Restricts the token probability pool to highly likely syntax.
* **Presence/Repeat Penalty (e.g., `0.0`/`1.0`):** Disabled so the model isn't penalized for writing identical JSON brackets or repeated function names.

---

### 2. The Multi-Turn ReAct Protocol (Data Exchange)

Once the environment is set, the script enters a `while iteration < max_iterations:` loop. This is the core protocol of what is sent and received.

1. **Sent to LLM:** The assembled Agentic Prompt + User Query.

When the user types a command in /agentic mode, the script does not send your standard brief system prompt. It dynamically builds a massive "instruction manual" using get_agentic_prompt().
Here is the exact Python structure of the messages array sent to the LLM (assuming the user asked: "What files are in this folder?" and using the Qwen strict JSON format):

```markdown
messages = [
    {
        "role": "system",
        "content": """You are a capable AI agent with access to tools. You operate in a terminal environment.

## Available tools
- fetch_url: Fetch a URL and return its content as plain text. Arguments: url: Full URL (http/https only) (required: url)
- run_command: Execute a shell command (compiler, build tool, etc.). Returns stdout/stderr. Default timeout: 10s, max: 300s. Arguments: command: Shell command to execute, timeout: Seconds (default 10, max 300). (required: command)
- read_file: Read a file from disk (text, max 100KB). Path relative to CWD. Arguments: file_path: File path relative to CWD (required: file_path)
[... other tools omitted for brevity ...]

## Output format
When you need to perform an action, respond with a JSON object:
{"tool": "tool_name", "arguments": {"arg1": "value1", ...}}

After the tool runs, you will receive the result as an observation. Use it to decide the next step.

## CRITICAL rules
- Output ONLY the JSON tool call — no surrounding text, no explanations, no markdown fences.
- Do NOT chain multiple commands with `&&`, `|`, `;` etc. Each tool call runs in isolation.

## ReAct protocol (Think → Act → Observe → Answer)
1. Think about what the user needs and which tool can help
2. Act by outputting a JSON tool call
3. Observe the tool result (it will be shown to you)
4. Repeat if more actions are needed
5. When you have the answer, respond in plain text (no JSON) — that is your final answer

## General rules
- Be precise with file paths.
- Mirror the user's language — if they write in French, reply in French."""
    },
    {
        "role": "user",
        "content": "What files are in this folder?"
    }
]
```


2. **Received from LLM:** The LLM evaluates the prompt and outputs a tool request:
`{"tool": "run_command", "arguments": {"command": "ls -la"}}`

3. **Local Execution:** The script's `ToolRegistry` catches this, parses the JSON, and runs the command locally (either on the host or inside a Podman/Docker container via the `Executor` class).

Once the LLM reads the prompt above, it generates a response. In strict mode, it generates only JSON.

Step A: The LLM Output
The LLM generates:

```
{"tool": "run_command", "arguments": {"command": "ls -la"}}
```

Step B: Parsing in the Script
The script runs self.parse_tool_calls(response_text). It uses regex (_TOOL_CALL_OPEN_RE) to find the {, verifies it matches the required format, and converts it into a Python dictionary:

```
tool_call = {
    "tool": "run_command",
    "arguments": {"command": "ls -la"}
}
```

Step C: Execution
The script passes this to self.tool_registry.execute("run_command", {"command": "ls -la"}).

The registry checks if the tool is destructive (like patch or run_command).

If it is, it prompts the user in the terminal: [Agentic] Run run_command(command='ls -la')? [y/N].

If confirmed, the Executor class runs subprocess.run(['ls', '-la'], ...) (either on the host or inside a Docker/Podman container).

The tool returns a standardized dictionary result to the script:

```
result = {
    "success": True,
    "output": "total 12\ndrwxr-xr-x 2 user user 4096 May 10 10:00 .\ndrwxr-xr-x 4 user user 4096 May 10 09:00 ..\n-rw-r--r-- 1 user user  120 May 10 10:00 script.py",
    "error": None
}
```





4. **Sent to LLM:** The script feeds the terminal output back to the LLM packaged exactly like this (including the tool name as prefix):
`{'role': 'user', 'content': "Tool result:\n[run_command] total 12\ndrwxr-xr-x 2 user user 4096 May 10 10:00 .\ndrwxr-xr-x 4 user user 4096 May 10 09:00 ..\n-rw-r--r-- 1 user user  120 May 10 10:00 script.py"}`


5. **Loop/Terminate:** The LLM receives the observation. If it needs more data, it outputs another JSON block (Step 2). If it has enough info, it outputs standard text to answer the user.


This is the cleverest part of the ReAct (Reasoning and Acting) protocol. The script doesn't decide when to terminate; the LLM does.

Look back at the AGENTIC_EXAMPLE_STRICT prompt in section 1. It explicitly tells the LLM:
"5. When you have the answer, respond in plain text (no JSON) — that is your final answer"



---

### 3. Context Management (Memory & State)

This is the most complex and clever part of the script. Managing memory in an agentic loop requires balancing two things: giving the model enough context to solve the problem, while preventing the long-term chat history from bloating with terminal logs.

It achieves this by splitting memory into **Ephemeral (Short-term)** and **Persistent (Long-term)** states.



#### Ephemeral Memory (The `messages` array)

When `/agentic` mode processes a query, it creates a temporary list called `messages`.

* **Intermediate States ARE kept:** Every tool call the assistant makes, and every `Tool result:` the script returns, is appended to this temporary `messages` list. This allows the model to "remember" what tools it already ran in previous iterations of the *current* task.
* **Thinking is DISCARDED:** If the backend (like llama.cpp) supports native reasoning and sends it via the `reasoning_content` API key, the script prints it to your screen (if `/agentic thinking` is on) but **does not** append it to the `messages` array. This is a brilliant optimization—it saves massive amounts of context window space by treating the model's internal monologue as exhaust rather than context. *(Note: If the model forcefully outputs `<think>` tags directly inside the standard text `content` block, it will inadvertently be kept).*

#### Persistent Memory (`self.messages`)

Once the agent finishes its loop and generates a final human-readable answer, a crucial context swap happens:

1. The script takes the final text answer and streams it to the user.
2. The massive temporary `messages` array (filled with terminal logs and JSON blocks) is **thrown away**.
3. Only the final, polished response is appended to `self.messages` (the persistent history).

**Result:** In turn 2 of the conversation, the LLM remembers the *conclusion* of Turn 1, but its context window isn't clogged by the 5 failed `ls` and `grep` commands it took to get there.


During an agentic task, the script creates a temporary, ephemeral list called messages to hold all the messy intermediate tool logs. Once the loop terminates (step 5 — Loop/Terminate), the script updates self.messages — which represents the long-term memory of the whole chat session.

Crucially, self.messages completely ignores the intermediate tool steps. It only saves the user's original request and the LLM's final, human-readable conclusion.

If you were to execute /dumpcontext memory.json after the task finished, the persistent self.messages structure would look exactly like this:

```
[
  {
    "role": "system",
    "content": "You are a coding specialist focused on Python and C++..."
  },
  {
    "role": "user",
    "content": "What files are in this folder?"
  },
  {
    "role": "assistant",
    "content": "There is one file in this folder: a python script named script.py."
  }
]
```

If you ask a follow-up question ("Can you explain what script.py does?"), the LLM reads self.messages. It knows script.py exists because it's in the assistant's previous answer. But because the massive terminal output of ls -la was discarded along with the ephemeral memory, your context window stays extremely small and cheap, preventing Max Token / Out of Memory errors during long debugging sessions!



---

### 4. Two-Tier Timeout System

```
                   ┌──────────────────────────────────────────────────────────┐
                   │               ReAct STEP (default 120s)                  │
                   │                                                          │
   User Query      │   ┌──────────────┐    ┌──────────────┐                   │
 ──────────────────┼──▶│  LLM thinks  │───▶│ Tool executes │                   │
                   │   │  (generates  │    │  (timeout:    │                   │
                   │   │   tool call) │    │   default 10s)│                   │
                   │   └──────────────┘    └──────┬───────┘                   │
                   │                              │                           │
                   │                              ▼                           │
                   │                      ┌──────────────┐                   │
                   │                      │  Result fed   │                   │
                   │                      │  back to LLM  │                   │
                   │                      └──────┬───────┘                   │
                   │                             │                           │
                   │            ┌─────────────────┼──────────────┐            │
                   │            ▼                 ▼              ▼            │
                   │    ┌────────────┐    ┌──────────────┐  ┌───────────┐    │
                   │    │ More tools?│    │ Have answer? │  │ Timed out │    │
                   │    │  ──▶ loop  │    │  ──▶ output  │  │  ──▶ abort│    │
                   │    └────────────┘    └──────────────┘  └───────────┘    │
                   └──────────────────────────────────────────────────────────┘

   Timeline (seconds):
   ──┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬──▶
     │     │     │     │     │     │     │     │     │     │     │
    0│     │   10│     │     │     │   60│     │     │   90│     │  120
     │     │     │     │     │     │     │     │     │     │     │
     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼

   Example 1: Fast tool, fast LLM
   ┌───────┐  ┌──────┐
   │ LLM   │  │ Tool │
   │ thinks│  │ runs │
   │ 2s    │  │ 0.5s │
   └───────┘  └──────┘
   ←──── 2.5s total ────→   ✓ Within step timeout

   Example 2: Slow tool, LLM overrides timeout
   ┌───────┐  ┌────────────────────┐
   │ LLM   │  │ Tool runs for 30s  │
   │ thinks│  │ (LLM set timeout:  │
   │ 3s    │  │  "timeout": 60)    │
   └───────┘  └────────────────────┘
   ←──── 33s total ────→   ✓ Within step timeout, tool didn't hit its 60s limit

   Example 3: LLM stuck thinking → step timeout kills it
   ┌─────────────────────────────────────────┐
   │ LLM thinks ... and thinks ... and thinks │
   │ (never outputs tool call or answer)      │
   └─────────────────────────────────────────┘
   ←────────── 120s total ──────────→   ✗ Step timeout: "timed out after 120s"

   Example 4: Tool stuck → tool timeout kills it
   ┌───────┐  ┌──────────────────────┐
   │ LLM   │  │ Tool runs but hangs  │
   │ thinks│  │ (e.g. ./dnsresolver  │
   │ 3s    │  │  waiting for DNS)    │
   └───────┘  └──────────────────────┘
   ←──── total ────→   Tool killed at 10s: "Timed out after 10s"
                        Step continues because 10s < 120s step timeout

The agentic loop has two independent timeout mechanisms that protect against stuck commands and runaway LLM thinking.

#### Tool-Level Timeout

Each tool call has its own timeout, enforced by Python's `subprocess.run(timeout=...)`.

| Tool | Default timeout | Max | Controlled by |
|------|----------------|-----|---------------|
| `run_command` | 10s | 300s | `args.get("timeout", 10)` |
| `run_python` | 10s | 300s | `args.get("timeout", 10)` |
| Other tools | N/A | N/A | Instant execution |

The LLM can override the timeout per-call by passing a `timeout` argument:
```json
{"tool": "run_command", "arguments": {"command": "gcc -o bigfile bigfile.c", "timeout": 60}}
```

If the tool exceeds its timeout, the user sees:
```
[Tool] run_command(command='gcc -o bigfile bigfile.c') → ERROR (10.0s)
```

The LLM receives a JSON error: `{"stdout": "", "stderr": "Timed out after 60s", "returncode": -1}` and can retry with a higher timeout.

#### Step-Level (ReAct Loop) Timeout

Each iteration of the ReAct loop has a global timeout that covers the entire round-trip:

```
LLM thinking time  +  tool execution time  ≤  step timeout
```

Default: **120s**. Configurable via `/agentic timeout <seconds>`.

If the LLM takes 115s to think and the tool takes 10s, that's 125s total → the step times out:
```
[Agentic] Step timed out after 120s.
```

This protects against the LLM getting stuck in infinite thinking loops (repetitive reasoning without generating a tool call or answer).

#### Comparison

| Aspect | Tool timeout | Step timeout |
|--------|-------------|-------------|
| What it limits | One command execution | Full round-trip (LLM + tool) |
| Default | 10s | 120s |
| Configurable by LLM? | Yes (via `timeout` arg) | No (user setting via `/agentic timeout`) |
| Configurable by user? | No (hardcoded max 300s) | Yes (`/agentic timeout <sec>`) |
| Visible in output? | Yes — `ERROR (10.0s)` | Yes — `timed out after 120s` |

#### Timing Display

The user sees elapsed time for every tool call:

```
[Tool] write_file(path='test.c', content='...') → OK (0.1s)
[Tool] run_command(command='gcc test.c -o test') → OK (2.3s)
[Tool] run_command(command='./test') → ERROR (10.0s)
```

The timing information is also included in the observation sent to the LLM as structured JSON:
```json
{"tool": "run_command", "duration_s": 10.0, "success": false, "output": "..."}
```

### 5. Areas for Improvement in the Workflow

While highly effective, there are a few architectural bottlenecks in this workflow that you could highlight in your documentation as future improvement areas:

* **No Observation Truncation:** Currently, if a tool (like `read_file`) returns up to 100KB of text, it is appended entirely to the ephemeral `messages` array. If the model runs this tool 3 times, the context will explode, and the API will throw an Out-Of-Memory (OOM) or Max Tokens error.
* *Improvement:* Implement a dynamic sliding window that summarizes or truncates older `Tool result:` blocks in the ephemeral memory if the total token count exceeds 80% of `self.ctx.context_window_size`.


* **Strict JSON Parsing Rigidity:** The `parse_tool_call` function relies heavily on regex to find JSON boundaries. If the LLM generates a slightly malformed JSON string (e.g., missing a closing quote or adding a trailing comma), the regex will fail, the script will assume it's a final answer, and the loop will break prematurely.
* *Improvement:* Integrate a library like `json_repair` to automatically fix minor syntax hallucinations before throwing them to the `json.loads()` parser.


* **Lack of Error Backoff:** If a tool fails (e.g., "File not found"), the script feeds the error back to the LLM. However, if the LLM gets stuck in a loop trying the exact same failed command, the script catches it (`Same tool call repeated, breaking loop.`) but immediately aborts the whole task.
* *Improvement:* Instead of aborting, the script could inject a hardcoded prompt constraint: `{'role': 'user', 'content': "System override: You are stuck repeating the same command. Try a completely different tool or file path."}` to nudge it out of the loop.


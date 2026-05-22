# ollamaquery

CLI suite for interacting with local LLMs (Ollama, Llama.cpp, LM Studio). Two versions available.

## [ollamaquery2/](ollamaquery2/) (recommended)

Active version with full feature set: streaming, inline commands, color themes, multi-backend auto-detection, agentic ReAct mode (11 tools), spawn shell, context tracking, ~190 tests.

```
python3 ollamaquery2/ollamaquery2.py -c
```

See [ollamaquery2/README.md](ollamaquery2/README.md) for details.

## [ollamaquery/](ollamaquery/)

Original lightweight version — single-shot queries, batch processing, basic chat mode.

```
python3 ollamaquery/ollamaquery.py -I "hello" -m llama3
```

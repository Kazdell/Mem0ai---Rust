# Mem0ai but with Rust

A lightweight, 100% offline, in-process, standalone implementation of **Mem0** (long-term memory layer for AI agents) built entirely in Rust.

This repository is a customized standalone implementation designed to serve as a general-purpose offline memory solution.

## 🔗 References & Credits
* **Original Mem0 Repository (Python):** [mem0ai/mem0](https://github.com/mem0ai/mem0) - The official project that inspired this Rust version.

## 🚀 Features
* **In-Process Database:** Built on local JSON file flat-DB (`mem0_rust_db.json`) for zero runtime configuration and maximum speed.
* **Offline Embeddings:** Employs the `glowrs` crate (pure Rust Sentence Transformers using Hugging Face Candle) running locally on CPU. No external API keys required!
* **Multi-Mode Integration:**
  * **Interactive Console Menu:** Launch directly to select tasks via numeric inputs, resolving window self-closing issues on Windows.
  * **StdIO MCP Server:** Compatible with Model Context Protocol (MCP) for modern AI editors like Cursor, Windsurf, or Claude Desktop.
  * **Web Dashboard:** Embedded lightweight HTTP server (`tiny_http`) rendering a premium Glassmorphism UI to easily manage, search, and delete memories.
  * **CLI Subcommands:** Command-line options (`add`, `search`, `list`, `delete`, `clear`) for quick scripting and diagnostics.

## 🛠️ Usage
Refer to configuration details or run:
```powershell
.\mem0_rust_server.exe
```
and select options from the interactive terminal menu.

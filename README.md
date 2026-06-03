# Mem0 Local (Rust Standalone) v1.0.0

> **Offline, in-process, long-term memory layer for AI agents — Built 100% in Rust. No Docker. No API keys.**

A lightweight standalone implementation of [Mem0](https://github.com/mem0ai/mem0) designed to run entirely on your local machine. Stores vector embeddings in a plain JSON file and exposes them via CLI, Web Dashboard, or MCP (Model Context Protocol) for AI editor integration.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Offline Embeddings** | Uses `glowrs` (Rust Sentence Transformers via Hugging Face Candle) — pure CPU, no GPU needed |
| **Zero Dependencies at Runtime** | Single `.exe` file. No Docker, no Python, no external services |
| **MCP Server** | StdIO JSON-RPC server compatible with Cursor, Windsurf, Claude Desktop |
| **Web Dashboard** | Embedded HTTP server with Glassmorphism UI for managing memories visually |
| **CLI Subcommands** | `add`, `search`, `list`, `delete`, `clear` for scripting and automation |
| **Interactive Menu** | Auto-launched when no subcommand is given — prevents window self-closing on Windows |
| **Bilingual UI** | Toggle between English and Vietnamese in the interactive menu |
| **Multi-User Support** | Separate memory namespaces per `user_id` |

---

## 🖥️ System Requirements

| Requirement | Details |
|---|---|
| **OS** | Windows 10/11 x64 (primary), Linux/macOS (build from source) |
| **RAM** | ≥ 4 GB recommended (embedding model uses ~100–200 MB) |
| **Disk** | ~30 MB for the `.exe` + model files (auto-downloaded on first run) |
| **Internet** | Only needed on first run to download the embedding model from HuggingFace (then fully offline) |

> **To build from source:** Requires Rust toolchain (`rustup`) + `windres` and `ar` (from MinGW-w64) in your `PATH` on Windows.

---

## 🚀 Quick Start

### Option 1: Download Pre-built Binary
Download `mem0_rust_server.exe` from the [Releases page](https://github.com/mem0ai/mem0/releases) and run:

```powershell
.\mem0_rust_server.exe
```

The interactive menu will launch. On first use, the embedding model is automatically downloaded from HuggingFace (~90 MB).

### Option 2: Use a Local Model (Fully Offline)
Place the model files in a `model\` folder next to the `.exe`:

```
mem0_rust_server.exe
model\
  model.safetensors
  config.json
  tokenizer.json
```

The app detects the local model automatically and skips the network download.

### Option 3: Build from Source
```powershell
git clone https://github.com/mem0ai/mem0
cd mem0
cargo build --release
.\target\release\mem0_rust_server.exe
```

---

## 📁 Data Storage

All memories are stored in:
```
~/.mem0_rust/db.json
```

User preferences (language setting) are stored in:
```
~/.mem0_rust/config.json
```

---

## 🔧 CLI Reference

```powershell
# Add a new memory
.\mem0_rust_server.exe add "User prefers Rust for backend development" --user alice

# Semantic search
.\mem0_rust_server.exe search "favorite programming language" --user alice --limit 5

# List all memories for a user
.\mem0_rust_server.exe list --user alice

# Delete a memory by ID
.\mem0_rust_server.exe delete <uuid>

# Clear all memories for a user
.\mem0_rust_server.exe clear --user alice

# Launch Web Dashboard on port 8899
.\mem0_rust_server.exe dashboard --port 8899

# Launch MCP StdIO Server
.\mem0_rust_server.exe mcp
```

> **Default `--user` value:** `default` (used when `--user` is not specified).

---

## 🌐 Web Dashboard

Launch with:
```powershell
.\mem0_rust_server.exe dashboard
```
Then open your browser at: **http://localhost:8899**

Features:
- View, search, and delete memories in a Glassmorphism UI
- Real-time semantic search
- Dark/Light mode toggle

---

## 🤖 MCP Integration (for AI Editors)

### Claude Desktop / Cursor / Windsurf

Add to your MCP config file (e.g., `claude_desktop_config.json` or `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "mem0": {
      "command": "C:/path/to/mem0_rust_server.exe",
      "args": ["mcp"]
    }
  }
}
```

> **Config file locations:**
> - **Claude Desktop (Windows):** `%APPDATA%\Claude\claude_desktop_config.json`
> - **Cursor:** `.cursor/mcp.json` in your project root or `~/.cursor/mcp.json` globally
> - **Windsurf:** `~/.codeium/windsurf/mcp_config.json`

### Available MCP Tools

| Tool | Description | Required Args |
|---|---|---|
| `add_fact` | Add a new fact/memory | `fact`, `user_id` |
| `search_facts` | Semantic search for memories | `query`, `user_id` |
| `get_all_facts` | List all memories for a user | `user_id` |
| `delete_fact` | Delete a memory by ID | `fact_id` |
| `delete_all_facts` | Clear all memories for a user | `user_id` |

### Example MCP Call
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "add_fact",
    "arguments": {
      "fact": "User prefers concise code reviews",
      "user_id": "alice"
    }
  }
}
```

---

## 🏗️ Architecture

```
src/
├── main.rs        — CLI entry point, DB path resolution, command routing
├── config.rs      — Language preference persistence
├── db.rs          — JSON flat-file database (load/save)
├── embedding.rs   — Sentence Transformer init + cosine similarity
├── mcp.rs         — StdIO MCP JSON-RPC server (5 tools)
├── dashboard.rs   — Embedded HTTP server + Glassmorphism HTML/CSS/JS UI
└── interactive.rs — Interactive terminal menu (bilingual EN/VI)
build.rs           — Windows resource compiler (icon embedding via windres)
```

---

## 🔗 References & Credits

- **Original Mem0 (Python):** [mem0ai/mem0](https://github.com/mem0ai/mem0)
- **Embedding Engine:** [glowrs](https://crates.io/crates/glowrs) — Rust Sentence Transformers
- **Model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

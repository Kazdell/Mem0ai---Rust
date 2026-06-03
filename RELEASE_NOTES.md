## Mem0 Local (Rust Standalone) v1.0.0

### What's New
- **Portable**: No more hardcoded user IDs. Default user is now `default` — works for everyone
- **Universal DB path**: Data stored at `~/.mem0_rust/db.json` on any machine
- **MCP Server**: StdIO JSON-RPC compatible with Claude Desktop, Cursor, Windsurf
- **Web Dashboard**: Glassmorphism UI at `http://localhost:8899`
- **Windows app icon** embedded in executable
- **Full CLI**: `add`, `search`, `list`, `delete`, `clear` subcommands

### Quick Start
Download `mem0_rust_server-v1.0.0-windows-x64.exe` and run:

```powershell
.\mem0_rust_server.exe
```

### MCP Config (Claude Desktop / Cursor / Windsurf)

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

### System Requirements
- Windows 10/11 x64
- RAM: 4 GB+ recommended
- Internet: Only on first run (~90 MB model download, then fully offline)

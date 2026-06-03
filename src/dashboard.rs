use std::path::PathBuf;
use serde_json::json;
use uuid::Uuid;
use crate::db::{Database, MemoryRecord};
use crate::embedding::{init_embedder, generate_embedding, cosine_similarity};

fn parse_query_param(url: &str, name: &str) -> Option<String> {
    let parts: Vec<&str> = url.split('?').collect();
    if parts.len() < 2 {
        return None;
    }
    let query = parts[1];
    for pair in query.split('&') {
        let kv: Vec<&str> = pair.split('=').collect();
        if kv.len() == 2 && kv[0] == name {
            return Some(kv[1].to_string());
        }
    }
    None
}

pub fn run_dashboard(port: u16, db_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let server = tiny_http::Server::http(format!("127.0.0.1:{}", port))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    
    let url = format!("http://127.0.0.1:{}", port);
    println!("============================================================");
    println!("🚀 Mem0 Local Dashboard Server (Rust) is launching!");
    println!("🔗 Open your browser to access: {}", url);
    println!("============================================================");

    std::process::Command::new("cmd")
        .args(["/c", "start", &url])
        .spawn()
        .ok();

    let mut db = Database::load(db_path);
    let embedder = init_embedder()?;

    for mut request in server.incoming_requests() {
        let method = request.method().as_str();
        let path = request.url();

        match (method, path) {
            ("GET", "/" | "/index.html") => {
                let response = tiny_http::Response::from_string(HTML_TEMPLATE)
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap());
                let _ = request.respond(response);
            }
            ("GET", _) if path.starts_with("/api/facts") => {
                let user_id = parse_query_param(path, "user_id").unwrap_or_else(|| "acer".to_string());
                let results: Vec<serde_json::Value> = db.records.iter()
                    .filter(|r| r.user_id == user_id)
                    .map(|r| json!({
                        "id": r.id,
                        "text": r.text
                    }))
                    .collect();

                let res = json!({
                    "status": "success",
                    "results": results
                });
                let response = tiny_http::Response::from_string(res.to_string())
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap());
                let _ = request.respond(response);
            }
            ("POST", "/api/facts") => {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                if let Ok(data) = serde_json::from_str::<serde_json::Value>(&body) {
                    let fact = data["fact"].as_str().unwrap_or("").to_string();
                    let user_id = data["user_id"].as_str().unwrap_or("acer").to_string();
                    if !fact.is_empty() {
                        if let Ok(vector) = generate_embedding(&embedder, &fact) {
                            let fact_id = Uuid::new_v4().to_string();
                            db.records.push(MemoryRecord {
                                id: fact_id.clone(),
                                text: fact.clone(),
                                vector,
                                user_id: user_id.clone(),
                            });
                            let _ = db.save();

                            let res = json!({
                                "status": "success",
                                "memory_id": fact_id,
                                "fact": fact
                            });
                            let response = tiny_http::Response::from_string(res.to_string())
                                .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap());
                            let _ = request.respond(response);
                            continue;
                        }
                    }
                }
                let response = tiny_http::Response::from_string(json!({"status": "error", "message": "Invalid data"}).to_string())
                    .with_status_code(400)
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap());
                let _ = request.respond(response);
            }
            ("POST", "/api/search") => {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                if let Ok(data) = serde_json::from_str::<serde_json::Value>(&body) {
                    let query = data["query"].as_str().unwrap_or("").to_string();
                    let user_id = data["user_id"].as_str().unwrap_or("acer").to_string();
                    let limit = data["limit"].as_u64().unwrap_or(10) as usize;

                    if !query.is_empty() {
                        if let Ok(query_vector) = generate_embedding(&embedder, &query) {
                            let mut matches: Vec<(f32, MemoryRecord)> = db.records.iter()
                                .filter(|r| r.user_id == user_id)
                                .map(|r| {
                                    let score = cosine_similarity(&query_vector, &r.vector);
                                    (score, r.clone())
                                })
                                .collect();

                            matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                            let results: Vec<serde_json::Value> = matches.into_iter()
                                .take(limit)
                                .map(|(score, record)| json!({
                                    "id": record.id,
                                    "score": score,
                                    "text": record.text
                                }))
                                .collect();

                            let res = json!({
                                "status": "success",
                                "query": query,
                                "results": results
                            });
                            let response = tiny_http::Response::from_string(res.to_string())
                                .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap());
                            let _ = request.respond(response);
                            continue;
                        }
                    }
                }
                let response = tiny_http::Response::from_string(json!({"status": "error", "message": "Invalid query"}).to_string())
                    .with_status_code(400)
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap());
                let _ = request.respond(response);
            }
            ("POST", _) if path.starts_with("/api/facts/clear") => {
                let user_id = parse_query_param(path, "user_id").unwrap_or_else(|| "acer".to_string());
                db.records.retain(|r| r.user_id != user_id);
                let _ = db.save();

                let res = json!({
                    "status": "success",
                    "message": format!("Cleared all memories for user {}", user_id)
                });
                let response = tiny_http::Response::from_string(res.to_string())
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap());
                let _ = request.respond(response);
            }
            ("DELETE", _) if path.starts_with("/api/facts") => {
                let id = parse_query_param(path, "id").unwrap_or_default();
                let original_len = db.records.len();
                db.records.retain(|r| r.id != id);
                let res = if db.records.len() < original_len {
                    let _ = db.save();
                    json!({"status": "success", "message": format!("Deleted memory {}", id)})
                } else {
                    json!({"status": "error", "message": format!("Memory ID {} not found", id)})
                };
                let response = tiny_http::Response::from_string(res.to_string())
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap());
                let _ = request.respond(response);
            }
            _ => {
                let response = tiny_http::Response::from_string("Not Found")
                    .with_status_code(404);
                let _ = request.respond(response);
            }
        }
    }
    Ok(())
}

const HTML_TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mem0 Local - Memory Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            min-height: 100vh;
            color: #f8fafc;
        }
        .glass {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .glow-hover:hover {
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }
    </style>
</head>
<body class="p-6 md:p-12">
    <div class="max-w-6xl mx-auto">
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-center mb-10 gap-4">
            <div>
                <h1 class="text-4xl font-extrabold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent flex items-center gap-3">
                    <i class="fa-solid fa-brain text-indigo-400"></i> Mem0 Local Dashboard
                </h1>
                <p class="text-slate-400 mt-2">Offline long-term memory management dashboard for AI agents (Rust Backend)</p>
            </div>
            <div class="flex gap-4">
                <div class="glass px-6 py-3 rounded-2xl flex items-center gap-3">
                    <span class="text-sm text-slate-400">Total Facts:</span>
                    <span id="fact-count" class="text-2xl font-bold text-indigo-400">0</span>
                </div>
                <button onclick="clearAllMemories()" class="px-5 py-3 rounded-2xl border border-red-500/30 text-red-400 hover:bg-red-500/10 transition flex items-center gap-2">
                    <i class="fa-solid fa-trash-can"></i> Clear All
                </button>
            </div>
        </header>

        <!-- Main Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Left Side: Controls -->
            <div class="lg:col-span-1 space-y-6">
                <!-- Add Fact Card -->
                <div class="glass p-6 rounded-3xl shadow-xl">
                    <h2 class="text-xl font-semibold mb-4 text-purple-300 flex items-center gap-2">
                        <i class="fa-solid fa-plus-circle"></i> Add New Memory
                    </h2>
                    <div class="space-y-4">
                        <textarea id="new-fact" rows="4" placeholder="Example: User prefers PostgreSQL and Rust for API development..." 
                            class="w-full px-4 py-3 rounded-2xl bg-slate-900/60 border border-slate-700/50 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500/80 transition resize-none"></textarea>
                        <button onclick="saveMemory()" class="w-full py-3.5 rounded-2xl bg-gradient-to-r from-indigo-500 to-purple-600 font-semibold text-white glow-hover transition duration-300 flex justify-center items-center gap-2">
                            <i class="fa-solid fa-cloud-arrow-up"></i> Save to Memory
                        </button>
                    </div>
                </div>

                <!-- Search Card -->
                <div class="glass p-6 rounded-3xl shadow-xl">
                    <h2 class="text-xl font-semibold mb-4 text-pink-300 flex items-center gap-2">
                        <i class="fa-solid fa-magnifying-glass"></i> Semantic Search
                    </h2>
                    <div class="space-y-4">
                        <div class="relative">
                            <input id="search-query" type="text" placeholder="Enter search keywords..." 
                                class="w-full pl-4 pr-10 py-3 rounded-2xl bg-slate-900/60 border border-slate-700/50 text-white placeholder-slate-500 focus:outline-none focus:border-pink-500/80 transition">
                            <button onclick="searchMemories()" class="absolute right-3 top-3.5 text-slate-400 hover:text-pink-400">
                                <i class="fa-solid fa-arrow-right"></i>
                            </button>
                        </div>
                        <button onclick="resetSearch()" id="btn-reset-search" class="w-full py-2.5 rounded-2xl border border-slate-700/80 text-sm text-slate-400 hover:bg-slate-800/40 transition hidden">
                            Back to All Memories
                        </button>
                    </div>
                </div>
            </div>

            <!-- Right Side: Memory List -->
            <div class="lg:col-span-2 space-y-4">
                <div class="flex justify-between items-center mb-2">
                    <h2 id="list-title" class="text-2xl font-bold text-slate-200">Current Memories</h2>
                    <span id="search-indicator" class="text-sm bg-pink-500/10 border border-pink-500/30 text-pink-400 px-3 py-1 rounded-full hidden">Search Results</span>
                </div>
                <div id="memories-container" class="space-y-4 max-h-[650px] overflow-y-auto pr-2">
                    <!-- Cards will be dynamically injected here -->
                </div>
            </div>
        </div>
    </div>

    <!-- Script JavaScript -->
    <script>
        const USER_ID = 'acer';

        // Initial Load
        document.addEventListener('DOMContentLoaded', () => {
            fetchMemories();
            
            // Search on Enter
            document.getElementById('search-query').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') searchMemories();
            });
        });

        // Fetch Memories
        async function fetchMemories() {
            try {
                const response = await fetch(`/api/facts?user_id=${USER_ID}`);
                const data = await response.json();
                if (data.status === 'success') {
                    renderMemories(data.results);
                }
            } catch (err) {
                console.error("Fetch error:", err);
            }
        }

        // Render Memory List
        function renderMemories(facts, isSearchResult = false) {
            const container = document.getElementById('memories-container');
            const countSpan = document.getElementById('fact-count');
            
            if (!isSearchResult) {
                countSpan.textContent = facts.length;
                document.getElementById('list-title').textContent = "Current Memories";
                document.getElementById('search-indicator').classList.add('hidden');
                document.getElementById('btn-reset-search').classList.add('hidden');
            } else {
                document.getElementById('list-title').textContent = "Semantic Search Results";
                document.getElementById('search-indicator').classList.remove('hidden');
                document.getElementById('btn-reset-search').classList.remove('hidden');
            }

            container.innerHTML = '';
            
            if (facts.length === 0) {
                container.innerHTML = `
                    <div class="glass p-12 rounded-3xl text-center text-slate-500">
                        <i class="fa-solid fa-box-open text-5xl mb-4 text-slate-600"></i>
                        <p class="text-lg">No memories found.</p>
                    </div>
                `;
                return;
            }

            facts.forEach(item => {
                const scoreText = item.score ? `<span class="text-xs bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 px-2 py-1 rounded-lg">Score: ${(item.score * 100).toFixed(1)}%</span>` : '';
                container.innerHTML += `
                    <div class="glass p-6 rounded-2xl shadow-md flex justify-between items-start gap-4 transition hover:bg-slate-800/30 hover:border-slate-700/60 duration-200">
                        <div class="space-y-2 flex-1">
                            <p class="text-slate-100 leading-relaxed text-lg">${item.text}</p>
                            <div class="flex items-center gap-3">
                                <span class="text-xs text-slate-500 font-mono">ID: ${item.id}</span>
                                ${scoreText}
                            </div>
                        </div>
                        <button onclick="deleteMemory('${item.id}')" class="text-slate-500 hover:text-red-400 p-2 rounded-xl hover:bg-red-500/10 transition" title="Delete Fact">
                            <i class="fa-solid fa-trash"></i>
                        </button>
                    </div>
                `;
            });
        }

        // Save Fact
        async function saveMemory() {
            const textarea = document.getElementById('new-fact');
            const fact = textarea.value.trim();
            if (!fact) return;

            try {
                const response = await fetch('/api/facts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fact, user_id: USER_ID })
                });
                const res = await response.json();
                if (res.status === 'success') {
                    textarea.value = '';
                    fetchMemories();
                }
            } catch (err) {
                console.error("Save error:", err);
            }
        }

        // Semantic Search
        async function searchMemories() {
            const queryInput = document.getElementById('search-query');
            const query = queryInput.value.trim();
            if (!query) return;

            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, user_id: USER_ID })
                });
                const res = await response.json();
                if (res.status === 'success') {
                    renderMemories(res.results, true);
                }
            } catch (err) {
                console.error("Search error:", err);
            }
        }

        // Reset Search
        function resetSearch() {
            document.getElementById('search-query').value = '';
            fetchMemories();
        }

        // Delete Fact
        async function deleteMemory(id) {
            if (!confirm("Are you sure you want to delete this memory?")) return;
            try {
                const response = await fetch(`/api/facts?id=${id}`, { method: 'DELETE' });
                const res = await response.json();
                if (res.status === 'success') {
                    fetchMemories();
                }
            } catch (err) {
                console.error("Delete error:", err);
            }
        }

        // Clear All
        async function clearAllMemories() {
            if (!confirm("WARNING: This will permanently clear ALL memories! Continue?")) return;
            try {
                const response = await fetch(`/api/facts/clear?user_id=${USER_ID}`, { method: 'POST' });
                const res = await response.json();
                if (res.status === 'success') {
                    fetchMemories();
                }
            } catch (err) {
                console.error("Clear error:", err);
            }
        }
    </script>
</body>
</html>"#;

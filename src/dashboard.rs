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
            ("GET", "/logo.png") => {
                let logo_bytes = include_bytes!("../logo.png");
                let response = tiny_http::Response::from_data(logo_bytes.to_vec())
                    .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"image/png"[..]).unwrap());
                let _ = request.respond(response);
            }
            ("GET", _) if path.starts_with("/api/facts") => {
                let user_id = parse_query_param(path, "user_id").unwrap_or_else(|| "default".to_string());
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
                    let user_id = data["user_id"].as_str().unwrap_or("default").to_string();
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
                    let user_id = data["user_id"].as_str().unwrap_or("default").to_string();
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
                let user_id = parse_query_param(path, "user_id").unwrap_or_else(|| "default".to_string());
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
    <link rel="icon" type="image/png" href="/logo.png">
    <link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        primary: '#1A1C1E',     // Deep ink
                        secondary: '#6C7278',   // Slate border & metadata
                        tertiary: '#B8422E',    // Boston Clay màu nhấn đỏ gạch
                        neutral: '#F7F5F2',     // Warm limestone nền
                    },
                    fontFamily: {
                        sans: ['"Public Sans"', 'sans-serif'],
                        mono: ['"Space Grotesk"', 'monospace'],
                    },
                    borderRadius: {
                        sm: '4px',
                        md: '8px',
                        lg: '16px',
                    }
                }
            }
        }
    </script>
    <style>
        body {
            font-family: 'Public Sans', sans-serif;
            background-color: #F7F5F2;
            color: #1A1C1E;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .dark body {
            background-color: #1A1C1E;
            color: #F7F5F2;
        }
        .font-space-grotesk {
            font-family: 'Space Grotesk', sans-serif;
        }
        .transition-all {
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Custom scrollbar to match the design system */
        #memories-container::-webkit-scrollbar {
            width: 6px;
        }
        #memories-container::-webkit-scrollbar-track {
            background: transparent;
        }
        #memories-container::-webkit-scrollbar-thumb {
            background-color: rgba(108, 114, 120, 0.25);
            border-radius: 3px;
        }
        #memories-container::-webkit-scrollbar-thumb:hover {
            background-color: rgba(108, 114, 120, 0.45);
        }
        .dark #memories-container::-webkit-scrollbar-thumb {
            background-color: rgba(255, 255, 255, 0.15);
        }
        .dark #memories-container::-webkit-scrollbar-thumb:hover {
            background-color: rgba(255, 255, 255, 0.3);
        }
    </style>
</head>
<body class="bg-neutral text-primary dark:bg-primary dark:text-neutral p-6 md:p-12 font-sans selection:bg-tertiary/10 selection:text-tertiary transition-all duration-300">
    <div class="max-w-6xl mx-auto">
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center mb-12 pb-6 border-b border-secondary/15 dark:border-secondary/30 gap-4">
            <div>
                <h1 class="text-3xl md:text-4xl font-bold tracking-tight text-primary dark:text-neutral flex items-center gap-3">
                    <img src="/logo.png" alt="Mem0 Logo" class="w-8 h-8 object-contain"> Mem0 Local Dashboard
                </h1>
                <p class="text-secondary dark:text-secondary/80 mt-2 text-sm md:text-base font-normal">Offline long-term memory management dashboard for AI agents (Rust Backend)</p>
            </div>
            <div class="flex items-center gap-3 flex-wrap">
                <!-- Total Facts Badge aligned to h-10 -->
                <div class="h-10 px-4 bg-white dark:bg-[#252729] border border-secondary/15 dark:border-secondary/30 rounded-md flex items-center gap-2 shadow-sm transition-all duration-300">
                    <span class="text-[10px] text-secondary dark:text-secondary/80 font-bold tracking-wider font-mono uppercase">Total Facts</span>
                    <span id="fact-count" class="text-base font-bold text-tertiary font-mono">0</span>
                </div>
                <!-- Theme Toggle Button aligned to w-10 h-10 -->
                <button id="theme-toggle" onclick="toggleTheme()" class="w-10 h-10 rounded-md border border-secondary/30 dark:border-secondary/50 text-secondary dark:text-secondary/80 hover:bg-secondary/5 dark:hover:bg-secondary/10 transition-all flex items-center justify-center shadow-sm" title="Toggle Light/Dark Mode">
                    <i id="theme-icon" class="fa-solid fa-moon"></i>
                </button>
                <!-- Clear All Button aligned to h-10 -->
                <button onclick="clearAllMemories()" class="h-10 px-4 rounded-md border border-tertiary/30 text-tertiary hover:bg-tertiary/5 dark:hover:bg-tertiary/10 transition-all text-sm font-semibold flex items-center gap-2 shadow-sm">
                    <i class="fa-solid fa-trash-can"></i> Clear All
                </button>
            </div>
        </header>

        <!-- Main Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Left Side: Controls -->
            <div class="lg:col-span-1 space-y-6">
                <!-- Add Fact Card -->
                <div class="bg-white dark:bg-[#252729] border border-secondary/15 dark:border-secondary/30 p-6 md:p-8 rounded-lg shadow-sm transition-all duration-300">
                    <h2 class="text-lg font-bold mb-4 text-primary dark:text-neutral tracking-tight flex items-center gap-2">
                        <i class="fa-solid fa-circle-plus text-tertiary"></i> Add New Memory
                    </h2>
                    <div class="space-y-4">
                        <textarea id="new-fact" rows="4" placeholder="Example: User prefers PostgreSQL and Rust for API development..." 
                            class="w-full px-4 py-3 rounded-md bg-neutral/30 border border-secondary/20 dark:bg-primary/40 dark:border-secondary/40 text-primary dark:text-neutral placeholder-secondary/50 focus:outline-none focus:border-tertiary/60 transition-all resize-none text-sm leading-relaxed"></textarea>
                        <button onclick="saveMemory()" class="w-full py-3 rounded-md bg-tertiary font-semibold text-white hover:bg-tertiary/90 shadow-sm hover:shadow transition-all duration-200 flex justify-center items-center gap-2 text-sm tracking-wide">
                            <i class="fa-solid fa-cloud-arrow-up"></i> Save to Memory
                        </button>
                    </div>
                </div>

                <!-- Search Card -->
                <div class="bg-white dark:bg-[#252729] border border-secondary/15 dark:border-secondary/30 p-6 md:p-8 rounded-lg shadow-sm transition-all duration-300">
                    <h2 class="text-lg font-bold mb-4 text-primary dark:text-neutral tracking-tight flex items-center gap-2">
                        <i class="fa-solid fa-magnifying-glass text-tertiary"></i> Semantic Search
                    </h2>
                    <div class="space-y-4">
                        <div class="relative">
                            <input id="search-query" type="text" placeholder="Enter search query..." 
                                class="w-full pl-4 pr-10 py-3 rounded-md bg-neutral/30 border border-secondary/20 dark:bg-primary/40 dark:border-secondary/40 text-primary dark:text-neutral placeholder-secondary/50 focus:outline-none focus:border-tertiary/60 transition-all text-sm">
                            <button onclick="searchMemories()" class="absolute right-3 top-3.5 text-secondary hover:text-tertiary transition-colors">
                                <i class="fa-solid fa-arrow-right"></i>
                            </button>
                        </div>
                        <button onclick="resetSearch()" id="btn-reset-search" class="w-full py-2.5 rounded-md border border-secondary/30 text-xs font-semibold text-secondary dark:text-secondary/80 hover:bg-neutral dark:hover:bg-primary/50 transition-all hidden">
                            Back to All Memories
                        </button>
                    </div>
                </div>
            </div>

            <!-- Right Side: Memory List -->
            <div class="lg:col-span-2 space-y-4">
                <div class="flex justify-between items-center mb-2">
                    <h2 id="list-title" class="text-xl font-bold tracking-tight text-primary dark:text-neutral">Current Memories</h2>
                    <span id="search-indicator" class="text-xs bg-tertiary/10 border border-tertiary/20 text-tertiary px-3 py-1 rounded-full font-semibold hidden font-mono">SEARCH RESULTS</span>
                </div>
                <div id="memories-container" class="space-y-4 max-h-[650px] overflow-y-auto pr-2">
                    <!-- Cards will be dynamically injected here -->
                </div>
            </div>
        </div>
    </div>

    <!-- Script JavaScript -->
    <script>
        const USER_ID = 'default';
        let memories = [];
        let isSearching = false;

        // Theme Toggle Logic
        function initTheme() {
            const theme = localStorage.getItem('theme');
            const icon = document.getElementById('theme-icon');
            if (theme === 'dark' || (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                document.documentElement.classList.add('dark');
                if (icon) icon.className = 'fa-solid fa-sun';
            } else {
                document.documentElement.classList.remove('dark');
                if (icon) icon.className = 'fa-solid fa-moon';
            }
        }

        fn toggleTheme() {
            const icon = document.getElementById('theme-icon');
            if (document.documentElement.classList.contains('dark')) {
                document.documentElement.classList.remove('dark');
                localStorage.setItem('theme', 'light');
                if (icon) icon.className = 'fa-solid fa-moon';
                showToast("Switched to Light Mode", "success");
            } else {
                document.documentElement.classList.add('dark');
                localStorage.setItem('theme', 'dark');
                if (icon) icon.className = 'fa-solid fa-sun';
                showToast("Switched to Dark Mode", "success");
            }
        }

        // Initial Load
        document.addEventListener('DOMContentLoaded', () => {
            initTheme();
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
                    memories = data.results;
                    isSearching = false;
                    renderMemories(memories);
                }
            } catch (err) {
                console.error("Fetch error:", err);
                showToast("Cannot connect to server", "error");
            }
        }

        // Render Memory List
        function renderMemories(facts, isSearchResult = false) {
            const container = document.getElementById('memories-container');
            const countSpan = document.getElementById('fact-count');
            
            if (!isSearchResult) {
                countSpan.textContent = facts.filter(f => !f.syncing).length;
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
                    <div class="bg-white dark:bg-[#252729] border border-secondary/10 dark:border-secondary/30 p-12 rounded-lg text-center text-secondary dark:text-secondary/80 transition-all duration-300">
                        <i class="fa-solid fa-box-open text-4xl mb-4 text-secondary/60"></i>
                        <p class="text-lg">No memories found.</p>
                    </div>
                `;
                return;
            }

            facts.forEach(item => {
                const scoreText = item.score ? `<span class="text-xs bg-tertiary/10 text-tertiary border border-tertiary/20 px-2 py-0.5 rounded font-mono">Score: ${(item.score * 100).toFixed(1)}%</span>` : '';
                const syncingLabel = item.syncing ? `
                    <span class="text-xs bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 px-2 py-0.5 rounded font-mono flex items-center gap-1 animate-pulse">
                        <span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span> Syncing
                    </span>` : '';
                
                const opacityClass = item.syncing ? 'opacity-70 pointer-events-none' : '';
                
                container.innerHTML += `
                    <div class="bg-white dark:bg-[#252729] border border-secondary/15 dark:border-secondary/30 p-6 rounded-lg shadow-sm flex justify-between items-start gap-4 transition-all hover:-translate-y-0.5 hover:shadow-md dark:hover:shadow-black/30 duration-200 ${opacityClass}">
                        <div class="space-y-3 flex-1">
                            <p class="text-primary dark:text-neutral leading-relaxed text-base md:text-lg">${item.text}</p>
                            <div class="flex flex-wrap items-center gap-3">
                                <span class="text-[11px] text-secondary dark:text-secondary/70 font-mono">ID: ${item.id}</span>
                                ${scoreText}
                                ${syncingLabel}
                            </div>
                        </div>
                        <button onclick="deleteMemory('${item.id}')" class="text-secondary dark:text-secondary/70 hover:text-tertiary p-2 rounded-md hover:bg-tertiary/5 dark:hover:bg-tertiary/10 transition-all" title="Delete Fact">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </div>
                `;
            });
        }

        // Save Fact (Optimistic UI)
        async function saveMemory() {
            const textarea = document.getElementById('new-fact');
            const fact = textarea.value.trim();
            if (!fact) return;

            textarea.value = '';

            const tempId = 'temp-' + Date.now();
            const tempRecord = {
                id: tempId,
                text: fact,
                syncing: true
            };

            const previousMemories = [...memories];

            if (!isSearching) {
                memories.unshift(tempRecord);
                renderMemories(memories);
            } else {
                isSearching = false;
                memories.unshift(tempRecord);
                document.getElementById('search-query').value = '';
                renderMemories(memories);
            }

            try {
                const response = await fetch('/api/facts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fact, user_id: USER_ID })
                });
                const res = await response.json();
                if (res.status === 'success') {
                    const index = memories.findIndex(m => m.id === tempId);
                    if (index !== -1) {
                        memories[index].id = res.memory_id;
                        memories[index].text = res.fact || fact;
                        delete memories[index].syncing;
                        renderMemories(memories);
                    }
                    showToast("Memory saved successfully!", "success");
                } else {
                    throw new Error(res.message || "Failed to save memory");
                }
            } catch (err) {
                console.error("Save error:", err);
                memories = previousMemories;
                renderMemories(memories);
                showToast("Failed to save memory. Pls try again.", "error");
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
                    isSearching = true;
                    renderMemories(res.results, true);
                }
            } catch (err) {
                console.error("Search error:", err);
                showToast("Search failed", "error");
            }
        }

        // Reset Search
        function resetSearch() {
            document.getElementById('search-query').value = '';
            isSearching = false;
            renderMemories(memories);
        }

        // Delete Fact (Optimistic UI)
        async function deleteMemory(id) {
            if (id.startsWith('temp-')) return;
            if (!confirm("Are you sure you want to delete this memory?")) return;

            const index = memories.findIndex(m => m.id === id);
            if (index === -1) return;

            const previousMemories = [...memories];

            memories.splice(index, 1);
            renderMemories(memories);

            try {
                const response = await fetch(`/api/facts?id=${id}`, { method: 'DELETE' });
                const res = await response.json();
                if (res.status === 'success') {
                    showToast("Memory deleted successfully!", "success");
                } else {
                    throw new Error(res.message || "Failed to delete memory");
                }
            } catch (err) {
                console.error("Delete error:", err);
                memories = previousMemories;
                renderMemories(memories);
                showToast("Failed to delete memory. Rolled back.", "error");
            }
        }

        // Clear All (Optimistic UI)
        async function clearAllMemories() {
            if (memories.length === 0) return;
            if (!confirm("WARNING: This will permanently clear ALL memories! Continue?")) return;

            const previousMemories = [...memories];

            memories = [];
            renderMemories(memories);

            try {
                const response = await fetch(`/api/facts/clear?user_id=${USER_ID}`, { method: 'POST' });
                const res = await response.json();
                if (res.status === 'success') {
                    showToast("All memories cleared!", "success");
                } else {
                    throw new Error(res.message || "Failed to clear memories");
                }
            } catch (err) {
                console.error("Clear error:", err);
                memories = previousMemories;
                renderMemories(memories);
                showToast("Failed to clear memories. Rolled back.", "error");
            }
        }

        // Helper Toast Notification
        function showToast(message, type = "success") {
            const toast = document.createElement('div');
            toast.className = `fixed bottom-5 right-5 px-6 py-3.5 rounded-md shadow-lg dark:shadow-black/30 transition-all transform translate-y-0 opacity-100 font-sans z-50 text-sm font-medium border flex items-center gap-2 duration-300`;
            
            if (type === "success") {
                toast.className += " bg-white dark:bg-[#252729] border-emerald-500/30 dark:border-emerald-500/50 text-emerald-700 dark:text-emerald-400";
                toast.innerHTML = `<i class="fa-solid fa-circle-check text-emerald-500"></i> ${message}`;
            } else {
                toast.className += " bg-white dark:bg-[#252729] border-tertiary/30 dark:border-tertiary/50 text-tertiary dark:text-tertiary/80";
                toast.innerHTML = `<i class="fa-solid fa-circle-exclamation text-tertiary"></i> ${message}`;
            }
            
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.classList.add('opacity-0', 'translate-y-2');
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }
    </script>
</body>
</html>"#;

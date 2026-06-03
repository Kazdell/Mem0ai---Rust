use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;
use glowrs::{SentenceTransformer, Device, PoolingStrategy};

// Cấu trúc một bản ghi ký ức
#[derive(Serialize, Deserialize, Clone, Debug)]
struct MemoryRecord {
    id: String,
    text: String,
    vector: Vec<f32>,
    user_id: String,
}

// Trình quản lý database file-based cục bộ
struct Database {
    path: PathBuf,
    records: Vec<MemoryRecord>,
}

impl Database {
    fn load(path: PathBuf) -> Self {
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).ok();
            }
            return Database { path, records: Vec::new() };
        }
        let data = fs::read_to_string(&path).unwrap_or_default();
        let records = serde_json::from_str(&data).unwrap_or_else(|_| Vec::new());
        Database { path, records }
    }

    fn save(&self) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(&self.records)?;
        fs::write(&self.path, data)?;
        Ok(())
    }
}

// Tính Cosine Similarity thuần Rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot_product / (norm_a * norm_b)
}

// Khởi tạo mô hình Embedding cục bộ offline bằng Candle (glowrs)
fn init_embedder() -> Result<SentenceTransformer, io::Error> {
    // Tìm thư mục model bên cạnh file thực thi exe
    let mut exe_dir = std::env::current_exe()?;
    exe_dir.pop(); // Lấy thư mục chứa file exe
    let local_model_dir = exe_dir.join("model");

    let embedder = if local_model_dir.join("model.safetensors").exists()
        && local_model_dir.join("config.json").exists()
        && local_model_dir.join("tokenizer.json").exists()
    {
        eprintln!("⚙️ Phát hiện mô hình cục bộ tại {:?}. Đang khởi tạo...", local_model_dir);
        SentenceTransformer::from_folder(&local_model_dir, &Device::Cpu)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
    } else {
        eprintln!("⚙️ Không tìm thấy mô hình cục bộ bên cạnh file exe. Tiến hành kiểm tra/tải từ HuggingFace...");
        SentenceTransformer::from_repo_string(
            "sentence-transformers/all-MiniLM-L6-v2",
            &Device::Cpu,
        ).map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
    };

    eprintln!("✅ Mô hình Embedding đã sẵn sàng!");
    Ok(embedder)
}

// Sinh embedding cho một văn bản
fn generate_embedding(embedder: &SentenceTransformer, text: &str) -> Result<Vec<f32>, io::Error> {
    let sentences = vec![text.to_string()];
    // Encode batch và normalize vector đầu ra (phục vụ Cosine Similarity)
    let embeddings_tensor = embedder.encode_batch(sentences, true, PoolingStrategy::Mean)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    
    // Chuyển đổi Candle Tensor sang Vec<Vec<f32>>
    let embeddings: Vec<Vec<f32>> = embeddings_tensor.to_vec2()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    
    if let Some(vector) = embeddings.first() {
        Ok(vector.clone())
    } else {
        Err(io::Error::new(io::ErrorKind::Other, "Không thể sinh vector nhúng"))
    }
}

// --- CLI DEFINITIONS ---
#[derive(Parser, Debug)]
#[command(name = "mem0_rust", version = "1.0", about = "Mem0 Local Server & CLI in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Thêm một Fact mới thủ công
    Add {
        fact: String,
        #[arg(short, long, default_value = "acer")]
        user: String,
    },
    /// Tìm kiếm ngữ nghĩa các Fact
    Search {
        query: String,
        #[arg(short, long, default_value = "acer")]
        user: String,
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
    },
    /// Liệt kê tất cả các Fact của user
    List {
        #[arg(short, long, default_value = "acer")]
        user: String,
    },
    /// Xóa một Fact dựa trên ID
    Delete {
        id: String,
    },
    /// Xóa sạch tất cả Fact của user
    Clear {
        #[arg(short, long, default_value = "acer")]
        user: String,
    },
    /// Khởi chạy Web Dashboard trực tiếp từ Rust
    Dashboard {
        #[arg(short, long, default_value_t = 8899)]
        port: u16,
    },
    /// Khởi chạy StdIO MCP Server cho AI Client/Editor
    Mcp,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Định vị đường dẫn DB
    let home = dirs::home_dir().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Không tìm thấy thư mục Home"))?;
    let db_path = home.join(".gemini").join("antigravity").join("mem0_rust_db.json");

    if let Some(cmd) = cli.command {
        match cmd {
            Commands::Add { fact, user } => {
                let mut db = Database::load(db_path);
                let embedder = init_embedder()?;
                let vector = generate_embedding(&embedder, &fact)?;
                let fact_id = Uuid::new_v4().to_string();
                db.records.push(MemoryRecord {
                    id: fact_id.clone(),
                    text: fact.clone(),
                    vector,
                    user_id: user.clone(),
                });
                db.save()?;
                let result = json!({
                    "status": "success",
                    "memory_id": fact_id,
                    "fact": fact,
                    "user_id": user
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Commands::Search { query, user, limit } => {
                let db = Database::load(db_path);
                let embedder = init_embedder()?;
                let query_vector = generate_embedding(&embedder, &query)?;
                let mut matches: Vec<(f32, MemoryRecord)> = db.records.iter()
                    .filter(|r| r.user_id == user)
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

                let result = json!({
                    "status": "success",
                    "query": query,
                    "results": results
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Commands::List { user } => {
                let db = Database::load(db_path);
                let results: Vec<serde_json::Value> = db.records.iter()
                    .filter(|r| r.user_id == user)
                    .map(|r| json!({
                        "id": r.id,
                        "text": r.text
                    }))
                    .collect();

                let result = json!({
                    "status": "success",
                    "results": results
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Commands::Delete { id } => {
                let mut db = Database::load(db_path);
                let original_len = db.records.len();
                db.records.retain(|r| r.id != id);
                if db.records.len() < original_len {
                    db.save()?;
                    let result = json!({"status": "success", "message": format!("Deleted memory {}", id)});
                    println!("{}", serde_json::to_string_pretty(&result)?);
                } else {
                    let result = json!({"status": "error", "message": format!("Memory ID {} not found", id)});
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            }
            Commands::Clear { user } => {
                let mut db = Database::load(db_path);
                db.records.retain(|r| r.user_id != user);
                db.save()?;
                let result = json!({"status": "success", "message": format!("Cleared all memories for user {}", user)});
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Commands::Dashboard { port } => {
                run_dashboard(port, db_path)?;
            }
            Commands::Mcp => {
                // --- CHẾ ĐỘ MCP SERVER (StdIO JSON-RPC 2.0) ---
                run_mcp_server(db_path).await?;
            }
        }
    } else {
        // --- KHÔNG TRUYỀN SUBCOMMAND: Khởi chạy Menu Tương Tác Trực Quan ---
        run_interactive_menu(db_path).await?;
    }

    Ok(())
}

// Xử lý logic MCP Server qua Stdin/Stdout
async fn run_mcp_server(db_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin);
    let mut line = String::new();

    // Khởi tạo trễ (Lazy init) để server khởi động ngay lập tức trước khi tải mô hình embedding
    let mut embedder_opt: Option<SentenceTransformer> = None;

    while reader.read_line(&mut line)? > 0 {
        let req_str = line.trim();
        if req_str.is_empty() {
            line.clear();
            continue;
        }

        // Parse JSON-RPC Request
        if let Ok(req) = serde_json::from_str::<serde_json::Value>(req_str) {
            let id = &req["id"];
            let method = req["method"].as_str().unwrap_or("");

            match method {
                "initialize" => {
                    let response = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {}
                            },
                            "serverInfo": {
                                "name": "Antigravity-Mem0-Rust",
                                "version": "1.0.0"
                            }
                        }
                    });
                    println!("{}", response.to_string());
                }
                "tools/list" => {
                    let response = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "tools": [
                                {
                                    "name": "add_fact",
                                    "description": "Thêm một Fact (sự thật/thông tin ký ức ngắn) đã được trích xuất vào bộ nhớ dài hạn của người dùng.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "fact": {
                                                "type": "string",
                                                "description": "Nội dung Fact ngắn gọn (Ví dụ: \"User thích code bằng Rust\")."
                                            },
                                            "user_id": {
                                                "type": "string",
                                                "description": "ID định danh của người dùng (Ví dụ: \"acer\")."
                                            }
                                        },
                                        "required": ["fact", "user_id"]
                                    }
                                },
                                {
                                    "name": "search_facts",
                                    "description": "Tìm kiếm ngữ nghĩa các Fact liên quan trong bộ nhớ dài hạn của người dùng.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {
                                                "type": "string",
                                                "description": "Câu hỏi hoặc từ khóa tìm kiếm (Ví dụ: \"ngôn ngữ lập trình yêu thích\")."
                                            },
                                            "user_id": {
                                                "type": "string",
                                                "description": "ID định danh của người dùng."
                                            },
                                            "limit": {
                                                "type": "integer",
                                                "description": "Số lượng kết quả tối đa trả về (mặc định: 5)."
                                            }
                                        },
                                        "required": ["query", "user_id"]
                                    }
                                },
                                {
                                    "name": "get_all_facts",
                                    "description": "Lấy toàn bộ danh sách Fact đang được lưu trữ của người dùng.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "user_id": {
                                                "type": "string",
                                                "description": "ID định danh của người dùng."
                                            }
                                        },
                                        "required": ["user_id"]
                                    }
                                },
                                {
                                    "name": "delete_fact",
                                    "description": "Xóa một Fact cụ thể khỏi bộ nhớ dài hạn dựa trên Memory ID.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "fact_id": {
                                                "type": "string",
                                                "description": "ID của Fact cần xóa."
                                            }
                                        },
                                        "required": ["fact_id"]
                                    }
                                },
                                {
                                    "name": "delete_all_facts",
                                    "description": "Xóa sạch toàn bộ ký ức của người dùng cụ thể.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "user_id": {
                                                "type": "string",
                                                "description": "ID định danh của người dùng cần xóa sạch bộ nhớ."
                                            }
                                        },
                                        "required": ["user_id"]
                                    }
                                }
                            ]
                        }
                    });
                    println!("{}", response.to_string());
                }
                "tools/call" => {
                    let tool_name = req["params"]["name"].as_str().unwrap_or("");
                    let args = &req["params"]["arguments"];

                    // Tải mô hình Embedding trễ (chỉ khi có tool call đầu tiên)
                    if embedder_opt.is_none() {
                        match init_embedder() {
                            Ok(emb) => embedder_opt = Some(emb),
                            Err(e) => {
                                send_mcp_error(id, &format!("Lỗi load embedder: {}", e));
                                line.clear();
                                continue;
                            }
                        }
                    }
                    let embedder = embedder_opt.as_ref().unwrap();
                    let mut db = Database::load(db_path.clone());

                    match tool_name {
                        "add_fact" => {
                            let fact = args["fact"].as_str().unwrap_or("");
                            let user_id = args["user_id"].as_str().unwrap_or("acer");

                            match generate_embedding(embedder, fact) {
                                Ok(vector) => {
                                    let fact_id = Uuid::new_v4().to_string();
                                    db.records.push(MemoryRecord {
                                        id: fact_id.clone(),
                                        text: fact.to_string(),
                                        vector,
                                        user_id: user_id.to_string(),
                                    });
                                    if let Err(e) = db.save() {
                                        send_mcp_error(id, &format!("Lỗi ghi DB: {}", e));
                                    } else {
                                        let result = json!({
                                            "status": "success",
                                            "memory_id": fact_id,
                                            "fact": fact
                                        });
                                        send_mcp_success(id, &result.to_string());
                                    }
                                }
                                Err(e) => send_mcp_error(id, &format!("Lỗi tạo embedding: {}", e)),
                            }
                        }
                        "search_facts" => {
                            let query = args["query"].as_str().unwrap_or("");
                            let user_id = args["user_id"].as_str().unwrap_or("acer");
                            let limit = args["limit"].as_u64().unwrap_or(5) as usize;

                            match generate_embedding(embedder, query) {
                                Ok(query_vector) => {
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

                                    let result = json!({
                                        "status": "success",
                                        "query": query,
                                        "results": results
                                    });
                                    send_mcp_success(id, &result.to_string());
                                }
                                Err(e) => send_mcp_error(id, &format!("Lỗi tạo embedding: {}", e)),
                            }
                        }
                        "get_all_facts" => {
                            let user_id = args["user_id"].as_str().unwrap_or("acer");
                            let results: Vec<serde_json::Value> = db.records.iter()
                                .filter(|r| r.user_id == user_id)
                                .map(|r| json!({
                                    "id": r.id,
                                    "text": r.text
                                }))
                                .collect();

                            let result = json!({
                                "status": "success",
                                "results": results
                            });
                            send_mcp_success(id, &result.to_string());
                        }
                        "delete_fact" => {
                            let fact_id = args["fact_id"].as_str().unwrap_or("");
                            let original_len = db.records.len();
                            db.records.retain(|r| r.id != fact_id);
                            if db.records.len() < original_len {
                                if let Err(e) = db.save() {
                                    send_mcp_error(id, &format!("Lỗi ghi DB: {}", e));
                                } else {
                                    let result = json!({"status": "success", "message": format!("Deleted memory {}", fact_id)});
                                    send_mcp_success(id, &result.to_string());
                                }
                            } else {
                                let result = json!({"status": "error", "message": format!("Memory ID {} not found", fact_id)});
                                send_mcp_success(id, &result.to_string());
                            }
                        }
                        "delete_all_facts" => {
                            let user_id = args["user_id"].as_str().unwrap_or("acer");
                            db.records.retain(|r| r.user_id != user_id);
                            if let Err(e) = db.save() {
                                send_mcp_error(id, &format!("Lỗi ghi DB: {}", e));
                            } else {
                                let result = json!({"status": "success", "message": format!("Cleared all memories for user {}", user_id)});
                                send_mcp_success(id, &result.to_string());
                            }
                        }
                        _ => send_mcp_error(id, &format!("Tool '{}' không được hỗ trợ", tool_name)),
                    }
                }
                _ => {
                    // Phản hồi rỗng cho các method khác (như notifications)
                }
            }
        }

        line.clear();
    }

    Ok(())
}

// Helper gửi phản hồi JSON-RPC thành công
fn send_mcp_success(id: &serde_json::Value, text_result: &str) {
    let response = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": text_result
                }
            ]
        }
    });
    println!("{}", response.to_string());
}

// Helper gửi phản hồi JSON-RPC lỗi
fn send_mcp_error(id: &serde_json::Value, error_msg: &str) {
    let response = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": -32000,
            "message": error_msg
        }
    });
    println!("{}", response.to_string());
}

// --- WEB DASHBOARD SERVER (tiny-http Backend) ---

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

fn run_dashboard(port: u16, db_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let server = tiny_http::Server::http(format!("127.0.0.1:{}", port))
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    
    let url = format!("http://127.0.0.1:{}", port);
    println!("============================================================");
    println!("🚀 Mem0 Local Dashboard Server (Rust) đang khởi chạy!");
    println!("🔗 Hãy mở trình duyệt truy cập: {}", url);
    println!("============================================================");

    // Tự động mở trình duyệt trên Windows
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
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mem0 Cục Bộ - Bảng Điều Khiển Ký Ức</title>
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
                <p class="text-slate-400 mt-2">Bảng điều khiển quản lý bộ nhớ dài hạn offline của Jarvis (Rust Backend)</p>
            </div>
            <div class="flex gap-4">
                <div class="glass px-6 py-3 rounded-2xl flex items-center gap-3">
                    <span class="text-sm text-slate-400">Tổng Fact:</span>
                    <span id="fact-count" class="text-2xl font-bold text-indigo-400">0</span>
                </div>
                <button onclick="clearAllMemories()" class="px-5 py-3 rounded-2xl border border-red-500/30 text-red-400 hover:bg-red-500/10 transition flex items-center gap-2">
                    <i class="fa-solid fa-trash-can"></i> Xóa sạch
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
                        <i class="fa-solid fa-plus-circle"></i> Thêm ký ức mới
                    </h2>
                    <div class="space-y-4">
                        <textarea id="new-fact" rows="4" placeholder="Ví dụ: Người dùng thích phát triển API bằng Rust và dùng PostgreSQL..." 
                            class="w-full px-4 py-3 rounded-2xl bg-slate-900/60 border border-slate-700/50 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500/80 transition resize-none"></textarea>
                        <button onclick="saveMemory()" class="w-full py-3.5 rounded-2xl bg-gradient-to-r from-indigo-500 to-purple-600 font-semibold text-white glow-hover transition duration-300 flex justify-center items-center gap-2">
                            <i class="fa-solid fa-cloud-arrow-up"></i> Lưu vào bộ nhớ
                        </button>
                    </div>
                </div>

                <!-- Search Card -->
                <div class="glass p-6 rounded-3xl shadow-xl">
                    <h2 class="text-xl font-semibold mb-4 text-pink-300 flex items-center gap-2">
                        <i class="fa-solid fa-magnifying-glass"></i> Tìm kiếm ngữ nghĩa
                    </h2>
                    <div class="space-y-4">
                        <div class="relative">
                            <input id="search-query" type="text" placeholder="Nhập từ khóa cần tìm..." 
                                class="w-full pl-4 pr-10 py-3 rounded-2xl bg-slate-900/60 border border-slate-700/50 text-white placeholder-slate-500 focus:outline-none focus:border-pink-500/80 transition">
                            <button onclick="searchMemories()" class="absolute right-3 top-3.5 text-slate-400 hover:text-pink-400">
                                <i class="fa-solid fa-arrow-right"></i>
                            </button>
                        </div>
                        <button onclick="resetSearch()" id="btn-reset-search" class="w-full py-2.5 rounded-2xl border border-slate-700/80 text-sm text-slate-400 hover:bg-slate-800/40 transition hidden">
                            Quay lại danh sách chính
                        </button>
                    </div>
                </div>
            </div>

            <!-- Right Side: Memory List -->
            <div class="lg:col-span-2 space-y-4">
                <div class="flex justify-between items-center mb-2">
                    <h2 id="list-title" class="text-2xl font-bold text-slate-200">Kho ký ức hiện tại</h2>
                    <span id="search-indicator" class="text-sm bg-pink-500/10 border border-pink-500/30 text-pink-400 px-3 py-1 rounded-full hidden">Kết quả tìm kiếm</span>
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

        // Load ban đầu
        document.addEventListener('DOMContentLoaded', () => {
            fetchMemories();
            
            // Lắng nghe phím Enter cho tìm kiếm
            document.getElementById('search-query').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') searchMemories();
            });
        });

        // Lấy danh sách memories
        async function fetchMemories() {
            try {
                const response = await fetch(`/api/facts?user_id=${USER_ID}`);
                const data = await response.json();
                if (data.status === 'success') {
                    renderMemories(data.results);
                }
            } catch (err) {
                console.error("Lỗi khi tải dữ liệu:", err);
            }
        }

        // Render danh sách ra giao diện
        function renderMemories(facts, isSearchResult = false) {
            const container = document.getElementById('memories-container');
            const countSpan = document.getElementById('fact-count');
            
            if (!isSearchResult) {
                countSpan.textContent = facts.length;
                document.getElementById('list-title').textContent = "Kho ký ức hiện tại";
                document.getElementById('search-indicator').classList.add('hidden');
                document.getElementById('btn-reset-search').classList.add('hidden');
            } else {
                document.getElementById('list-title').textContent = "Kết quả truy vấn tương đồng";
                document.getElementById('search-indicator').classList.remove('hidden');
                document.getElementById('btn-reset-search').classList.remove('hidden');
            }

            container.innerHTML = '';
            
            if (facts.length === 0) {
                container.innerHTML = `
                    <div class="glass p-12 rounded-3xl text-center text-slate-500">
                        <i class="fa-solid fa-box-open text-5xl mb-4 text-slate-600"></i>
                        <p class="text-lg">Không tìm thấy ký ức nào.</p>
                    </div>
                `;
                return;
            }

            facts.forEach(item => {
                const scoreText = item.score ? `<span class="text-xs bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 px-2 py-1 rounded-lg">Độ trùng khớp: ${(item.score * 100).toFixed(1)}%</span>` : '';
                container.innerHTML += `
                    <div class="glass p-6 rounded-2xl shadow-md flex justify-between items-start gap-4 transition hover:bg-slate-800/30 hover:border-slate-700/60 duration-200">
                        <div class="space-y-2 flex-1">
                            <p class="text-slate-100 leading-relaxed text-lg">${item.text}</p>
                            <div class="flex items-center gap-3">
                                <span class="text-xs text-slate-500 font-mono">ID: ${item.id}</span>
                                ${scoreText}
                            </div>
                        </div>
                        <button onclick="deleteMemory('${item.id}')" class="text-slate-500 hover:text-red-400 p-2 rounded-xl hover:bg-red-500/10 transition" title="Xóa Fact">
                            <i class="fa-solid fa-trash"></i>
                        </button>
                    </div>
                `;
            });
        }

        // Lưu Fact mới
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
                console.error("Lỗi khi lưu ký ức:", err);
            }
        }

        // Tìm kiếm ngữ nghĩa
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
                console.error("Lỗi tìm kiếm:", err);
            }
        }

        // Reset tìm kiếm về danh sách chính
        function resetSearch() {
            document.getElementById('search-query').value = '';
            fetchMemories();
        }

        // Xóa một Fact
        async function deleteMemory(id) {
            if (!confirm("Bạn có chắc chắn muốn xóa Fact này?")) return;
            try {
                const response = await fetch(`/api/facts?id=${id}`, { method: 'DELETE' });
                const res = await response.json();
                if (res.status === 'success') {
                    fetchMemories();
                }
            } catch (err) {
                console.error("Lỗi khi xóa:", err);
            }
        }

        // Xóa sạch toàn bộ ký ức
        async function clearAllMemories() {
            if (!confirm("CẢNH BÁO: Hành động này sẽ xóa SẠCH TOÀN BỘ ký ức đã lưu! Bạn có muốn tiếp tục?")) return;
            try {
                const response = await fetch(`/api/facts/clear?user_id=${USER_ID}`, { method: 'POST' });
                const res = await response.json();
                if (res.status === 'success') {
                    fetchMemories();
                }
            } catch (err) {
                console.error("Lỗi khi xóa sạch:", err);
            }
        }
    </script>
</body>
</html>"#;

fn print_welcome_menu() {
    println!(r#"
======================================================================
🧠 Antigravity Mem0 Local (Rust Standalone v1.0)
======================================================================
Hệ thống bộ nhớ dài hạn offline, in-process, không cần Docker.

Các tùy chọn lệnh hiện có (Chạy kèm tham số để thực thi):

  dashboard             Khởi chạy giao diện Web Dashboard (Glassmorphism)
                        Ví dụ: .\mem0_rust_server.exe dashboard

  add "<Nội dung>"     Thêm một ký ức mới thủ công
                        Ví dụ: .\mem0_rust_server.exe add "User thích Rust"

  search "<Từ khóa>"    Tìm kiếm ngữ nghĩa các ký ức tương đồng
                        Ví dụ: .\mem0_rust_server.exe search "ngôn ngữ yêu thích"

  list                  Liệt kê toàn bộ ký ức đang lưu trữ
                        Ví dụ: .\mem0_rust_server.exe list

  delete <ID>           Xóa một ký ức cụ thể theo ID
                        Ví dụ: .\mem0_rust_server.exe delete <fact_id>

  clear                 Xóa sạch toàn bộ ký ức của người dùng
                        Ví dụ: .\mem0_rust_server.exe clear

  mcp                   Khởi chạy StdIO MCP Server (Dành cho AI Editor kết nối)
                        Ví dụ: .\mem0_rust_server.exe mcp

----------------------------------------------------------------------
* Sử dụng tham số "--help" hoặc "[LỆNH] --help" để xem hướng dẫn chi tiết.
======================================================================
"#);
}

// Menu tương tác nâng cao trải nghiệm người dùng khi chạy trực tiếp file .exe
async fn run_interactive_menu(db_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin);
    
    loop {
        println!("\n======================================================================");
        println!("🧠 Antigravity Mem0 Local (Rust Standalone v1.0) - INTERACTIVE MENU");
        println!("======================================================================");
        println!("Offline, in-process long-term memory system (No Docker required).");
        println!();
        println!("Please select an option below (Enter 0-8 and press Enter):");
        println!();
        println!("  [1] 🚀 Launch Web Dashboard (Default port 8899)");
        println!("  [2] 📋 View all stored memories (List)");
        println!("  [3] 🔍 Semantic search memories (Vector Search)");
        println!("  [4] ➕ Add a new memory manually (Add Fact)");
        println!("  [5] ❌ Delete a memory by ID");
        println!("  [6] 🧹 Clear all memories for a user");
        println!("  [7] 🔌 Run StdIO MCP Server (For AI Editor integration)");
        println!("  [8] 📖 View detailed CLI usage instructions (CLI Help)");
        println!("  [0] 🚪 Exit");
        println!();
        print!("👉 Enter your choice (0-8): ");
        io::stdout().flush().ok();

        let mut choice = String::new();
        if reader.read_line(&mut choice)? == 0 {
            break; // End of file/input stream
        }
        let choice = choice.trim();

        match choice {
            "1" => {
                println!("\n--- Launching Web Dashboard ---");
                println!("Running Web Dashboard on port 8899...");
                if let Err(e) = run_dashboard(8899, db_path.clone()) {
                    println!("❌ Dashboard error: {}", e);
                }
            }
            "2" => {
                println!("\n--- Memory List ---");
                print!("Enter User ID [default: acer]: ");
                io::stdout().flush().ok();
                let mut user = String::new();
                reader.read_line(&mut user)?;
                let mut user = user.trim().to_string();
                if user.is_empty() {
                    user = "acer".to_string();
                }

                let db = Database::load(db_path.clone());
                let records: Vec<&MemoryRecord> = db.records.iter().filter(|r| r.user_id == user).collect();
                if records.is_empty() {
                    println!("📭 No memories found for user '{}'.", user);
                } else {
                    println!("Found {} memories for user '{}':", records.len(), user);
                    for (i, r) in records.iter().enumerate() {
                        println!("  [{}] ID: {} | Fact: {}", i + 1, r.id, r.text);
                    }
                }
                println!("\nPress Enter to return to main menu...");
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "3" => {
                println!("\n--- Semantic Search ---");
                print!("Enter search query: ");
                io::stdout().flush().ok();
                let mut query = String::new();
                reader.read_line(&mut query)?;
                let query = query.trim().to_string();
                if query.is_empty() {
                    println!("❌ Search query cannot be empty.");
                    continue;
                }

                print!("Enter User ID [default: acer]: ");
                io::stdout().flush().ok();
                let mut user = String::new();
                reader.read_line(&mut user)?;
                let mut user = user.trim().to_string();
                if user.is_empty() {
                    user = "acer".to_string();
                }

                println!("⚙️ Loading Embedding model and searching...");
                match init_embedder() {
                    Ok(embedder) => {
                        match generate_embedding(&embedder, &query) {
                            Ok(query_vector) => {
                                let db = Database::load(db_path.clone());
                                let mut matches: Vec<(f32, &MemoryRecord)> = db.records.iter()
                                    .filter(|r| r.user_id == user)
                                    .map(|r| {
                                        let score = cosine_similarity(&query_vector, &r.vector);
                                        (score, r)
                                    })
                                    .collect();

                                matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                                if matches.is_empty() {
                                    println!("📭 No matching memories found.");
                                } else {
                                    println!("Top semantic search results:");
                                    for (i, (score, r)) in matches.iter().take(5).enumerate() {
                                        println!("  [{}] Score: {:.1}% | ID: {} | Fact: {}", i + 1, score * 100.0, r.id, r.text);
                                    }
                                }
                            }
                            Err(e) => println!("❌ Embedding generation error: {}", e),
                        }
                    }
                    Err(e) => println!("❌ Model initialization error: {}", e),
                }

                println!("\nPress Enter to return to main menu...");
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "4" => {
                println!("\n--- Add New Memory ---");
                print!("Enter memory content (Fact): ");
                io::stdout().flush().ok();
                let mut fact = String::new();
                reader.read_line(&mut fact)?;
                let fact = fact.trim().to_string();
                if fact.is_empty() {
                    println!("❌ Memory content cannot be empty.");
                    continue;
                }

                print!("Enter User ID [default: acer]: ");
                io::stdout().flush().ok();
                let mut user = String::new();
                reader.read_line(&mut user)?;
                let mut user = user.trim().to_string();
                if user.is_empty() {
                    user = "acer".to_string();
                }

                println!("⚙️ Loading Embedding model and saving memory...");
                match init_embedder() {
                    Ok(embedder) => {
                        match generate_embedding(&embedder, &fact) {
                            Ok(vector) => {
                                let mut db = Database::load(db_path.clone());
                                let fact_id = Uuid::new_v4().to_string();
                                db.records.push(MemoryRecord {
                                    id: fact_id.clone(),
                                    text: fact.clone(),
                                    vector,
                                    user_id: user.clone(),
                                });
                                if let Err(e) = db.save() {
                                    println!("❌ DB write error: {}", e);
                                } else {
                                    println!("✅ Memory saved successfully!");
                                    println!("   - ID: {}", fact_id);
                                    println!("   - User ID: {}", user);
                                    println!("   - Content: {}", fact);
                                }
                            }
                            Err(e) => println!("❌ Embedding generation error: {}", e),
                        }
                    }
                    Err(e) => println!("❌ Model initialization error: {}", e),
                }

                println!("\nPress Enter to return to main menu...");
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "5" => {
                println!("\n--- Delete Memory by ID ---");
                print!("Enter memory ID to delete: ");
                io::stdout().flush().ok();
                let mut id = String::new();
                reader.read_line(&mut id)?;
                let id = id.trim().to_string();
                if id.is_empty() {
                    println!("❌ ID cannot be empty.");
                    continue;
                }

                let mut db = Database::load(db_path.clone());
                let original_len = db.records.len();
                db.records.retain(|r| r.id != id);
                if db.records.len() < original_len {
                    match db.save() {
                        Ok(_) => println!("✅ Memory deleted successfully for ID: {}", id),
                        Err(e) => println!("❌ DB write error: {}", e),
                    }
                } else {
                    println!("❌ No memory found with ID: {}", id);
                }

                println!("\nPress Enter to return to main menu...");
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "6" => {
                println!("\n--- Clear All Memories ---");
                print!("Enter User ID to clear [default: acer]: ");
                io::stdout().flush().ok();
                let mut user = String::new();
                reader.read_line(&mut user)?;
                let mut user = user.trim().to_string();
                if user.is_empty() {
                    user = "acer".to_string();
                }

                print!("⚠️ WARNING: Are you sure you want to clear ALL memories for user '{}'? (y/N): ", user);
                io::stdout().flush().ok();
                let mut confirm = String::new();
                reader.read_line(&mut confirm)?;
                let confirm = confirm.trim().to_lowercase();

                if confirm == "y" || confirm == "yes" {
                    let mut db = Database::load(db_path.clone());
                    db.records.retain(|r| r.user_id != user);
                    match db.save() {
                        Ok(_) => println!("✅ Cleared all memories for user '{}'.", user),
                        Err(e) => println!("❌ DB write error: {}", e),
                    }
                } else {
                    println!("❌ Clear cancelled.");
                }

                println!("\nPress Enter to return to main menu...");
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "7" => {
                println!("\n--- Running MCP Server ---");
                println!("⚠️ Note: MCP Server communicates via Stdin/Stdout using JSON-RPC.");
                println!("Running it here will lock this terminal to handle requests from your AI Editor.");
                println!("Press Ctrl+C to terminate.");
                run_mcp_server(db_path.clone()).await?;
                break;
            }
            "8" => {
                println!("\n--- CLI Usage Guide ---");
                print_welcome_menu();
                println!("\nPress Enter to return to main menu...");
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "0" => {
                println!("🚪 Exiting program. Goodbye!");
                break;
            }
            _ => {
                println!("❌ Invalid option. Please enter a number between 0 and 8.");
                println!("Press Enter to try again...");
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
        }
    }
    Ok(())
}


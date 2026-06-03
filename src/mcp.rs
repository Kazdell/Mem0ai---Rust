use std::io::{self, BufRead};
use std::path::PathBuf;
use serde_json::json;
use uuid::Uuid;
use glowrs::SentenceTransformer;
use crate::db::{Database, MemoryRecord};
use crate::embedding::{init_embedder, generate_embedding, cosine_similarity};

pub async fn run_mcp_server(db_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin);
    let mut line = String::new();
    let mut embedder_opt: Option<SentenceTransformer> = None;

    while reader.read_line(&mut line)? > 0 {
        let req_str = line.trim();
        if req_str.is_empty() {
            line.clear();
            continue;
        }

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
                                "name": "Mem0-Local-Rust",
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
                                    "description": "Add a new fact or memory to the user's long-term memory store.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "fact": {
                                                "type": "string",
                                                "description": "The fact content (e.g. \"User prefers Rust for backend development\")."
                                            },
                                            "user_id": {
                                                "type": "string",
                                                "description": "The identifier of the user (e.g. \"acer\")."
                                            }
                                        },
                                        "required": ["fact", "user_id"]
                                    }
                                },
                                {
                                    "name": "search_facts",
                                    "description": "Semantically search for relevant facts in the user's long-term memory.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {
                                                "type": "string",
                                                "description": "The search query or keyword (e.g. \"favorite programming language\")."
                                            },
                                            "user_id": {
                                                "type": "string",
                                                "description": "The identifier of the user."
                                            },
                                            "limit": {
                                                "type": "integer",
                                                "description": "Maximum number of results to return (default: 5)."
                                            }
                                        },
                                        "required": ["query", "user_id"]
                                    }
                                },
                                {
                                    "name": "get_all_facts",
                                    "description": "Retrieve all stored facts and memories for a user.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "user_id": {
                                                "type": "string",
                                                "description": "The identifier of the user."
                                            }
                                        },
                                        "required": ["user_id"]
                                    }
                                },
                                {
                                    "name": "delete_fact",
                                    "description": "Delete a specific fact from the long-term memory by its ID.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "fact_id": {
                                                "type": "string",
                                                "description": "The ID of the fact memory to delete."
                                            }
                                        },
                                        "required": ["fact_id"]
                                    }
                                },
                                {
                                    "name": "delete_all_facts",
                                    "description": "Clear all memories and facts for a specific user.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "user_id": {
                                                "type": "string",
                                                "description": "The identifier of the user to clear memories for."
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

                    if embedder_opt.is_none() {
                        match init_embedder() {
                            Ok(emb) => embedder_opt = Some(emb),
                            Err(e) => {
                                send_mcp_error(id, &format!("Failed to load embedder: {}", e));
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
                            let user_id = args["user_id"].as_str().unwrap_or("default");

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
                                        send_mcp_error(id, &format!("Database save error: {}", e));
                                    } else {
                                        let result = json!({
                                            "status": "success",
                                            "memory_id": fact_id,
                                            "fact": fact
                                        });
                                        send_mcp_success(id, &result.to_string());
                                    }
                                }
                                Err(e) => send_mcp_error(id, &format!("Failed to generate embedding: {}", e)),
                            }
                        }
                        "search_facts" => {
                            let query = args["query"].as_str().unwrap_or("");
                            let user_id = args["user_id"].as_str().unwrap_or("default");
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
                                Err(e) => send_mcp_error(id, &format!("Failed to generate embedding: {}", e)),
                            }
                        }
                        "get_all_facts" => {
                            let user_id = args["user_id"].as_str().unwrap_or("default");
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
                                    send_mcp_error(id, &format!("Database save error: {}", e));
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
                            let user_id = args["user_id"].as_str().unwrap_or("default");
                            db.records.retain(|r| r.user_id != user_id);
                            if let Err(e) = db.save() {
                                send_mcp_error(id, &format!("Database save error: {}", e));
                            } else {
                                let result = json!({"status": "success", "message": format!("Cleared all memories for user {}", user_id)});
                                send_mcp_success(id, &result.to_string());
                            }
                        }
                        _ => send_mcp_error(id, &format!("Tool '{}' is not supported", tool_name)),
                    }
                }
                _ => {}
            }
        }
        line.clear();
    }
    Ok(())
}

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

use clap::{Parser, Subcommand};
use serde_json::json;
use uuid::Uuid;

mod config;
mod db;
mod embedding;
mod mcp;
mod dashboard;
mod interactive;

use db::{Database, MemoryRecord};
use embedding::{init_embedder, generate_embedding, cosine_similarity};
use dashboard::run_dashboard;
use mcp::run_mcp_server;
use interactive::run_interactive_menu;

// --- CLI DEFINITIONS ---
#[derive(Parser, Debug)]
#[command(name = "mem0_rust_server", version = "1.0.0", about = "Mem0 Local — Offline long-term memory layer for AI agents (Rust Standalone)")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Add a new fact manually
    Add {
        fact: String,
        #[arg(short, long, default_value = "default")]
        user: String,
    },
    /// Search facts semantically
    Search {
        query: String,
        #[arg(short, long, default_value = "default")]
        user: String,
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
    },
    /// List all facts for a user
    List {
        #[arg(short, long, default_value = "default")]
        user: String,
    },
    /// Delete a fact by its ID
    Delete {
        id: String,
    },
    /// Clear all facts for a user
    Clear {
        #[arg(short, long, default_value = "default")]
        user: String,
    },
    /// Launch the Web Dashboard
    Dashboard {
        #[arg(short, long, default_value_t = 8899)]
        port: u16,
    },
    /// Launch the StdIO MCP Server
    Mcp,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Locate database path
    let home = dirs::home_dir().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Home directory not found"))?;
    let db_path = home.join(".mem0_rust").join("db.json");

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
                run_mcp_server(db_path).await?;
            }
        }
    } else {
        run_interactive_menu(db_path).await?;
    }

    Ok(())
}

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use uuid::Uuid;
use crate::config::{AppConfig, Language};
use crate::db::{Database, MemoryRecord};
use crate::embedding::{init_embedder, generate_embedding, cosine_similarity};
use crate::dashboard::run_dashboard;
use crate::mcp::run_mcp_server;

pub fn print_welcome_menu() {
    println!(r#"
======================================================================
Mem0 Local (Rust Standalone v1.0)
======================================================================
Hoi thoai/Ký ức dài hạn offline, in-process, không cần Docker.

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

pub async fn run_interactive_menu(db_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin);
    
    let config_path = db_path.parent()
        .unwrap_or(&PathBuf::from("."))
        .join("config.json");
    
    let mut config = AppConfig::load(config_path.clone());

    loop {
        let is_vi = config.language == Language::Vietnamese;

        println!("\n======================================================================");
        if is_vi {
            println!("Mem0 Local (Rust Standalone v1.0) - MENU TUONG TAC");
        } else {
            println!("Mem0 Local (Rust Standalone v1.0) - INTERACTIVE MENU");
        }
        println!("======================================================================");
        if is_vi {
            println!("He thong bo nho dai han offline, in-process, khong can Docker.");
        } else {
            println!("Offline, in-process long-term memory system (No Docker required).");
        }
        println!();
        if is_vi {
            println!("Hay chon mot tuy chon ben duoi (Nhap so tu 0 den 9 roi nhan Enter):");
        } else {
            println!("Please select an option below (Enter 0-9 and press Enter):");
        }
        println!();
        
        if is_vi {
            println!("  [1] Khoi chay Web Dashboard (Cong mac dinh 8899)");
            println!("  [2] Xem danh sach tat ca ky uc dang luu tru");
            println!("  [3] Tim kiem ngu nghia ky uc (Vector Search)");
            println!("  [4] Them ky uc moi thu cong (Add Fact)");
            println!("  [5] Xoa mot ky uc theo ID");
            println!("  [6] Xoa sach tat ca ky uc cua mot user");
            println!("  [7] Chay StdIO MCP Server (Danh cho AI Editor ket noi)");
            println!("  [8] Xem huong dan su dung dong lenh (CLI Help)");
            println!("  [9] Toggle Language (Vietnamese)");
            println!("  [0] Thoat");
        } else {
            println!("  [1] Launch Web Dashboard (Default port 8899)");
            println!("  [2] View all stored memories (List)");
            println!("  [3] Semantic search memories (Vector Search)");
            println!("  [4] Add a new memory manually (Add Fact)");
            println!("  [5] Delete a memory by ID");
            println!("  [6] Clear all memories for a user");
            println!("  [7] Run StdIO MCP Server (For AI Editor integration)");
            println!("  [8] View detailed CLI usage instructions (CLI Help)");
            println!("  [9] Toggle Language (English)");
            println!("  [0] Exit");
        }
        println!();
        if is_vi {
            print!("Lua chon cua ban (0-9): ");
        } else {
            print!("Enter your choice (0-9): ");
        }
        io::stdout().flush().ok();

        let mut choice = String::new();
        if reader.read_line(&mut choice)? == 0 {
            break;
        }
        let choice = choice.trim();

        match choice {
            "1" => {
                if is_vi {
                    println!("\n--- Khoi chay Web Dashboard ---");
                    println!("Dang chay Web Dashboard tren cong 8899...");
                } else {
                    println!("\n--- Launching Web Dashboard ---");
                    println!("Running Web Dashboard on port 8899...");
                }
                if let Err(e) = run_dashboard(8899, db_path.clone()) {
                    if is_vi {
                        println!("Loi khi chay Dashboard: {}", e);
                    } else {
                        println!("Dashboard error: {}", e);
                    }
                }
            }
            "2" => {
                if is_vi {
                    println!("\n--- Danh sach ky uc ---");
                    print!("Nhap User ID [mac dinh: default]: ");
                } else {
                    println!("\n--- Memory List ---");
                    print!("Enter User ID [default: default]: ");
                }
                io::stdout().flush().ok();
                let mut user = String::new();
                reader.read_line(&mut user)?;
                let mut user = user.trim().to_string();
                if user.is_empty() {
                    user = "default".to_string();
                }

                let db = Database::load(db_path.clone());
                let records: Vec<&MemoryRecord> = db.records.iter().filter(|r| r.user_id == user).collect();
                if records.is_empty() {
                    if is_vi {
                        println!("Khong co ky uc nao cho user '{}'.", user);
                    } else {
                        println!("No memories found for user '{}'.", user);
                    }
                } else {
                    if is_vi {
                        println!("Tim thay {} ky uc cho user '{}':", records.len(), user);
                        for (i, r) in records.iter().enumerate() {
                            println!("  [{}] ID: {} | Fact: {}", i + 1, r.id, r.text);
                        }
                    } else {
                        println!("Found {} memories for user '{}':", records.len(), user);
                        for (i, r) in records.iter().enumerate() {
                            println!("  [{}] ID: {} | Fact: {}", i + 1, r.id, r.text);
                        }
                    }
                }
                if is_vi {
                    println!("\nNhan Enter de quay lai Menu chinh...");
                } else {
                    println!("\nPress Enter to return to main menu...");
                }
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "3" => {
                if is_vi {
                    println!("\n--- Tim kiem ngu nghia ky uc ---");
                    print!("Nhap cau truy van tim kiem: ");
                } else {
                    println!("\n--- Semantic Search ---");
                    print!("Enter search query: ");
                }
                io::stdout().flush().ok();
                let mut query = String::new();
                reader.read_line(&mut query)?;
                let query = query.trim().to_string();
                if query.is_empty() {
                    if is_vi {
                        println!("Truy van khong duoc de trong.");
                    } else {
                        println!("Search query cannot be empty.");
                    }
                    continue;
                }

                if is_vi {
                    print!("Nhap User ID [mac dinh: default]: ");
                } else {
                    print!("Enter User ID [default: default]: ");
                }
                io::stdout().flush().ok();
                let mut user = String::new();
                reader.read_line(&mut user)?;
                let mut user = user.trim().to_string();
                if user.is_empty() {
                    user = "default".to_string();
                }

                if is_vi {
                    println!("Dang tai mo hinh Embedding va tien hanh tim kiem...");
                } else {
                    println!("Loading Embedding model and searching...");
                }
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
                                    if is_vi {
                                        println!("Khong tim thay ket qua nao tuong hop.");
                                    } else {
                                        println!("No matching memories found.");
                                    }
                                } else {
                                    if is_vi {
                                        println!("Ket qua tim kiem ngu nghia hang dau:");
                                        for (i, (score, r)) in matches.iter().take(5).enumerate() {
                                            println!("  [{}] Do trung khop: {:.1}% | ID: {} | Fact: {}", i + 1, score * 100.0, r.id, r.text);
                                        }
                                    } else {
                                        println!("Top semantic search results:");
                                        for (i, (score, r)) in matches.iter().take(5).enumerate() {
                                            println!("  [{}] Score: {:.1}% | ID: {} | Fact: {}", i + 1, score * 100.0, r.id, r.text);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                if is_vi {
                                    println!("Loi tao vector nhung: {}", e);
                                } else {
                                    println!("Embedding generation error: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if is_vi {
                            println!("Loi khoi tao mo hinh embedding: {}", e);
                        } else {
                            println!("Model initialization error: {}", e);
                        }
                    }
                }

                if is_vi {
                    println!("\nNhan Enter de quay lai Menu chinh...");
                } else {
                    println!("\nPress Enter to return to main menu...");
                }
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "4" => {
                if is_vi {
                    println!("\n--- Them ky uc moi ---");
                    print!("Nhap noi dung ky uc (Fact): ");
                } else {
                    println!("\n--- Add New Memory ---");
                    print!("Enter memory content (Fact): ");
                }
                io::stdout().flush().ok();
                let mut fact = String::new();
                reader.read_line(&mut fact)?;
                let fact = fact.trim().to_string();
                if fact.is_empty() {
                    if is_vi {
                        println!("Noi dung ky uc khong duoc de trong.");
                    } else {
                        println!("Memory content cannot be empty.");
                    }
                    continue;
                }

                if is_vi {
                    print!("Nhap User ID [mac dinh: default]: ");
                } else {
                    print!("Enter User ID [default: default]: ");
                }
                io::stdout().flush().ok();
                let mut user = String::new();
                reader.read_line(&mut user)?;
                let mut user = user.trim().to_string();
                if user.is_empty() {
                    user = "default".to_string();
                }

                if is_vi {
                    println!("Dang tai mo hinh Embedding va luu ky uc...");
                } else {
                    println!("Loading Embedding model and saving memory...");
                }
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
                                    if is_vi {
                                        println!("Loi luu du lieu DB: {}", e);
                                    } else {
                                        println!("DB write error: {}", e);
                                    }
                                } else {
                                    if is_vi {
                                        println!("Da luu ky uc thanh cong!");
                                        println!("   - ID: {}", fact_id);
                                        println!("   - User ID: {}", user);
                                        println!("   - Noi dung: {}", fact);
                                    } else {
                                        println!("Memory saved successfully!");
                                        println!("   - ID: {}", fact_id);
                                        println!("   - User ID: {}", user);
                                        println!("   - Content: {}", fact);
                                    }
                                }
                            }
                            Err(e) => {
                                if is_vi {
                                    println!("Loi tao vector nhung: {}", e);
                                } else {
                                    println!("Embedding generation error: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if is_vi {
                            println!("Loi khoi tao mo hinh embedding: {}", e);
                        } else {
                            println!("Model initialization error: {}", e);
                        }
                    }
                }

                if is_vi {
                    println!("\nNhan Enter de quay lai Menu chinh...");
                } else {
                    println!("\nPress Enter to return to main menu...");
                }
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "5" => {
                if is_vi {
                    println!("\n--- Xoa ky uc theo ID ---");
                    print!("Nhap ID cua ky uc can xoa: ");
                } else {
                    println!("\n--- Delete Memory by ID ---");
                    print!("Enter memory ID to delete: ");
                }
                io::stdout().flush().ok();
                let mut id = String::new();
                reader.read_line(&mut id)?;
                let id = id.trim().to_string();
                if id.is_empty() {
                    if is_vi {
                        println!("ID khong duoc de trong.");
                    } else {
                        println!("ID cannot be empty.");
                    }
                    continue;
                }

                let mut db = Database::load(db_path.clone());
                let original_len = db.records.len();
                db.records.retain(|r| r.id != id);
                if db.records.len() < original_len {
                    match db.save() {
                        Ok(_) => {
                            if is_vi {
                                println!("Da xoa thanh cong ky uc co ID: {}", id);
                            } else {
                                println!("Memory deleted successfully for ID: {}", id);
                            }
                        }
                        Err(e) => {
                            if is_vi {
                                println!("Loi ghi DB: {}", e);
                            } else {
                                println!("DB write error: {}", e);
                            }
                        }
                    }
                } else {
                    if is_vi {
                        println!("Khong tim thay ky uc nao phu hop voi ID: {}", id);
                    } else {
                        println!("No memory found with ID: {}", id);
                    }
                }

                if is_vi {
                    println!("\nNhan Enter de quay lai Menu chinh...");
                } else {
                    println!("\nPress Enter to return to main menu...");
                }
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "6" => {
                if is_vi {
                    println!("\n--- Xoa sach tat ca ky uc ---");
                    print!("Nhap User ID can xoa sach bo nho [mac dinh: default]: ");
                } else {
                    println!("\n--- Clear All Memories ---");
                    print!("Enter User ID to clear [default: default]: ");
                }
                io::stdout().flush().ok();
                let mut user = String::new();
                reader.read_line(&mut user)?;
                let mut user = user.trim().to_string();
                if user.is_empty() {
                    user = "default".to_string();
                }

                let confirm = if is_vi {
                    print!("CANH BAO: Hanh dong nay se xoa SACH tat ca ky uc cua user '{}'! Co tiep tuc? (y/N): ", user);
                    io::stdout().flush().ok();
                    let mut confirm = String::new();
                    reader.read_line(&mut confirm)?;
                    confirm.trim().to_lowercase()
                } else {
                    print!("WARNING: Are you sure you want to clear ALL memories for user '{}'? (y/N): ", user);
                    io::stdout().flush().ok();
                    let mut confirm = String::new();
                    reader.read_line(&mut confirm)?;
                    confirm.trim().to_lowercase()
                };

                if confirm == "y" || confirm == "yes" {
                    let mut db = Database::load(db_path.clone());
                    db.records.retain(|r| r.user_id != user);
                    match db.save() {
                        Ok(_) => {
                            if is_vi {
                                println!("Da xoa sach toan bo ky uc cua user '{}'.", user);
                            } else {
                                println!("Cleared all memories for user '{}'.", user);
                            }
                        }
                        Err(e) => {
                            if is_vi {
                                println!("Loi khi ghi DB: {}", e);
                            } else {
                                println!("DB write error: {}", e);
                            }
                        }
                    }
                } else {
                    if is_vi {
                        println!("Da huy thao tac xoa sach.");
                    } else {
                        println!("Clear cancelled.");
                    }
                }

                if is_vi {
                    println!("\nNhan Enter de quay lai Menu chinh...");
                } else {
                    println!("\nPress Enter to return to main menu...");
                }
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "7" => {
                if is_vi {
                    println!("\n--- Khoi chay MCP Server ---");
                    println!("Luu y: MCP Server giao tiep qua Stdin/Stdout dang JSON-RPC.");
                    println!("Khi chay tu giao dien terminal nay, tien trinh se bi khoa va tuong tac truc tiep voi AI Editor.");
                    println!("Nhan Ctrl+C de thoat.");
                } else {
                    println!("\n--- Running MCP Server ---");
                    println!("Note: MCP Server communicates via Stdin/Stdout using JSON-RPC.");
                    println!("Running it here will lock this terminal to handle requests from your AI Editor.");
                    println!("Press Ctrl+C to terminate.");
                }
                run_mcp_server(db_path.clone()).await?;
                break;
            }
            "8" => {
                if is_vi {
                    println!("\n--- Huong dan su dung CLI ---");
                } else {
                    println!("\n--- CLI Usage Guide ---");
                }
                print_welcome_menu();
                if is_vi {
                    println!("\nNhan Enter de quay lai Menu chinh...");
                } else {
                    println!("\nPress Enter to return to main menu...");
                }
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
            "9" => {
                if config.language == Language::English {
                    config.language = Language::Vietnamese;
                } else {
                    config.language = Language::English;
                }
                if let Err(e) = config.save(config_path.clone()) {
                    println!("Config save error: {}", e);
                }
            }
            "0" => {
                if is_vi {
                    println!("Dang thoat chuong trinh. Tam biet!");
                } else {
                    println!("Exiting program. Goodbye!");
                }
                break;
            }
            _ => {
                if is_vi {
                    println!("Lua chon khong hop le. Vui loi nhap so tu 0 den 9.");
                    println!("Nhan Enter de thu lai...");
                } else {
                    println!("Invalid option. Please enter a number between 0 and 9.");
                    println!("Press Enter to try again...");
                }
                let mut temp = String::new();
                reader.read_line(&mut temp)?;
            }
        }
    }
    Ok(())
}

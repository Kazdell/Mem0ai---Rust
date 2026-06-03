use std::fs;
use std::io;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryRecord {
    pub id: String,
    pub text: String,
    pub vector: Vec<f32>,
    pub user_id: String,
}

pub struct Database {
    pub path: PathBuf,
    pub records: Vec<MemoryRecord>,
}

impl Database {
    pub fn load(path: PathBuf) -> Self {
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

    pub fn save(&self) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(&self.records)?;
        fs::write(&self.path, data)?;
        Ok(())
    }
}

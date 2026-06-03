use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum Language {
    English,
    Vietnamese,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppConfig {
    pub language: Language,
}

impl AppConfig {
    pub fn default() -> Self {
        AppConfig {
            language: Language::English,
        }
    }

    pub fn load(path: PathBuf) -> Self {
        if !path.exists() {
            return AppConfig::default();
        }
        let data = fs::read_to_string(&path).unwrap_or_default();
        serde_json::from_str(&data).unwrap_or_else(|_| AppConfig::default())
    }

    pub fn save(&self, path: PathBuf) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }
}

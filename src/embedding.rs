use std::io;
use glowrs::{SentenceTransformer, Device, PoolingStrategy};

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

pub fn init_embedder() -> Result<SentenceTransformer, io::Error> {
    let mut exe_dir = std::env::current_exe()?;
    exe_dir.pop();
    let local_model_dir = exe_dir.join("model");

    let embedder = if local_model_dir.join("model.safetensors").exists()
        && local_model_dir.join("config.json").exists()
        && local_model_dir.join("tokenizer.json").exists()
    {
        eprintln!("⚙️ Local model detected at {:?}. Initializing...", local_model_dir);
        SentenceTransformer::from_folder(&local_model_dir, &Device::Cpu)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
    } else {
        eprintln!("⚙️ Local model not found next to exe. Fetching/loading from HuggingFace...");
        SentenceTransformer::from_repo_string(
            "sentence-transformers/all-MiniLM-L6-v2",
            &Device::Cpu,
        ).map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
    };

    eprintln!("✅ Embedding model ready!");
    Ok(embedder)
}

pub fn generate_embedding(embedder: &SentenceTransformer, text: &str) -> Result<Vec<f32>, io::Error> {
    let sentences = vec![text.to_string()];
    let embeddings_tensor = embedder.encode_batch(sentences, true, PoolingStrategy::Mean)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    
    let embeddings: Vec<Vec<f32>> = embeddings_tensor.to_vec2()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    
    if let Some(vector) = embeddings.first() {
        Ok(vector.clone())
    } else {
        Err(io::Error::new(io::ErrorKind::Other, "Could not generate vector embedding"))
    }
}

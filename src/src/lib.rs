//! vROM.js: A WebAssembly vector search database engine.
//!
//! Features:
//! - HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search
//! - Multiple distance metrics: Cosine, Euclidean, Dot Product
//! - Serialization/deserialization for persistence (IndexedDB, localStorage, etc.)
//! - Metadata support per vector
//! - Configurable index parameters (M, efConstruction, efSearch)
//!
//! Architecture:
//!   Rust core (distance.rs, hnsw.rs) → wasm-bindgen → JavaScript API

pub mod distance;
pub mod hnsw;

use distance::DistanceMetric;
use hnsw::{HnswConfig, HnswIndex};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct VectorDB {
    index: HnswIndex,
}

#[wasm_bindgen]
impl VectorDB {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, metric: Option<String>, m: Option<usize>,
               ef_construction: Option<usize>, ef_search: Option<usize>) -> Result<VectorDB, JsError> {
        if dim == 0 { return Err(JsError::new("Dimension must be > 0")); }
        let metric = match metric.as_deref().unwrap_or("cosine") {
            "cosine" => DistanceMetric::Cosine,
            "euclidean" | "l2" => DistanceMetric::Euclidean,
            "dot_product" | "ip" | "inner_product" => DistanceMetric::DotProduct,
            other => return Err(JsError::new(&format!("Unknown metric: '{}'", other))),
        };
        let m_val = m.unwrap_or(16);
        let config = HnswConfig {
            m: m_val, m_max0: 2 * m_val,
            ef_construction: ef_construction.unwrap_or(128),
            ef_search: ef_search.unwrap_or(40),
            level_multiplier: 1.0 / (m_val as f64).ln(), metric,
        };
        Ok(VectorDB { index: HnswIndex::new(dim, config) })
    }

    pub fn insert(&mut self, vector: &[f32], metadata: Option<String>) -> Result<usize, JsError> {
        if vector.len() != self.index.dim() {
            return Err(JsError::new(&format!("Expected {}-dim vector, got {}", self.index.dim(), vector.len())));
        }
        Ok(self.index.insert(vector.to_vec(), metadata))
    }

    pub fn insert_batch(&mut self, vectors: &[f32], n: usize) -> Result<usize, JsError> {
        let dim = self.index.dim();
        if vectors.len() != n * dim {
            return Err(JsError::new(&format!("Expected {} floats, got {}", n * dim, vectors.len())));
        }
        let start_id = self.index.len();
        for i in 0..n {
            self.index.insert(vectors[i * dim..(i + 1) * dim].to_vec(), None);
        }
        Ok(start_id)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<String, JsError> {
        if query.len() != self.index.dim() {
            return Err(JsError::new(&format!("Expected {}-dim query, got {}", self.index.dim(), query.len())));
        }
        let results = self.index.search(query, k);
        let json: Vec<serde_json::Value> = results.into_iter().map(|(id, dist)| {
            let mut obj = serde_json::json!({"id": id, "distance": dist});
            if let Some(meta) = self.index.get_metadata(id) {
                obj["metadata"] = serde_json::Value::String(meta.to_string());
            }
            obj
        }).collect();
        serde_json::to_string(&json).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn search_with_ef(&self, query: &[f32], k: usize, ef_search: usize) -> Result<String, JsError> {
        if query.len() != self.index.dim() {
            return Err(JsError::new(&format!("Expected {}-dim query, got {}", self.index.dim(), query.len())));
        }
        let results = self.index.search_with_ef(query, k, ef_search);
        let json: Vec<serde_json::Value> = results.into_iter().map(|(id, dist)| {
            let mut obj = serde_json::json!({"id": id, "distance": dist});
            if let Some(meta) = self.index.get_metadata(id) {
                obj["metadata"] = serde_json::Value::String(meta.to_string());
            }
            obj
        }).collect();
        serde_json::to_string(&json).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn get_vector(&self, id: usize) -> Option<Vec<f32>> { self.index.get_vector(id).map(|v| v.to_vec()) }
    pub fn get_metadata(&self, id: usize) -> Option<String> { self.index.get_metadata(id).map(|s| s.to_string()) }
    pub fn len(&self) -> usize { self.index.len() }
    pub fn dim(&self) -> usize { self.index.dim() }

    pub fn stats(&self) -> String {
        serde_json::to_string_pretty(&self.index.stats()).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn save(&self) -> Result<String, JsError> {
        self.index.serialize().map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen(js_name = "load")]
    pub fn load_from_json(json: &str) -> Result<VectorDB, JsError> {
        let index = HnswIndex::deserialize(json, None).map_err(|e| JsError::new(&e))?;
        Ok(VectorDB { index })
    }
}

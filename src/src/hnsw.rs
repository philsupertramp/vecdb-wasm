/// HNSW (Hierarchical Navigable Small World) index implementation.
///
/// Based on: Malkov & Yashunin, "Efficient and Robust Approximate Nearest
/// Neighbor Search Using Hierarchical Navigable Small World Graphs" (2018).
/// arxiv:1603.09320
///
/// Parameters follow industry consensus from arxiv:2405.17813:
///   M = 16, efConstruction = 128, efSearch = 40

use crate::distance::DistanceMetric;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;

#[derive(Clone, Debug)]
struct Candidate {
    id: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool { self.distance == other.distance }
}
impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

#[derive(Clone, Debug)]
struct MinCandidate {
    id: usize,
    distance: f32,
}

impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool { self.distance == other.distance }
}
impl Eq for MinCandidate {}
impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswNode {
    vector: Vec<f32>,
    neighbors: Vec<Vec<usize>>,
    max_layer: usize,
    metadata: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub level_multiplier: f64,
    pub metric: DistanceMetric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        HnswConfig {
            m, m_max0: 2 * m, ef_construction: 128, ef_search: 40,
            level_multiplier: 1.0 / (m as f64).ln(),
            metric: DistanceMetric::Cosine,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HnswIndex {
    config: HnswConfig,
    nodes: Vec<HnswNode>,
    entry_point: Option<usize>,
    max_layer: usize,
    dim: usize,
    #[serde(skip)]
    rng: Option<SmallRng>,
}

impl HnswIndex {
    pub fn new(dim: usize, config: HnswConfig) -> Self {
        HnswIndex { config, nodes: Vec::new(), entry_point: None, max_layer: 0, dim,
            rng: Some(SmallRng::from_entropy()) }
    }

    pub fn with_seed(dim: usize, config: HnswConfig, seed: u64) -> Self {
        HnswIndex { config, nodes: Vec::new(), entry_point: None, max_layer: 0, dim,
            rng: Some(SmallRng::seed_from_u64(seed)) }
    }

    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }
    pub fn dim(&self) -> usize { self.dim }

    fn random_level(&mut self) -> usize {
        let rng = self.rng.as_mut().expect("RNG not initialized");
        let r: f64 = rng.gen::<f64>();
        (-r.ln() * self.config.level_multiplier).floor() as usize
    }

    #[inline]
    fn distance(&self, query: &[f32], node_id: usize) -> f32 {
        self.config.metric.compute(query, &self.nodes[node_id].vector)
    }

    pub fn insert(&mut self, vector: Vec<f32>, metadata: Option<String>) -> usize {
        assert_eq!(vector.len(), self.dim);
        let new_id = self.nodes.len();
        let new_level = self.random_level();
        let node = HnswNode { vector, neighbors: vec![Vec::new(); new_level + 1], max_layer: new_level, metadata };
        self.nodes.push(node);

        if self.nodes.len() == 1 {
            self.entry_point = Some(new_id);
            self.max_layer = new_level;
            return new_id;
        }

        let ep = self.entry_point.unwrap();
        let query = self.nodes[new_id].vector.clone();
        let mut current_ep = ep;

        for layer in (new_level + 1..=self.max_layer).rev() {
            current_ep = self.search_layer_single(&query, current_ep, layer);
        }

        let start_layer = new_level.min(self.max_layer);
        let mut ep_set = vec![current_ep];

        for layer in (0..=start_layer).rev() {
            let m_max = if layer == 0 { self.config.m_max0 } else { self.config.m };
            let candidates = self.search_layer(&query, &ep_set, self.config.ef_construction, layer);
            let neighbors: Vec<usize> = candidates.iter().take(m_max).map(|c| c.id).collect();
            self.nodes[new_id].neighbors[layer] = neighbors.clone();

            for &neighbor_id in &neighbors {
                if layer < self.nodes[neighbor_id].neighbors.len() {
                    self.nodes[neighbor_id].neighbors[layer].push(new_id);
                    if self.nodes[neighbor_id].neighbors[layer].len() > m_max {
                        self.shrink_connections(neighbor_id, layer, m_max);
                    }
                }
            }
            ep_set = candidates.iter().take(self.config.m).map(|c| c.id).collect();
        }

        if new_level > self.max_layer {
            self.entry_point = Some(new_id);
            self.max_layer = new_level;
        }
        new_id
    }

    fn search_layer_single(&self, query: &[f32], entry: usize, layer: usize) -> usize {
        let mut current = entry;
        let mut current_dist = self.distance(query, current);
        loop {
            let mut changed = false;
            if layer < self.nodes[current].neighbors.len() {
                for &neighbor in &self.nodes[current].neighbors[layer] {
                    let d = self.distance(query, neighbor);
                    if d < current_dist { current = neighbor; current_dist = d; changed = true; }
                }
            }
            if !changed { break; }
        }
        current
    }

    fn search_layer(&self, query: &[f32], entry_points: &[usize], ef: usize, layer: usize) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for &ep in entry_points {
            if visited.insert(ep) {
                let d = self.distance(query, ep);
                candidates.push(MinCandidate { id: ep, distance: d });
                results.push(Candidate { id: ep, distance: d });
            }
        }

        while let Some(MinCandidate { id: current, distance: current_dist }) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current_dist > worst.distance { break; }
                }
            }
            if layer < self.nodes[current].neighbors.len() {
                for &neighbor in &self.nodes[current].neighbors[layer] {
                    if visited.insert(neighbor) {
                        let d = self.distance(query, neighbor);
                        let should_add = results.len() < ef
                            || d < results.peek().map(|w| w.distance).unwrap_or(f32::MAX);
                        if should_add {
                            candidates.push(MinCandidate { id: neighbor, distance: d });
                            results.push(Candidate { id: neighbor, distance: d });
                            if results.len() > ef { results.pop(); }
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<Candidate> = results.into_vec();
        result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        result_vec
    }

    fn shrink_connections(&mut self, node_id: usize, layer: usize, max_connections: usize) {
        let node_vec = self.nodes[node_id].vector.clone();
        let mut neighbors_with_dist: Vec<(usize, f32)> = self.nodes[node_id].neighbors[layer]
            .iter().map(|&nid| (nid, self.config.metric.compute(&node_vec, &self.nodes[nid].vector))).collect();
        neighbors_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        neighbors_with_dist.truncate(max_connections);
        self.nodes[node_id].neighbors[layer] = neighbors_with_dist.iter().map(|&(id, _)| id).collect();
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    pub fn search_with_ef(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dim);
        if self.nodes.is_empty() { return Vec::new(); }

        let ep = self.entry_point.unwrap();
        let mut current_ep = ep;
        if self.max_layer > 0 {
            for layer in (1..=self.max_layer).rev() {
                current_ep = self.search_layer_single(query, current_ep, layer);
            }
        }

        let ef = ef_search.max(k);
        let candidates = self.search_layer(query, &[current_ep], ef, 0);
        candidates.into_iter().take(k).map(|c| (c.id, c.distance)).collect()
    }

    pub fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dim);
        let mut distances: Vec<(usize, f32)> = self.nodes.iter().enumerate()
            .map(|(id, node)| (id, self.config.metric.compute(query, &node.vector))).collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        distances.truncate(k);
        distances
    }

    pub fn get_vector(&self, id: usize) -> Option<&[f32]> { self.nodes.get(id).map(|n| n.vector.as_slice()) }
    pub fn get_metadata(&self, id: usize) -> Option<&str> { self.nodes.get(id).and_then(|n| n.metadata.as_deref()) }

    pub fn serialize(&self) -> Result<String, String> { serde_json::to_string(self).map_err(|e| e.to_string()) }

    pub fn deserialize(json: &str, seed: Option<u64>) -> Result<Self, String> {
        let mut index: HnswIndex = serde_json::from_str(json).map_err(|e| e.to_string())?;
        index.rng = Some(match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        });
        Ok(index)
    }

    pub fn stats(&self) -> IndexStats {
        let total_connections: usize = self.nodes.iter()
            .flat_map(|n| n.neighbors.iter()).map(|layer| layer.len()).sum();
        let avg_connections = if self.nodes.is_empty() { 0.0 }
            else { total_connections as f64 / self.nodes.len() as f64 };
        let memory_bytes = self.nodes.iter().map(|n| {
            std::mem::size_of::<HnswNode>()
                + n.vector.len() * std::mem::size_of::<f32>()
                + n.neighbors.iter().map(|l| l.len() * std::mem::size_of::<usize>()).sum::<usize>()
                + n.metadata.as_ref().map(|m| m.len()).unwrap_or(0)
        }).sum::<usize>();

        IndexStats { num_vectors: self.nodes.len(), dimensions: self.dim, max_layer: self.max_layer,
            total_connections, avg_connections_per_node: avg_connections, memory_bytes }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub num_vectors: usize,
    pub dimensions: usize,
    pub max_layer: usize,
    pub total_connections: usize,
    pub avg_connections_per_node: f64,
    pub memory_bytes: usize,
}

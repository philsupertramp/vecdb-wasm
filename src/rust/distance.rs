/// Distance metrics for vector similarity search.
///
/// Supports: Cosine, Euclidean (L2), Dot Product (Inner Product).
/// Cosine similarity is implemented as 1 - cos(a, b) so that all metrics
/// return a "distance" where lower = more similar.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl DistanceMetric {
    /// Compute distance between two vectors using this metric.
    /// Lower values = more similar for all metrics.
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
        match self {
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::DotProduct => dot_product_distance(a, b),
        }
    }
}

/// Squared Euclidean distance (avoids sqrt for performance; monotonic).
#[inline]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Cosine distance = 1 - cosine_similarity.
/// cosine_similarity = dot(a,b) / (||a|| * ||b||)
#[inline]
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        return 1.0; // undefined → max distance
    }
    1.0 - (dot / denom)
}

/// Negative dot product (so lower = more similar, consistent with other metrics).
#[inline]
fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    -dot
}

/// Normalize a vector to unit length (in-place). Used for cosine→IP optimization.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

//! # Monotile Border Trait
//! Handles shape-specific logic
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Feb-02
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** -
//! - **Todo:** -

use crate::mesh_definition::TexCoord;
use tri_mesh::VertexID;

pub trait MonotileBorder {
    fn distribute_vertices_around_monotile(&self, boundary_vertices: &[VertexID], side_length: f64, tolerance: f64, total_length: f64) -> Vec<TexCoord>;
    fn corners(&self, origin: TexCoord, side_length: f64) -> Vec<TexCoord>;
}

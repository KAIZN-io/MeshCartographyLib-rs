//! # Mesh Definition
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Dec-11
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** -
//! - **Todo:** -

use std::collections::HashMap;
use tri_mesh::Mesh;

#[derive(Clone)]
pub struct TexCoord(pub f64, pub f64);

pub struct MeshTexCoords {
    coords: HashMap<tri_mesh::VertexID, TexCoord>,
}

impl MeshTexCoords {
    pub fn new(mesh: &Mesh) -> Self {
        let coords = mesh.vertex_iter()
                         .map(|v| (v, TexCoord(0.0, 0.0)))
                         .collect();
        MeshTexCoords { coords }
    }

    pub fn set_tex_coord(&mut self, vertex: tri_mesh::VertexID, coord: TexCoord) {
        self.coords.insert(vertex, coord);
    }

    pub fn get_tex_coord(&self, vertex: tri_mesh::VertexID) -> Option<&TexCoord> {
        self.coords.get(&vertex)
    }
}

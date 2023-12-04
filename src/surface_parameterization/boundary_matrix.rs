//! # Boundary Matrix
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Dec-04
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** -
//! - **Todo:** -

use nalgebra::DMatrix;

extern crate tri_mesh;
use tri_mesh::Mesh;

use crate::mesh_definition;

pub fn set_boundary_constraints(mesh: &Mesh, mesh_tex_coords: &mut mesh_definition::MeshTexCoords) -> DMatrix<f64> {
    // Build the RHS vector B
    const DIM: usize = 2;
    let mut B = DMatrix::zeros(mesh.no_vertices(), DIM);
    for vertex_id in mesh.vertex_iter() {
        if mesh.is_vertex_on_boundary(vertex_id) {
            if let Some(tex_coord) = mesh_tex_coords.get_tex_coord(vertex_id) {
                let index_as_u32: u32 = *vertex_id; // Dereference to get u32
                let index_as_usize: usize = index_as_u32 as usize; // Cast u32 to usize
                B.set_row(index_as_usize, &nalgebra::RowVector2::new(tex_coord.0, tex_coord.1));
            }
        }
    }

    B
}

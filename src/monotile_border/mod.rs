//! # Monotile Border Module
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

pub mod square_border_helper;

use crate::mesh_definition;
use crate::mesh_definition::TexCoord;
use std::collections::HashMap;
use tri_mesh::VertexID;
use crate::monotile_border::square_border_helper::square_corners;

pub fn get_sub_borders(boundary_vertices: &[VertexID], mesh_tex_coords: &mesh_definition::MeshTexCoords) -> (HashMap<usize, Vec<VertexID>>, HashMap<usize, Vec<TexCoord>>) {
    // Get the corner coordinates of the square
    let origin = TexCoord(0.0, 0.0);
    let side_length = 1.0;
    let corners = square_corners(origin, side_length);

    let mut border_v_map: HashMap<usize, Vec<VertexID>> = HashMap::new();
    let mut border_map: HashMap<usize, Vec<TexCoord>> = HashMap::new();

    let mut current_border = 0;
    let first_v = boundary_vertices[0];
    let first_point = mesh_tex_coords.get_tex_coord(first_v).unwrap().clone();
    border_v_map.entry(current_border).or_insert(Vec::new()).push(first_v.clone());
    border_map.entry(current_border).or_insert(Vec::new()).push(first_point.clone());

    for v in boundary_vertices.iter().skip(1) {
        let point = mesh_tex_coords.get_tex_coord(*v).unwrap().clone();

        border_v_map.entry(current_border).or_insert(Vec::new()).push(*v);
        border_map.entry(current_border).or_insert(Vec::new()).push(point.clone());

        // Check if we crossed a corner and need to start a new border
        for corner in corners.iter() {
            if (point.0 - corner.0).abs() < 1e-4 && (point.1 - corner.1).abs() < 1e-4 && *v != first_v {
                current_border += 1;
                border_v_map.entry(current_border).or_insert(Vec::new()).push(*v);
                border_map.entry(current_border).or_insert(Vec::new()).push(point.clone());
                break;
            }
        }
    }

    border_v_map.entry(current_border).or_insert(Vec::new()).push(first_v);
    border_map.entry(current_border).or_insert(Vec::new()).push(first_point.clone());

    (border_v_map, border_map)
}

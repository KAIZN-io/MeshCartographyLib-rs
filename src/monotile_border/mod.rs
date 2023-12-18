//! # Create a square border
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

use crate::mesh_definition;
use crate::mesh_definition::TexCoord;
use std::collections::HashMap;
use tri_mesh::{Mesh, VertexID};

pub fn distribute_vertices_around_square(boundary_vertices: &[tri_mesh::VertexID], side_length: f64, tolerance: f64, total_length: f64) -> Vec<TexCoord> {
    let n = boundary_vertices.len();
    let step_size = total_length / n as f64;
    let mut vertices = Vec::new();

    for i in 0..n {
        let mut l = i as f64 * step_size;
        let mut tex_coord;

        // Determine the side and calculate the position
        if l < side_length { // First side (bottom)
            tex_coord = TexCoord(l / side_length, 0.0);
        } else if l < 2.0 * side_length { // Second side (right)
            l -= side_length;
            tex_coord = TexCoord(1.0, l / side_length);
        } else if l < 3.0 * side_length { // Third side (top)
            l -= 2.0 * side_length;
            tex_coord = TexCoord((side_length - l) / side_length, 1.0);
        } else { // Fourth side (left)
            l -= 3.0 * side_length;
            tex_coord = TexCoord(0.0, (side_length - l) / side_length);
        }

        // Adjust precision
        tex_coord.0 = format!("{:.6}", tex_coord.0).parse().unwrap();
        tex_coord.1 = format!("{:.6}", tex_coord.1).parse().unwrap();

        // Apply tolerance
        if tex_coord.0 < tolerance {
            tex_coord.0 = 0.0;
        }
        if tex_coord.1 < tolerance {
            tex_coord.1 = 0.0;
        }

        vertices.push(tex_coord);
    }

    vertices
}

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


/// Returns the four corner coordinates of a square.
///
/// ## Arguments
///
/// - `origin`: The bottom-left corner of the square (x, y).
/// - `side_length`: The length of each side of the square.
///
/// ## Returns
///
/// Returns a `Vec<TexCoord>` containing the coordinates of the four corners of the square.
fn square_corners(origin: TexCoord, side_length: f64) -> Vec<TexCoord> {
    let TexCoord(x, y) = origin;
    vec![
        TexCoord(x, y),                            // Bottom-left
        TexCoord(x + side_length, y),              // Bottom-right
        TexCoord(x + side_length, y + side_length), // Top-right
        TexCoord(x, y + side_length),              // Top-left
    ]
}

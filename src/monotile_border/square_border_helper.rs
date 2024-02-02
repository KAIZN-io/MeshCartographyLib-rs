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

use crate::monotile_border::monotile_border_trait::MonotileBorder;
use crate::mesh_definition::TexCoord;
use tri_mesh::VertexID;

pub struct SquareBorderHelper;

impl MonotileBorder for SquareBorderHelper {
    fn distribute_vertices_around_monotile(&self, boundary_vertices: &[VertexID], side_length: f64, tolerance: f64, total_length: f64) -> Vec<TexCoord> {
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

    fn corners(&self, origin: TexCoord, side_length: f64) -> Vec<TexCoord> {
        let TexCoord(x, y) = origin;
        vec![
            TexCoord(x, y),                            // Bottom-left
            TexCoord(x + side_length, y),              // Bottom-right
            TexCoord(x + side_length, y + side_length), // Top-right
            TexCoord(x, y + side_length),              // Top-left
        ]
    }
}

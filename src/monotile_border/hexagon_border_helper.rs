// hexagon_border_helper.rs

use crate::monotile_border::monotile_border_trait::MonotileBorder;
use crate::mesh_definition::TexCoord;
use tri_mesh::VertexID;
use std::f64::consts::PI;

pub struct HexagonBorderHelper;

impl HexagonBorderHelper {
    fn map_to_hexagon(&self, l: f64, side_length: f64) -> TexCoord {
        let angle = PI / 3.0; // 60 degrees

        // Calculate the corner positions
        let corners = (0..6).map(|i| {
            let x = side_length * (i as f64 * angle).cos();
            let y = side_length * (i as f64 * angle).sin();
            TexCoord(x, y)
        }).collect::<Vec<_>>();

        let mut sum = 0.0;
        for i in 0..corners.len() {
            let corner = &corners[i];
            let next_corner = &corners[(i + 1) % corners.len()];
            if l <= sum + side_length {
                let frac = (l - sum) / side_length;
                return TexCoord(
                    corner.0 * (1.0 - frac) + next_corner.0 * frac,
                    corner.1 * (1.0 - frac) + next_corner.1 * frac,
                );
            }
            sum += side_length;
        }
        TexCoord(0.0, 0.0)
    }
}

impl MonotileBorder for HexagonBorderHelper {
    fn distribute_vertices_around_monotile(&self, boundary_vertices: &[VertexID], side_length: f64, tolerance: f64, total_length: f64) -> Vec<TexCoord> {
        let n = boundary_vertices.len();
        let step_size = total_length / n as f64;
        let mut vertices = Vec::new();

        let mut l = 0.0;
        for _ in 0..n {
            let tex_coord = self.map_to_hexagon(l, side_length);

            // Apply tolerance
            let adjusted_tex_coord = TexCoord(
                if tex_coord.0.abs() < tolerance { 0.0 } else { tex_coord.0 },
                if tex_coord.1.abs() < tolerance { 0.0 } else { tex_coord.1 },
            );

            vertices.push(adjusted_tex_coord);
            l += step_size;
        }

        vertices
    }

    fn corners(&self, origin: TexCoord, side_length: f64) -> Vec<TexCoord> {
        let angle = PI / 3.0; // 60 degrees
        (0..6).map(|i| {
            let x = origin.0 + side_length * (i as f64 * angle).cos();
            let y = origin.1 + side_length * (i as f64 * angle).sin();
            TexCoord(x, y)
        }).collect()
    }
}


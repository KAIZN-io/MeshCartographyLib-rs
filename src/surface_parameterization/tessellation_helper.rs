//! # Create the tesselation of the UV mesh monotile
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Dec-14
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
use nalgebra::{DMatrix, DVector, Vector2, Matrix2, SVD};

// ! MOVE it to the monotile_border module
fn create_twin_border_map(corner_count: usize, current_border: usize) -> HashMap<usize, usize> {
    let mut twin_border_map = HashMap::new();

    for i in 0..(corner_count - 1) {
        twin_border_map.insert(i, current_border - i);
        twin_border_map.insert(current_border - i, i);
    }

    twin_border_map
}

pub struct Tessellation {
    border_v_map: HashMap<usize, Vec<VertexID>>,
    border_map: HashMap<usize, Vec<TexCoord>>,
}

impl Tessellation {
    pub fn new(border_v_map: HashMap<usize, Vec<VertexID>>, border_map: HashMap<usize, Vec<TexCoord>>) -> Self {
        Tessellation {
            border_v_map,
            border_map,
        }
    }

    pub fn rotate_and_shift_mesh(&self, mesh: &mut tri_mesh::Mesh, angle_degrees: f64, docking_side: usize) {
        let angle_radians = angle_degrees.to_radians();

        // Get the border of the mesh
        let main_border = self.border_map.get(&docking_side).unwrap();
        let connection_side = self.border_map.get(&docking_side).unwrap();

        // Get the border of the mesh and convert it to Vec<Vector2<f64>>
        if let Some(main_border_coords) = self.border_map.get(&docking_side) {
            let mut main_border: Vec<Vector2<f64>> = main_border_coords
                .iter()
                .map(|coord| Vector2::new(coord.0, coord.1))
                .collect();

            // Now main_border is of the correct type for order_data
            self.order_data(&mut main_border);

            // Pre-rotation and thresholding
            if let Some(connection_side_coords) = self.border_map.get(&docking_side) {
                let mut vec = Vec::new();
                for pt_2d in connection_side_coords {
                    let mut transformed_2d = self.custom_rotate(Vector2::new(pt_2d.0, pt_2d.1), angle_radians);
                    vec.push(transformed_2d);
                }

                // Order the data of the rotated connection side
                self.order_data(&mut vec);

                println!("");
                println!("main_border: {:?}", main_border);
                println!("");
                println!("vec: {:?}", vec);

                // Calculate shifts
                let shift_x_coordinates = main_border[0].x - vec[0].x;
                let shift_y_coordinates = main_border[0].y - vec[0].y;

                for v in mesh.vertex_iter() {
                    let pt_3d = mesh.position(v);
                    let pt_2d = Vector2::new(pt_3d.x, pt_3d.y);
                    let mut transformed_2d = self.custom_rotate(pt_2d, angle_radians);

                    let x = transformed_2d.x + shift_x_coordinates;
                    let y = transformed_2d.y + shift_y_coordinates;
                    mesh.set_vertex_position(v, tri_mesh::vec3(x, y, 0.0));
                }
            }
        } else {
            panic!("The docking side {} is not found in border_map", docking_side);
        }
    }

    pub fn add_mesh(&mut self, mesh: &mut Mesh, mesh_original: &mut Mesh, docking_side: usize) {
        // A map to relate old vertex IDs in `mesh` to new ones in `mesh_original`
        let mut reindexed_vertices = HashMap::new();

        let current_border = 3;
        let corner_count = 4;
        let twin_border_map = create_twin_border_map(corner_count, current_border);

        for v in mesh.vertex_iter() {
            // let kachelmuster_twin_v = &mut self.equivalent_vertices[v.idx()];  // ! Fix this

            let border_list = &self.border_v_map[&twin_border_map[&docking_side]];

            let mut pt_3d = tri_mesh::vec3(0.0, 0.0, 0.0);
            if let Some(index) = border_list.iter().position(|&vertex| vertex == v) {
                pt_3d = mesh.position(border_list[index]);
            } else {
                pt_3d = mesh.position(v);
            }

            // Check if the vertex already exists in the mesh
            let existing_v = self.find_vertex_by_coordinates(mesh_original, pt_3d);

            let shifted_v = match existing_v {
                Some(vertex_id) => vertex_id,
                None => {
                    let new_vertex_id = mesh_original.add_vertex(pt_3d); // This is hypothetical
                    // kachelmuster_twin_v.push(new_vertex_id.idx());
                    new_vertex_id
                }
            };

            reindexed_vertices.insert(v, shifted_v);
        }

        // Add faces from the rotated mesh to the original mesh
        for face in mesh.face_iter() {
            let (vertex1, vertex2, vertex3) = mesh.face_vertices(face);

            let v1 = reindexed_vertices[&vertex1];
            let v2 = reindexed_vertices[&vertex2];
            let v3 = reindexed_vertices[&vertex3];

            mesh_original.add_face(v1, v2, v3);
        }
    }

    pub fn calculate_angle(&self, border1: &[Vector2<f64>], border2: &[Vector2<f64>]) -> f64 {
        let dir1 = self.fit_line(border1);
        let dir2 = self.fit_line(border2);

        let dot = dir1.dot(&dir2);
        let det = dir1.x * dir2.y - dir1.y * dir2.x;

        let angle = det.atan2(dot);
        let angle_in_degrees = angle * (180.0 / std::f64::consts::PI);

        // Normalize to [0, 360)
        if angle_in_degrees < 0.0 {
            angle_in_degrees + 360.0
        } else {
            angle_in_degrees
        }
    }

    fn fit_line(&self, points: &[Vector2<f64>]) -> Vector2<f64> {
        let n = points.len() as f64;
        let mean = points.iter().sum::<Vector2<f64>>() / n;

        let mut cov = Matrix2::zeros();
        for p in points {
            let centered = p - mean;
            cov += centered * centered.transpose();
        }
        cov /= n;

        // Find the eigenvector of the covariance matrix corresponding to the largest eigenvalue
        let svd = SVD::new(cov, true, true);
        svd.v_t.unwrap().column(0).into()
    }

    fn custom_rotate(&self, pt: Vector2<f64>, angle_radians: f64) -> Vector2<f64> {
        let cos_theta = angle_radians.cos();
        let sin_theta = angle_radians.sin();

        let mut x_prime = pt.x * cos_theta - pt.y * sin_theta;
        let mut y_prime = pt.x * sin_theta + pt.y * cos_theta;

        // Adjust precision
        x_prime = format!("{:.6}", x_prime).parse().unwrap();
        y_prime = format!("{:.6}", y_prime).parse().unwrap();

        // Apply threshold
        let threshold = 1e-10;
        if x_prime.abs() < threshold {
            x_prime = 0.0;
        }
        if y_prime.abs() < threshold {
            y_prime = 0.0;
        }

        Vector2::new(x_prime, y_prime)
    }

    fn order_data(&self, vec: &mut Vec<Vector2<f64>>) {
        let size = vec.len();
        let mut x = DVector::from_iterator(size, vec.iter().map(|v| v.x));
        let y = DVector::from_iterator(size, vec.iter().map(|v| v.y));

        // Creating the design matrix for linear regression
        let a = DVector::from_element(size, 1.0);
        let b = DMatrix::from_columns(&[x.clone(), a]);

        // Perform linear regression using SVD
        let svd = b.svd(true, true);
        let coeffs = svd.solve(&y, std::f64::EPSILON).unwrap();
        let m = coeffs[0];
        let coeff_b = coeffs[1]; // Renamed to avoid conflict with 'b' in the closure

        // Check if all x-values are the same (vertical line)
        let vertical_line = x.max() - x.min() < std::f64::EPSILON;

        // Sort the vector based on the parameter t
        vec.sort_by(|a, b| {
            if vertical_line {
                a.y.partial_cmp(&b.y).unwrap()
            } else {
                let ta = (a.x + m * (a.y - coeff_b)) / (1.0 + m * m).sqrt();
                let tb = (b.x + m * (b.y - coeff_b)) / (1.0 + m * m).sqrt();
                ta.partial_cmp(&tb).unwrap()
            }
        });
    }

    fn find_vertex_by_coordinates(&self, mesh: &Mesh, pt: tri_mesh::Vec3) -> Option<VertexID> {
        for vertex_id in mesh.vertex_iter() {
            if mesh.position(vertex_id) == pt {
                return Some(vertex_id);
            }
        }
        None
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use nalgebra::Vector2;
    use std::collections::HashMap;

    // Helper function to create dummy data for initialization
    fn create_dummy_data() -> (HashMap<usize, Vec<VertexID>>, HashMap<usize, Vec<TexCoord>>) {
        let border_v_map = HashMap::new();
        let border_map = HashMap::new();
        (border_v_map, border_map)
    }

    #[test]
    fn test_order_data() {
        let (border_v_map, border_map) = create_dummy_data();
        let tessellation = Tessellation::new(border_v_map, border_map);

        let mut points = vec![
            Vector2::new(1.0, 2.0),
            Vector2::new(2.0, 3.0),
            Vector2::new(0.5, 1.5),
        ];

        // Call the order_data function
        tessellation.order_data(&mut points);

        // Check if the points are sorted by x-coordinate:
        assert!(points.windows(2).all(|w| w[0].x <= w[1].x));
    }

    // #[test]
    // fn test_fit_line() {
    //     let (border_v_map, border_map) = create_dummy_data();
    //     let tessellation = Tessellation::new(border_v_map, border_map);
    //     let points = vec![
    //         Vector2::new(1.0, 2.0),
    //         Vector2::new(2.0, 3.0),
    //         Vector2::new(3.0, 4.0),
    //     ];
    //     let fitted_line = tessellation.fit_line(&points);
    //     // Expect the line to have a certain direction (example values)
    //     assert!((fitted_line.x - 0.7).abs() < 1e-5);
    //     assert!((fitted_line.y - 0.7).abs() < 1e-5);
    // }

    // #[test]
    // fn test_calculate_angle_90_degrees() {
    //     let tessellation = Tessellation;
    //     let border1 = vec![Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0)];
    //     let border2 = vec![Vector2::new(0.0, 0.0), Vector2::new(0.0, 1.0)];
    //     let angle = tessellation.calculate_angle(&border1, &border2);
    //     // Expect angle to be close to 90 degrees
    //     assert!((angle - 90.0).abs() < 1e-5);
    // }

    // #[test]
    // fn test_calculate_angle_180_degrees() {
    //     let tessellation = Tessellation;
    //     let border1 = vec![Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0)];
    //     let border2 = vec![Vector2::new(0.0, 0.0), Vector2::new(-1.0, 0.0)];
    //     let angle = tessellation.calculate_angle(&border1, &border2);
    //     // Expect angle to be close to 180 degrees
    //     assert!((angle - 180.0).abs() < 1e-5);
    // }

    // #[test]
    // fn test_calculate_angle_45_degrees() {
    //     let tessellation = Tessellation;
    //     let border1 = vec![Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0)];
    //     let border2 = vec![Vector2::new(0.0, 0.0), Vector2::new(1.0, 1.0)];
    //     let angle = tessellation.calculate_angle(&border1, &border2);
    //     // Expect angle to be close to 45 degrees

    //     println!("angle: {}", angle);
    //     assert!((angle - 45.0).abs() < 1e-5);
    // }

    #[test]
    fn test_custom_rotate_90_degrees() {
        let (border_v_map, border_map) = create_dummy_data();
        let tessellation = Tessellation::new(border_v_map, border_map);
        let point = Vector2::new(1.0, 0.0);
        let rotated_point = tessellation.custom_rotate(point, PI / 2.0); // Rotate 90 degrees
        assert!((rotated_point.x.abs() < 1e-5) && ((rotated_point.y - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_custom_rotate_180_degrees() {
        let (border_v_map, border_map) = create_dummy_data();
        let tessellation = Tessellation::new(border_v_map, border_map);
        let point = Vector2::new(1.0, 0.0);
        let rotated_point = tessellation.custom_rotate(point, PI); // Rotate 180 degrees
        assert!(((rotated_point.x + 1.0).abs() < 1e-5) && (rotated_point.y.abs() < 1e-5));
    }

    #[test]
    fn test_custom_rotate_45_degrees() {
        let (border_v_map, border_map) = create_dummy_data();
        let tessellation = Tessellation::new(border_v_map, border_map);
        let point = Vector2::new(1.0, 0.0);
        let rotated_point = tessellation.custom_rotate(point, PI / 4.0); // Rotate 45 degrees
        let sqrt2_over_2 = (2.0f64).sqrt() / 2.0;
        assert!(((rotated_point.x - sqrt2_over_2).abs() < 1e-5) && ((rotated_point.y - sqrt2_over_2).abs() < 1e-5));
    }
}

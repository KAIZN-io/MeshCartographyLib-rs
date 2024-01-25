use tri_mesh::{Mesh, VertexID, FaceID, Vec3};
use nalgebra as na;
use pathfinding::prelude::dijkstra;

pub struct MeshAnalysis {
    mesh: Mesh,
}

impl MeshAnalysis {
    pub fn new(mesh: Mesh) -> Self {
        MeshAnalysis { mesh }
    }

    pub fn get_gaussian_cutline(&self) -> Vec<tri_mesh::HalfEdgeID> {
        let (vertex_id, second_highest) = self.get_two_highest_cones();

        let successors = |&node: &usize| -> Vec<(usize, i32)> {
            self.mesh.edge_iter()
                .filter_map(|edge| {
                    let (start, end) = self.mesh.edge_vertices(edge);

                    let start_idx = *start as usize;
                    let end_idx = *end as usize;

                    if start_idx == node {
                        Some((end_idx, 1))  // Assuming a weight of 1 for simplicity
                    } else if end_idx == node {
                        Some((start_idx, 1)) // Assuming a weight of 1 for simplicity
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Use Dijkstra's algorithm from the pathfinding crate
        let start_node = *vertex_id as usize;
        let end_node = *second_highest as usize;
        let result = dijkstra(&start_node, successors, |&p| p == end_node);

        let mut edges: Vec<tri_mesh::HalfEdgeID> = Vec::new();

        // Check if a path was found
        if let Some((path, _cost)) = result {
            // Iterate over the path to get each pair of vertices
            for window in path.windows(2) {
                if let [v1, v2] = *window {
                    // Find the edge connecting v1 and v2
                    let v1_id = self.mesh.vertex_iter().nth(v1).unwrap();
                    let v2_id = self.mesh.vertex_iter().nth(v2).unwrap();
                    let edge = self.mesh.connecting_edge(v2_id, v1_id).unwrap();
                    edges.push(edge);
                }
            }

            const TWIN_EDGE_COUNT: usize = 2;
            while edges.len() % TWIN_EDGE_COUNT != 0 {
                edges.pop();
            }

            assert_eq!(edges.len() % TWIN_EDGE_COUNT, 0);
            assert!(edges.len() > 0);

        } else {
            println!("No path found between the vertices.");
        }

        edges
    }

    pub fn calculate_gaussian_curvature(&self) -> Vec<f64> {
        let mut curvatures = vec![0.0; self.mesh.no_vertices()];

        for vertex_id in self.mesh.vertex_iter() {
            let mut angle_sum = 0.0;
            let mut area_sum = 0.0;

            for face_id in self.mesh.face_iter() {
                let face = self.mesh.face_positions(face_id);

                if self.face_contains_vertex(face_id, vertex_id) {
                    let angles = self.calculate_angles_for_face(face);
                    angle_sum += angles[self.vertex_id_to_index_in_face_positions(vertex_id, face_id)];

                    area_sum += self.mesh.face_area(face_id) / 3.0; // Assuming Voronoi area
                }
            }

            let index_as_u32: u32 = *vertex_id; // Dereference VertexID to u32
            let index_as_usize: usize = index_as_u32 as usize; // Cast to usize
            curvatures[index_as_usize] = (2.0 * std::f64::consts::PI - angle_sum) / area_sum;
        }

        curvatures
    }

    pub fn get_two_highest_cones(&self) -> (VertexID, VertexID) {
        let curvatures = self.calculate_gaussian_curvature();
        let mut max_curvature = f64::MIN;
        let mut second_max_curvature = f64::MIN;
        let mut vertex_id_with_max_curvature = self.mesh.vertex_iter().next().unwrap();
        let mut vertex_id_with_second_max_curvature = self.mesh.vertex_iter().next().unwrap();

        for (i, &curvature) in curvatures.iter().enumerate() {
            let current_vertex_id = self.mesh.vertex_iter().nth(i).unwrap();

            if curvature > max_curvature {
                // Update second highest before updating the highest
                second_max_curvature = max_curvature;
                vertex_id_with_second_max_curvature = vertex_id_with_max_curvature;

                // Update the highest
                max_curvature = curvature;
                vertex_id_with_max_curvature = current_vertex_id;
            } else if curvature > second_max_curvature {
                // Update the second highest
                second_max_curvature = curvature;
                vertex_id_with_second_max_curvature = current_vertex_id;
            }
        }

        (vertex_id_with_max_curvature, vertex_id_with_second_max_curvature)
    }

    fn face_contains_vertex(&self, face_id: FaceID, vertex_id: VertexID) -> bool {
        let (v1, v2, v3) = self.mesh.face_vertices(face_id);
        [v1, v2, v3].contains(&vertex_id)
    }

    fn vertex_id_to_index_in_face_positions(&self, vertex_id: VertexID, face_id: FaceID) -> usize {
        let (v1, v2, v3) = self.mesh.face_vertices(face_id);
        [v1, v2, v3].iter().position(|&v| v == vertex_id).unwrap()
    }

    fn calculate_angles_for_face(&self, face_positions: (Vec3, Vec3, Vec3)) -> [f64; 3] {
        let (pos1, pos2, pos3) = face_positions;

        let vertices = [
            na::Vector3::new(pos1.x, pos1.y, pos1.z),
            na::Vector3::new(pos2.x, pos2.y, pos2.z),
            na::Vector3::new(pos3.x, pos3.y, pos3.z),
        ];

        let edge_lengths = [
            (vertices[1] - vertices[2]).norm(),
            (vertices[2] - vertices[0]).norm(),
            (vertices[0] - vertices[1]).norm(),
        ];

        [
            self.calculate_angle(edge_lengths[1], edge_lengths[2], edge_lengths[0]),
            self.calculate_angle(edge_lengths[2], edge_lengths[0], edge_lengths[1]),
            self.calculate_angle(edge_lengths[0], edge_lengths[1], edge_lengths[2]),
        ]
    }

    fn calculate_angle(&self, a: f64, b: f64, c: f64) -> f64 {
        // Check if the sides form a valid triangle
        if a + b <= c || a + c <= b || b + c <= a {
            return f64::NAN;
        }

        let cos_angle = ((b.powi(2) + c.powi(2) - a.powi(2)) / (2.0 * b * c)).clamp(-1.0, 1.0);
        cos_angle.acos()
    }

}



#[cfg(test)]
mod tests {
    use crate::io;
    use tri_mesh::Vec3;

    #[test]
    fn test_calculate_angles_for_known_triangle() {
        let mesh = io::load_test_mesh_closed();
        let mesh_analysis = super::MeshAnalysis::new(mesh);

        // Right-angled triangle (3-4-5 triangle)
        let face_positions = (
            Vec3::new(0.0, 0.0, 0.0),  // Vertex 1
            Vec3::new(4.0, 0.0, 0.0),  // Vertex 2
            Vec3::new(0.0, 3.0, 0.0),  // Vertex 3
        );

        let angles = mesh_analysis.calculate_angles_for_face(face_positions);

        // Check that angles match (right angle and two other angles)
        assert!((angles[0] - 0.643501).abs() < 1e-5); // Angle at vertex 1
        assert!((angles[1] - 0.927295).abs() < 1e-5); // Angle at vertex 2 (right angle)
        assert!((angles[2] - 1.570796).abs() < 1e-5); // Angle at vertex 3
    }

    #[test]
    fn test_calculate_angles_for_equilateral_triangle() {
        let mesh = io::load_test_mesh_closed();
        let mesh_analysis = super::MeshAnalysis::new(mesh);

        // Equilateral triangle (sides of length 1)
        let face_positions = (
            Vec3::new(0.0, 0.0, 0.0),   // Vertex 1
            Vec3::new(1.0, 0.0, 0.0),   // Vertex 2
            Vec3::new(0.5, 0.866, 0.0), // Vertex 3 (height of equilateral triangle)
        );

        let angles = mesh_analysis.calculate_angles_for_face(face_positions);

        // All angles should be π/3 (or 60 degrees)
        assert!((angles[0] - std::f64::consts::PI / 3.0).abs() < 1e-4);
        assert!((angles[1] - std::f64::consts::PI / 3.0).abs() < 1e-4);
        assert!((angles[2] - std::f64::consts::PI / 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_calculate_angles_for_isosceles_triangle() {
        let mesh = io::load_test_mesh_closed();
        let mesh_analysis = super::MeshAnalysis::new(mesh);

        // Isosceles Triangle
        let face_positions = (
            Vec3::new(0.0, 0.0, 0.0), // Base Vertex 1
            Vec3::new(3.0, 0.0, 0.0), // Base Vertex 2
            Vec3::new(1.5, f64::sqrt(2.25), 0.0), // Apex
        );

        let angles = mesh_analysis.calculate_angles_for_face(face_positions);

        assert!((angles[0] - 0.7853981633974483).abs() < 1e-6); // Base Vertex 1
        assert!((angles[1] - 1.5707963267948968).abs() < 1e-6); // Apex
        assert!((angles[2] - 0.7853981633974483).abs() < 1e-6); // Base Vertex 2
    }

    #[test]
    fn test_calculate_angles_for_scalene_triangle() {
        let mesh = io::load_test_mesh_closed();
        let mesh_analysis = super::MeshAnalysis::new(mesh);

        // Scalene Triangle
        let face_positions = (
            Vec3::new(0.0, 0.0, 0.0), // Vertex 1
            Vec3::new(4.0, 0.0, 0.0), // Vertex 2
            Vec3::new(2.0, f64::sqrt(3.0), 0.0), // Vertex 3
        );

        let angles = mesh_analysis.calculate_angles_for_face(face_positions);

        assert!((angles[0] - 0.7137243789447657).abs() < 1e-6); // Vertex 1
        assert!((angles[1] - 1.714143895700262).abs() < 1e-6); // Vertex 2
        assert!((angles[2] - 0.7137243789447657).abs() < 1e-6); // Vertex 3
    }

    #[test]
    fn test_calculate_angle() {
        let mesh = io::load_test_mesh_closed();
        let mesh_analysis = super::MeshAnalysis::new(mesh);

        // Test with a known triangle (e.g., 3-4-5 triangle)
        let angle = mesh_analysis.calculate_angle(3.0, 4.0, 5.0);
        assert!((angle - 0.643501).abs() < 1e-6);

        // Test with impossible triangle (should handle gracefully)
        let angle = mesh_analysis.calculate_angle(10.0, 1.0, 1.0);
        assert!(angle.is_nan());
    }

    #[test]
    fn test_gaussian_curvature() {
        let mesh = io::load_test_mesh_closed();
        let mesh_analysis = super::MeshAnalysis::new(mesh);
        let curvatures = mesh_analysis.calculate_gaussian_curvature();

        // for (i, &curvature) in curvatures.iter().enumerate() {
        //     println!("Vertex {:?} has curvature {}", mesh_analysis.mesh.position(mesh_analysis.mesh.vertex_iter().nth(i).unwrap()), curvature);
        // }
    }

    #[test]
    fn test_max_curvature() {
        let mesh = io::load_test_mesh_closed();
        let mesh_analysis = super::MeshAnalysis::new(mesh);
        let edges = mesh_analysis.get_gaussian_cutline();

        assert_eq!(edges.len(), 48);
    }
}

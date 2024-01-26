use tri_mesh::{Mesh, VertexID, FaceID, Vec3};
use nalgebra as na;

pub struct CurvatureAnalyzer {
    mesh: Mesh,
}

impl CurvatureAnalyzer {
    pub fn new(mesh: Mesh) -> Self {
        CurvatureAnalyzer { mesh }
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
        let mesh_analysis = super::CurvatureAnalyzer::new(mesh);

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
        let mesh_analysis = super::CurvatureAnalyzer::new(mesh);

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
        let mesh_analysis = super::CurvatureAnalyzer::new(mesh);

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
        let mesh_analysis = super::CurvatureAnalyzer::new(mesh);

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
        let mesh_analysis = super::CurvatureAnalyzer::new(mesh);

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
        let mesh_analysis = super::CurvatureAnalyzer::new(mesh);
        let curvatures = mesh_analysis.calculate_gaussian_curvature();

        // for (i, &curvature) in curvatures.iter().enumerate() {
        //     println!("Vertex {:?} has curvature {}", mesh_analysis.mesh.position(mesh_analysis.mesh.vertex_iter().nth(i).unwrap()), curvature);
        // }
    }
}

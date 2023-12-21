use tri_mesh::{Mesh, VertexID, FaceID, Vec3};
use nalgebra as na;

pub struct MeshAnalysis {
    mesh: Mesh,
}

impl MeshAnalysis {
    pub fn new(mesh: Mesh) -> Self {
        MeshAnalysis { mesh }
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
        let cos_angle = (b.powi(2) + c.powi(2) - a.powi(2)) / (2.0 * b * c);
        cos_angle.acos()
    }
}

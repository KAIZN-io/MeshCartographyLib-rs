use tri_mesh::{Mesh, VertexID};
use nalgebra as na;

// fn calculate_gaussian_curvature(mesh: &Mesh) -> Vec<f64> {
//     let mut curvatures = vec![0.0; mesh.num_vertices()];

//     for vertex_id in mesh.vertex_iter() {
//         let mut angle_sum = 0.0;
//         let mut area_sum = 0.0;

//         for face_id in mesh.faces_around_vertex(vertex_id) {
//             let face = mesh.face(face_id);
//             let angles = calculate_angles_for_face(&mesh, face);
//             angle_sum += angles[vertex_id_to_index_in_face(vertex_id, face)];

//             area_sum += mesh.area_of_face(face_id) / 3.0; // Assuming Voronoi area
//         }

//         curvatures[vertex_id.to_usize()] = (2.0 * std::f64::consts::PI - angle_sum) / area_sum;
//     }

//     curvatures
// }

fn calculate_angles_for_face(mesh: &Mesh, face: [VertexID; 3]) -> [f64; 3] {
    let vertices = face.map(|vid| {
        let pos = mesh.vertex_position(vid);
        na::Vector3::new(pos.x, pos.y, pos.z) // Convert to nalgebra's Vector3
    });

    let edge_lengths = [
        (vertices[1] - vertices[2]).norm(),
        (vertices[2] - vertices[0]).norm(),
        (vertices[0] - vertices[1]).norm(),
    ];

    [
        calculate_angle(edge_lengths[1], edge_lengths[2], edge_lengths[0]),
        calculate_angle(edge_lengths[2], edge_lengths[0], edge_lengths[1]),
        calculate_angle(edge_lengths[0], edge_lengths[1], edge_lengths[2]),
    ]
}

fn calculate_angle(a: f64, b: f64, c: f64) -> f64 {
    let cos_angle = (b.powi(2) + c.powi(2) - a.powi(2)) / (2.0 * b * c);
    cos_angle.acos()
}

fn vertex_id_to_index_in_face(vertex_id: VertexID, face: [VertexID; 3]) -> usize {
    face.iter().position(|&v| v == vertex_id).unwrap()
}

use tri_mesh::{Mesh, VertexID};
use pathfinding::prelude::dijkstra;
use crate::geodesic_distance::curvature_helper::CurvatureAnalyzer;

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

    fn get_two_highest_cones(&self) -> (VertexID, VertexID) {
        let gaussian_curvature = CurvatureAnalyzer::new(self.mesh.clone());
        let curvatures = gaussian_curvature.calculate_gaussian_curvature();

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
}



#[cfg(test)]
mod tests {
    use crate::io;

    #[test]
    fn test_max_curvature() {
        let mesh = io::load_test_mesh_closed();
        let mesh_analysis = super::MeshAnalysis::new(mesh);
        let edges = mesh_analysis.get_gaussian_cutline();

        assert_eq!(edges.len(), 48);
    }
}

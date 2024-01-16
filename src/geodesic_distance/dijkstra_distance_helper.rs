use pathfinding::prelude::dijkstra;
use tri_mesh::Mesh;

pub fn calculate_all_mesh_distances(mesh: Mesh) -> Vec<Vec<Option<i32>>> {
    let num_vertices = mesh.vertex_iter().count();
    let mut distance_matrix = vec![vec![None; num_vertices]; num_vertices];

    for i in 0..num_vertices {

        let successors = |&node: &usize| -> Vec<(usize, i32)> {
            mesh.edge_iter()
                .filter_map(|edge| {
                    let (start, end) = mesh.edge_vertices(edge);

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

        println!("Calculating distances from vertex {}...", i);
        for j in 0..num_vertices {
            if i != j {
                let result = dijkstra(&i, successors.clone(), |&p| p == j);
                if let Some((_path, cost)) = result {
                    distance_matrix[i][j] = Some(cost);
                }
            }
        }
        println!("Done!");
    }

    distance_matrix
}

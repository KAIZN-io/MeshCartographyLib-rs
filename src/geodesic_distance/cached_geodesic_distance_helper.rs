// geodesic_distance/cached_geodesic_distance_helper.rs

use crate::io::{save_to_csv, load_from_csv};
use crate::geodesic_distance::dijkstra_distance_helper::calculate_all_mesh_distances;
use std::path::PathBuf;
use std::error::Error;

pub fn get_mesh_distance_matrix(mesh_path: PathBuf) -> Result<Vec<Vec<Option<i32>>>, Box<dyn Error>> {
    let mut cache_path = mesh_path.parent().ok_or("Invalid mesh path")?.to_path_buf();
    cache_path.push("data");
    cache_path.push(format!("{}_distance_matrix_static.csv", mesh_path.file_stem().ok_or("Invalid mesh file name")?.to_str().ok_or("Invalid UTF-8 in file name")?));

    if !cache_path.exists() {
        // Load mesh from file
        let mesh = crate::io::load_mesh_from_obj(mesh_path).unwrap();

        let distance_matrix = calculate_all_mesh_distances(mesh);

        save_to_csv(&distance_matrix, cache_path.clone())?;
    }

    load_from_csv(cache_path)
}
